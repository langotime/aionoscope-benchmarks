from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, time

import torch

from .baselines import (
    BASELINE_LAYER,
    DEFAULT_FEATURE_BATCH_SIZE,
    BaselineSpec,
    collect_split_features,
    get_baseline_spec,
    resolve_baseline_names,
)
from .constants import DATASET_CONFIG_PATH, MODEL_RESULTS_ROOT, PROBE_CONFIG_PATH
from .offline_probe import (
    CollectedProbeFeatures,
    offline_probe_run_linear_multihead_by_layer_multi_val_from_collected,
)
from .probe_metrics import ensure_probe_metric_dependencies_available, probe_compute_metrics
from .results import build_model_result, write_model_result
from .run_model import (
    _format_elapsed_s,
    _load_probe_config,
    _runtime_dataset_batch_size_from_probe_config,
    result_output_path,
)
from .runtime_dataset import build_runtime_splits_by_validation_seed


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log_run(run_label: str, message: str) -> None:
    print(f"[{_utc_timestamp()}] [{run_label}] {message}", file=sys.stderr, flush=True)


def _run_label(*, baseline_name: str, channel_size: int, num_enabled: int) -> str:
    return f"{baseline_name} | L={int(channel_size)} | num_enabled={int(num_enabled)}"


def baseline_slug(*, spec: BaselineSpec, channel_size: int) -> str:
    return f"Baseline-{spec.slug}-L{int(channel_size)}"


def _baseline_model_metadata(*, spec: BaselineSpec, channel_size: int) -> dict[str, object]:
    return {
        "family": f"Baseline/{spec.family}",
        "checkpoint_name": f"{spec.name}-L{int(channel_size)}",
        "architecture": {
            "backbone": "metric_floor" if not spec.uses_probe else "fixed_feature_extractor",
        },
        "training": {
            "paradigm": "none",
        },
        "baseline": {
            "name": spec.name,
            "family": spec.family,
            "description": spec.description,
            "uses_probe": bool(spec.uses_probe),
            "is_oracle": bool(spec.is_oracle),
            "feature_dim_declared": spec.feature_dim,
            "synthetic_layer": int(BASELINE_LAYER),
        },
    }


def _baseline_adapter_metadata(
    *,
    spec: BaselineSpec,
    channel_size: int,
    feature_batch_size: int,
    feature_dim: int | None,
) -> dict[str, object]:
    return {
        "env": "baseline",
        "encode_batch_size": int(feature_batch_size),
        "cpu_feature_cache_dtype": "float32",
        "benchmark_sequence_length": int(channel_size),
        "benchmark_sequence_length_source": "baseline.channel_size",
        "input_length_policy": "exact",
        "parameter_count": 0,
        "parameter_count_total": 0,
        "trainable_parameter_count": 0,
        "parameter_count_source": "not_applicable",
        "parameter_count_prefix_by_layer": {str(int(BASELINE_LAYER)): 0},
        "parameter_count_prefix_source": "not_applicable",
        "baseline_name": spec.name,
        "baseline_family": spec.family,
        "baseline_description": spec.description,
        "baseline_uses_probe": bool(spec.uses_probe),
        "baseline_is_oracle": bool(spec.is_oracle),
        "baseline_feature_dim": None if feature_dim is None else int(feature_dim),
    }


def _validation_seed_order(manifest: dict[str, object]) -> list[int]:
    return [int(seed_value) for seed_value in manifest["validation_seed_values"]]


def _validation_seed_to_generator_seed(manifest: dict[str, object]) -> dict[int, int]:
    return {
        int(seed_value): int(generator_seed)
        for seed_value, generator_seed in dict(manifest["validation_seed_to_generator_seed"]).items()
    }


def _nanmean(values: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(values)
    safe = torch.nan_to_num(values, nan=0.0)
    count = valid.sum(dim=0)
    summed = (safe * valid.to(dtype=safe.dtype)).sum(dim=0)
    return torch.where(count > 0, summed / count.clamp_min(1).to(dtype=safe.dtype), torch.zeros_like(summed))


def _dense_metric_floor(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_names: list[str],
) -> dict[str, object]:
    valid = torch.isfinite(targets)
    safe_targets = torch.nan_to_num(targets.to(dtype=torch.float64), nan=0.0)
    safe_predictions = predictions.to(dtype=torch.float64).expand_as(safe_targets)
    valid64 = valid.to(dtype=torch.float64)
    errors = (safe_predictions - safe_targets) * valid64
    count = valid.sum(dim=0).to(dtype=torch.float64)
    if bool(torch.any(count == 0).item()):
        missing = [
            name
            for name, target_count in zip(target_names, count.tolist(), strict=True)
            if int(target_count) == 0
        ]
        raise ValueError("Validation split has no finite dense samples for targets: " + ", ".join(missing))

    sum_sq_error = torch.sum(errors * errors, dim=0)
    sum_abs_error = torch.sum(errors.abs(), dim=0)
    sum_targets = torch.sum(safe_targets * valid64, dim=0)
    sum_target_sq = torch.sum(safe_targets * safe_targets * valid64, dim=0)
    sum_predictions = torch.sum(safe_predictions * valid64, dim=0)
    sum_prediction_sq = torch.sum(safe_predictions * safe_predictions * valid64, dim=0)
    sum_target_prediction = torch.sum(safe_targets * safe_predictions * valid64, dim=0)

    mean_targets = sum_targets / count
    mean_predictions = sum_predictions / count
    ss_tot = sum_target_sq - count * mean_targets * mean_targets
    ss_pred = sum_prediction_sq - count * mean_predictions * mean_predictions
    covariance = sum_target_prediction - count * mean_targets * mean_predictions
    denom = torch.sqrt(ss_tot * ss_pred)

    mse = sum_sq_error / count
    mae = sum_abs_error / count
    r2 = torch.full_like(mse, float("nan"))
    r2_defined = ss_tot > 0
    r2[r2_defined] = 1.0 - (sum_sq_error[r2_defined] / ss_tot[r2_defined])
    pearson = torch.full_like(mse, float("nan"))
    pearson_defined = denom > 0
    pearson[pearson_defined] = covariance[pearson_defined] / denom[pearson_defined]

    return {
        "macro_mse": float(mse.mean()),
        "macro_mae": float(mae.mean()),
        "macro_r2": float(torch.nanmean(r2)),
        "macro_pearson": float(torch.nanmean(pearson)),
        "timings": {
            "regression": {
                "total_s": 0.0,
                "train_shared_s": 0.0,
                "eval_total_s": 0.0,
                "eval_calls": 0,
                "layer_staging_s": 0.0,
                "eval_forward_s": 0.0,
                "eval_finalize_s": 0.0,
                "validation_seed": 0,
                "num_validation_seeds": 0,
            }
        },
        "per_target_mse": {
            name: float(value) for name, value in zip(target_names, mse.tolist(), strict=True)
        },
        "per_target_mae": {
            name: float(value) for name, value in zip(target_names, mae.tolist(), strict=True)
        },
        "per_target_r2": {
            name: float(value) for name, value in zip(target_names, r2.tolist(), strict=True)
        },
        "per_target_pearson": {
            name: float(value) for name, value in zip(target_names, pearson.tolist(), strict=True)
        },
        "per_target_best_step": {name: 0 for name in target_names},
    }


def _pack_classification_metrics(
    *,
    metrics: tuple[float, dict[str, float], float, dict[str, float]],
) -> dict[str, object]:
    macro_auc, per_class_auc, macro_auprc, per_class_auprc = metrics
    return {
        "macro_auc": float(macro_auc),
        "per_class_auc": per_class_auc,
        "macro_auprc": float(macro_auprc),
        "per_class_auprc": per_class_auprc,
        "best_probe_step": 0,
    }


def _metric_floor_probe_results(
    *,
    train: dict[str, torch.Tensor],
    val_splits: dict[int, dict[str, torch.Tensor]],
    class_names: list[str],
    dense_target_names: list[str],
) -> dict[int, dict[str, object]]:
    class_prevalence = train["y_cls"].to(dtype=torch.float32).mean(dim=0).clamp(0.0, 1.0)
    dense_mean = _nanmean(train["y_dense"].to(dtype=torch.float32))
    train_size = int(train["y_cls"].size(0))
    results_by_seed: dict[int, dict[str, object]] = {}
    for seed_value, split in sorted(val_splits.items()):
        val_targets = split["y_cls"].to(dtype=torch.float32)
        predictions = class_prevalence.unsqueeze(0).expand_as(val_targets)
        metrics = probe_compute_metrics(
            targets=val_targets,
            predictions=predictions,
            class_names=class_names,
        )
        best_payload = _pack_classification_metrics(metrics=metrics)
        dense_payload = _dense_metric_floor(
            predictions=dense_mean,
            targets=split["y_dense"].to(dtype=torch.float32),
            target_names=dense_target_names,
        )
        dense_payload["timings"]["regression"]["validation_seed"] = int(seed_value)
        dense_payload["timings"]["regression"]["num_validation_seeds"] = int(len(val_splits))
        dense_payload["feature_dim"] = 1
        dense_payload["train_size"] = int(train_size)
        dense_payload["val_size"] = int(val_targets.size(0))
        dense_payload["val_num_crops"] = 1
        classification_payload = {
            "best_auc": dict(best_payload),
            "best_auprc": dict(best_payload),
            "feature_dim": 1,
            "train_size": int(train_size),
            "val_size": int(val_targets.size(0)),
            "val_num_crops": 1,
            "timings": {
                "classification": {
                    "total_s": 0.0,
                    "train_shared_s": 0.0,
                    "eval_total_s": 0.0,
                    "eval_calls": 0,
                    "layer_staging_s": 0.0,
                    "eval_forward_s": 0.0,
                    "eval_numpy_s": 0.0,
                    "eval_metrics_s": 0.0,
                    "eval_torchmetrics_s": 0.0,
                    "eval_pairwise_confusion_s": 0.0,
                    "validation_seed": int(seed_value),
                    "num_validation_seeds": int(len(val_splits)),
                }
            },
        }
        results_by_seed[int(seed_value)] = {
            "categorical": {int(BASELINE_LAYER): classification_payload},
            "dense": {int(BASELINE_LAYER): dense_payload},
            "dense_targets_available": True,
            "feature_dim": 1,
            "train_size": int(train_size),
            "val_size": int(val_targets.size(0)),
            "val_num_crops": 1,
            "timings": {
                "metric_floor": {
                    "total_s": 0.0,
                    "validation_seed": int(seed_value),
                    "num_validation_seeds": int(len(val_splits)),
                }
            },
        }
    return results_by_seed


def _collected_from_split_features(
    *,
    features: torch.Tensor,
    split: dict[str, torch.Tensor],
    timings: dict[str, float | int],
) -> CollectedProbeFeatures:
    return CollectedProbeFeatures(
        features_by_layer={int(BASELINE_LAYER): features},
        class_targets=split["y_cls"].to(dtype=torch.float32).cpu(),
        dense_targets=split["y_dense"].to(dtype=torch.float32).cpu(),
        has_crops=False,
        timings=timings,
    )


def _feature_probe_results(
    *,
    spec: BaselineSpec,
    train: dict[str, torch.Tensor],
    val_splits: dict[int, dict[str, torch.Tensor]],
    manifest: dict[str, object],
    eval_config,
    device: torch.device,
    feature_batch_size: int,
    feature_seed: int,
    probe_seed: int | None,
    run_label: str,
) -> tuple[dict[int, dict[str, object]], int, dict[str, object]]:
    collect_train_start = perf_counter()
    _log_run(run_label, f"phase: collect train baseline features (batch_size={feature_batch_size})")
    train_features, train_timings = collect_split_features(
        spec=spec,
        split=train,
        manifest=manifest,
        seed=int(feature_seed),
        device=device,
        batch_size=int(feature_batch_size),
    )
    train_collected = _collected_from_split_features(
        features=train_features,
        split=train,
        timings=train_timings,
    )
    collect_train_s = float(perf_counter() - collect_train_start)
    _log_run(
        run_label,
        f"train features ready: samples={train_timings.get('samples')} "
        f"feature_dim={train_features.size(1)} in {_format_elapsed_s(collect_train_s)}",
    )

    val_collected_by_seed: dict[int, CollectedProbeFeatures] = {}
    collect_val_by_seed_s: dict[str, float] = {}
    collect_val_start = perf_counter()
    for index, seed_value in enumerate(_validation_seed_order(manifest), start=1):
        seed_start = perf_counter()
        split = val_splits[int(seed_value)]
        val_features, val_timings = collect_split_features(
            spec=spec,
            split=split,
            manifest=manifest,
            seed=int(feature_seed),
            device=device,
            batch_size=int(feature_batch_size),
        )
        val_collected_by_seed[int(seed_value)] = _collected_from_split_features(
            features=val_features,
            split=split,
            timings=val_timings,
        )
        collect_val_by_seed_s[str(int(seed_value))] = float(perf_counter() - seed_start)
        _log_run(
            run_label,
            f"validation features {index}/{len(val_splits)} seed={int(seed_value)} "
            f"done in {_format_elapsed_s(collect_val_by_seed_s[str(int(seed_value))])}",
        )
    collect_val_total_s = float(perf_counter() - collect_val_start)

    offline_probe_start = perf_counter()
    _log_run(run_label, "phase: run offline probe on synthetic baseline layer 0")
    probe_results_by_validation_seed = offline_probe_run_linear_multihead_by_layer_multi_val_from_collected(
        train_collected=train_collected,
        val_collected_by_seed=val_collected_by_seed,
        num_classes=len(manifest["class_names"]),
        class_names=list(manifest["class_names"]),
        eval_config=eval_config,
        device=device,
        layers_categorical=(int(BASELINE_LAYER),),
        layers_dense=(int(BASELINE_LAYER),),
        layers_confusion=tuple(),
        dense_target_names=list(manifest["dense_target_names"]),
        dense_log_per_target=True,
        probe_seed=probe_seed,
        progress_callback=lambda message: _log_run(run_label, f"probe: {message}"),
    )
    offline_probe_s = float(perf_counter() - offline_probe_start)
    runtime_bits = {
        "collect_train_s": float(collect_train_s),
        "collect_val_total_s": float(collect_val_total_s),
        "collect_val_by_validation_seed_s": collect_val_by_seed_s,
        "offline_probe_s": float(offline_probe_s),
    }
    return probe_results_by_validation_seed, int(train_features.size(1)), runtime_bits


def _write_baseline_result(
    *,
    spec: BaselineSpec,
    channel_size: int,
    num_enabled: int,
    out_dir: Path,
    manifest: dict[str, object],
    probe_config_payload: dict[str, object],
    runtime_summary: dict[str, object],
    probe_results_by_validation_seed: dict[int, dict[str, object]],
    feature_batch_size: int,
    feature_dim: int | None,
    probe_seed: int | None,
) -> Path:
    slug = baseline_slug(spec=spec, channel_size=int(channel_size))
    payload = build_model_result(
        model_name=f"Baseline {spec.name} L{int(channel_size)}",
        model_slug=slug,
        model_type="baseline",
        model_metadata=_baseline_model_metadata(spec=spec, channel_size=int(channel_size)),
        checkpoint=f"builtin:{spec.name}",
        source="aionoscope_benchmarks.baselines",
        import_path="aionoscope_benchmarks.baselines",
        dataset_manifest=manifest,
        probe_config=probe_config_payload,
        layers=[int(BASELINE_LAYER)],
        adapter_metadata=_baseline_adapter_metadata(
            spec=spec,
            channel_size=int(channel_size),
            feature_batch_size=int(feature_batch_size),
            feature_dim=feature_dim,
        ),
        probe_results_by_validation_seed=probe_results_by_validation_seed,
        validation_seed_values=_validation_seed_order(manifest),
        validation_seed_to_generator_seed=_validation_seed_to_generator_seed(manifest),
        probe_seed=probe_seed,
        runtime_summary=runtime_summary,
    )
    out_path = result_output_path(out_dir=out_dir, model_slug=slug, num_enabled=int(num_enabled))
    write_model_result(out_path=out_path, payload=payload)
    return out_path


def run_baselines_for_num_enabled(
    *,
    baseline_names: list[str],
    channel_size: int,
    num_enabled: int,
    dataset_config_path: Path = DATASET_CONFIG_PATH,
    probe_config_path: Path = PROBE_CONFIG_PATH,
    out_dir: Path = MODEL_RESULTS_ROOT,
    device: torch.device | None = None,
    feature_batch_size: int = DEFAULT_FEATURE_BATCH_SIZE,
    train_batches: int | None = None,
    val_batches: int | None = None,
    validation_seed_values: list[int] | None = None,
    validation_seed_offset: int | None = None,
    feature_seed: int = 0,
    probe_seed: int | None = 0,
) -> list[Path]:
    resolved_baseline_names = resolve_baseline_names(baseline_names)
    specs = [get_baseline_spec(name) for name in resolved_baseline_names]
    actual_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ensure_probe_metric_dependencies_available()

    probe_config, probe_config_raw = _load_probe_config(probe_config_path)
    runtime_dataset_batch_size = _runtime_dataset_batch_size_from_probe_config(
        probe_config=probe_config,
        probe_config_raw=probe_config_raw,
    )
    dataset_label = f"baselines L={int(channel_size)} | num_enabled={int(num_enabled)}"
    dataset_build_start = perf_counter()
    _log_run(dataset_label, "phase: build online dataset splits")
    manifest, train, val_splits = build_runtime_splits_by_validation_seed(
        config_path=dataset_config_path,
        device=actual_device,
        batch_size=int(runtime_dataset_batch_size),
        channel_size_override=int(channel_size),
        channel_size_policy_override="baseline_exact",
        channel_size_source_override="baseline.channel_size",
        train_batches=train_batches,
        val_batches=val_batches,
        num_enabled=int(num_enabled),
        validation_seed_values=validation_seed_values,
        validation_seed_offset=validation_seed_offset,
        show_progress_bar=False,
        progress_callback=lambda message: _log_run(dataset_label, f"dataset: {message}"),
    )
    dataset_build_s = float(perf_counter() - dataset_build_start)
    validation_seed_order = _validation_seed_order(manifest)
    _log_run(
        dataset_label,
        f"dataset ready: channel_size={manifest['channel_size']} "
        f"train_batches={manifest['train_batches']} val_batches={manifest['val_batches']} "
        f"validation_seeds={validation_seed_order} in {_format_elapsed_s(dataset_build_s)}",
    )

    out_paths: list[Path] = []
    for spec in specs:
        baseline_start = perf_counter()
        run_started_at_unix = float(time())
        run_label = _run_label(
            baseline_name=spec.name,
            channel_size=int(channel_size),
            num_enabled=int(num_enabled),
        )
        _log_run(run_label, f"start: device={actual_device} out_dir={out_dir}")
        runtime_summary: dict[str, object] = {
            "started_at_unix": float(run_started_at_unix),
            "device": str(actual_device),
            "device_type": str(actual_device.type),
            "process_id": int(os.getpid()),
            "dataset_build_s": float(dataset_build_s),
            "feature_batch_size": int(feature_batch_size),
            "feature_seed": int(feature_seed),
            "layers_evaluated_count": 1,
            "validation_seed_count": int(len(validation_seed_order)),
            "baseline_name": spec.name,
            "baseline_family": spec.family,
            "baseline_uses_probe": bool(spec.uses_probe),
            "baseline_is_oracle": bool(spec.is_oracle),
        }
        if actual_device.type == "cuda":
            cuda_index = actual_device.index
            if cuda_index is None:
                cuda_index = int(torch.cuda.current_device())
            runtime_summary["cuda_device_index"] = int(cuda_index)
            runtime_summary["cuda_device_name"] = str(torch.cuda.get_device_name(cuda_index))

        if spec.uses_probe:
            probe_results_by_validation_seed, feature_dim, extra_runtime = _feature_probe_results(
                spec=spec,
                train=train,
                val_splits=val_splits,
                manifest=manifest,
                eval_config=probe_config,
                device=actual_device,
                feature_batch_size=int(feature_batch_size),
                feature_seed=int(feature_seed),
                probe_seed=probe_seed,
                run_label=run_label,
            )
            runtime_summary.update(extra_runtime)
            result_probe_seed = probe_seed
            probe_config_payload = dict(probe_config_raw)
            probe_config_payload["probe_seed"] = None if probe_seed is None else int(probe_seed)
        else:
            _log_run(run_label, "phase: compute metric-floor predictions")
            metric_start = perf_counter()
            probe_results_by_validation_seed = _metric_floor_probe_results(
                train=train,
                val_splits=val_splits,
                class_names=list(manifest["class_names"]),
                dense_target_names=list(manifest["dense_target_names"]),
            )
            runtime_summary["metric_floor_s"] = float(perf_counter() - metric_start)
            feature_dim = int(spec.feature_dim or 1)
            result_probe_seed = None
            probe_config_payload = dict(probe_config_raw)
            probe_config_payload["probe_seed"] = None

        runtime_summary["finished_at_unix"] = float(time())
        runtime_summary["total_wall_s"] = float(perf_counter() - baseline_start)
        out_path = _write_baseline_result(
            spec=spec,
            channel_size=int(channel_size),
            num_enabled=int(num_enabled),
            out_dir=out_dir,
            manifest=manifest,
            probe_config_payload=probe_config_payload,
            runtime_summary=runtime_summary,
            probe_results_by_validation_seed=probe_results_by_validation_seed,
            feature_batch_size=int(feature_batch_size),
            feature_dim=feature_dim,
            probe_seed=result_probe_seed,
        )
        out_paths.append(out_path)
        _log_run(
            run_label,
            f"done: wrote {out_path.name} in {_format_elapsed_s(runtime_summary['total_wall_s'])}",
        )
    return out_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="append",
        dest="baselines",
        default=None,
        help="Baseline name; can be repeated. Use 'paper-critical' or 'all' for groups.",
    )
    parser.add_argument("--list-baselines", action="store_true", help="List available baselines and exit.")
    parser.add_argument("--channel-size", type=int, required=False, help="Exact sequence length for the run.")
    parser.add_argument("--num-enabled", type=int, required=False, help="Active num_enabled value.")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DATASET_CONFIG_PATH,
        help="Dataset config YAML.",
    )
    parser.add_argument(
        "--probe-config",
        type=Path,
        default=PROBE_CONFIG_PATH,
        help="Probe config YAML.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MODEL_RESULTS_ROOT,
        help="Output directory for baseline JSON artifacts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device for dataset generation, feature extraction, and probes.",
    )
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=DEFAULT_FEATURE_BATCH_SIZE,
        help="Batch size for fixed baseline feature extraction.",
    )
    parser.add_argument("--train-batches", type=int, default=None, help="Optional train batch-count override.")
    parser.add_argument("--val-batches", type=int, default=None, help="Optional validation batch-count override.")
    parser.add_argument(
        "--validation-seed-values",
        "--validation-seed-value",
        action="append",
        dest="validation_seed_values",
        type=int,
        default=None,
        help="Optional validation seed value override; can be repeated.",
    )
    parser.add_argument("--validation-seed-offset", type=int, default=None)
    parser.add_argument("--feature-seed", type=int, default=0)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--no-probe-seed", action="store_true", help="Disable deterministic probe seeding.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_baselines:
        for name in resolve_baseline_names(["all"]):
            spec = get_baseline_spec(name)
            print(f"{spec.name}\t{spec.family}\tuses_probe={spec.uses_probe}\t{spec.description}")
        return
    if not args.baselines:
        raise SystemExit("--baseline is required unless --list-baselines is used")
    if args.channel_size is None:
        raise SystemExit("--channel-size is required")
    if args.num_enabled is None:
        raise SystemExit("--num-enabled is required")
    probe_seed = None if args.no_probe_seed else int(args.probe_seed)
    out_paths = run_baselines_for_num_enabled(
        baseline_names=[str(name) for name in args.baselines],
        channel_size=int(args.channel_size),
        num_enabled=int(args.num_enabled),
        dataset_config_path=args.dataset_config,
        probe_config_path=args.probe_config,
        out_dir=args.out_dir,
        device=torch.device(str(args.device)),
        feature_batch_size=int(args.feature_batch_size),
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        validation_seed_values=args.validation_seed_values,
        validation_seed_offset=args.validation_seed_offset,
        feature_seed=int(args.feature_seed),
        probe_seed=probe_seed,
    )
    for path in out_paths:
        print(path)


if __name__ == "__main__":
    main()
