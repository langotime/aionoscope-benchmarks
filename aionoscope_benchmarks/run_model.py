from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, time

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .constants import DATASET_CONFIG_PATH, MODEL_RESULTS_ROOT, PROBE_CONFIG_PATH
from .model_registry import create_adapter, model_taxonomy
from .offline_probe import (
    OfflineProbeConfig,
    collect_probe_features_by_layer,
    offline_probe_run_linear_multihead_by_layer_multi_val_from_collected,
    probe_learning_rate_scaling_payload,
)
from .probe_metrics import ensure_probe_metric_dependencies_available
from .results import build_model_result, write_model_result
from .runtime_dataset import build_runtime_splits_by_validation_seed, resolve_requested_num_enabled_values


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_elapsed_s(value: float) -> str:
    value = float(value)
    if value < 60.0:
        return f"{value:.1f}s"
    minutes, seconds = divmod(value, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {seconds:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def _log_run(model_name: str, message: str) -> None:
    print(f"[{_utc_timestamp()}] [{model_name}] {message}", file=sys.stderr, flush=True)


def result_filename(*, model_slug: str, num_enabled: int) -> str:
    return f"{str(model_slug)}__num_enabled_{int(num_enabled)}.json"


def result_output_path(*, out_dir: Path, model_slug: str, num_enabled: int) -> Path:
    return out_dir / result_filename(model_slug=model_slug, num_enabled=num_enabled)


def _run_label(model_name: str, num_enabled: int) -> str:
    return f"{model_name} | num_enabled={int(num_enabled)}"


def _require_numeric_timing(
    *,
    timings: dict[str, object],
    key: str,
    context: str,
) -> float:
    value = timings.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Missing numeric timing {key!r} for {context}, got {value!r}")
    return float(value)


def _encoder_forward_runtime_summary(
    *,
    train_collected,
    val_collected_by_seed: dict[int, object],
) -> dict[str, object]:
    train_forward_s = _require_numeric_timing(
        timings=train_collected.timings,
        key="forward_s",
        context="train feature collection",
    )
    val_forward_by_seed_s: dict[str, float] = {}
    val_forward_total_s = 0.0
    for seed_value in sorted(int(seed) for seed in val_collected_by_seed):
        collected = val_collected_by_seed[int(seed_value)]
        forward_s = _require_numeric_timing(
            timings=collected.timings,
            key="forward_s",
            context=f"validation feature collection seed={int(seed_value)}",
        )
        val_forward_by_seed_s[str(int(seed_value))] = float(forward_s)
        val_forward_total_s += float(forward_s)
    return {
        "encoder_forward_train_s": float(train_forward_s),
        "encoder_forward_val_total_s": float(val_forward_total_s),
        "encoder_forward_by_validation_seed_s": val_forward_by_seed_s,
        "encoder_forward_total_s": float(train_forward_s + val_forward_total_s),
    }


def _load_probe_config(path: Path) -> tuple[OfflineProbeConfig, dict[str, object]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML dict, got {type(raw).__name__}")
    learning_rate_scaling = raw.get("learning_rate_scaling", {})
    if learning_rate_scaling is None:
        learning_rate_scaling = {}
    if not isinstance(learning_rate_scaling, dict):
        raise ValueError(
            "learning_rate_scaling must be a YAML dict when provided, "
            f"got {type(learning_rate_scaling).__name__}"
        )
    scaling_enabled = learning_rate_scaling.get("enabled", True)
    if not isinstance(scaling_enabled, bool):
        raise ValueError(
            "learning_rate_scaling.enabled must be a boolean, "
            f"got {scaling_enabled!r}"
        )
    config = OfflineProbeConfig(
        steps=int(raw["steps"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        final_learning_rate=float(raw["final_learning_rate"]),
        learning_rate_warmup_steps=int(raw["learning_rate_warmup_steps"]),
        weight_decay=float(raw["weight_decay"]),
        opt_betas=(float(raw["opt_betas"][0]), float(raw["opt_betas"][1])),
        gradient_clip=float(raw["gradient_clip"]),
        checkpoint_interval=int(raw["checkpoint_interval"]),
        learning_rate_scaling_enabled=bool(scaling_enabled),
        learning_rate_reference_feature_dim=float(
            learning_rate_scaling.get("reference_feature_dim", 1024.0)
        ),
        learning_rate_feature_dim_power=float(learning_rate_scaling.get("feature_dim_power", 3.0)),
        min_learning_rate=float(learning_rate_scaling.get("min_learning_rate", 1.0e-3)),
    )
    return config, raw


def _runtime_dataset_batch_size_from_probe_config(
    *,
    probe_config: OfflineProbeConfig,
    probe_config_raw: dict[str, object],
) -> int:
    value = probe_config_raw.get("runtime_dataset_batch_size", probe_config.batch_size)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            "runtime_dataset_batch_size must be a positive integer when provided, "
            f"got {value!r}"
        )
    return int(value)


def _make_split_loader(
    *,
    split: dict[str, torch.Tensor],
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(split["x"], split["y_cls"], split["y_dense"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def run_single_model_for_num_enabled(
    *,
    model_name: str,
    num_enabled: int,
    dataset_config_path: Path = DATASET_CONFIG_PATH,
    probe_config_path: Path = PROBE_CONFIG_PATH,
    out_dir: Path = MODEL_RESULTS_ROOT,
    device: torch.device | None = None,
    encode_batch_size: int | None = None,
    layers: list[int] | None = None,
    train_batches: int | None = None,
    val_batches: int | None = None,
    validation_seed_values: list[int] | None = None,
    validation_seed_offset: int | None = None,
    probe_seed: int | None = 0,
) -> Path:
    run_label = _run_label(model_name=model_name, num_enabled=num_enabled)
    run_started_at_unix = float(time())
    run_start = perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    actual_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log_run(
        run_label,
        f"start: device={actual_device} dataset_config={dataset_config_path.name} "
        f"probe_config={probe_config_path.name} out_dir={out_dir}",
    )
    runtime_summary: dict[str, object] = {
        "started_at_unix": float(run_started_at_unix),
        "device": str(actual_device),
        "device_type": str(actual_device.type),
        "process_id": int(os.getpid()),
    }
    if actual_device.type == "cuda":
        cuda_index = actual_device.index
        if cuda_index is None:
            cuda_index = int(torch.cuda.current_device())
        runtime_summary["cuda_device_index"] = int(cuda_index)
        runtime_summary["cuda_device_name"] = str(torch.cuda.get_device_name(cuda_index))

    _log_run(run_label, "phase: preflight dependency checks")
    ensure_probe_metric_dependencies_available()
    _log_run(run_label, "preflight ready: probe metric dependencies import successfully")

    _log_run(run_label, "phase: load probe config")
    probe_config, probe_config_raw = _load_probe_config(probe_config_path)
    runtime_dataset_batch_size = _runtime_dataset_batch_size_from_probe_config(
        probe_config=probe_config,
        probe_config_raw=probe_config_raw,
    )
    _log_run(
        run_label,
        f"probe config ready: steps={probe_config.steps} batch_size={probe_config.batch_size} "
        f"runtime_dataset_batch_size={runtime_dataset_batch_size} "
        f"checkpoint_interval={probe_config.checkpoint_interval}",
    )

    adapter_load_start = perf_counter()
    _log_run(run_label, "phase: load adapter")
    spec, adapter = create_adapter(model_name)
    adapter = adapter.to(actual_device)
    adapter.eval()
    resolved_channel_size = adapter.exact_benchmark_sequence_length()
    runtime_summary["adapter_load_s"] = float(perf_counter() - adapter_load_start)
    _log_run(
        run_label,
        f"adapter ready: checkpoint={spec.checkpoint} layers={len(adapter.available_layers)} "
        f"encode_batch_size={adapter.default_encode_batch_size} "
        f"benchmark_sequence_length={resolved_channel_size} "
        f"in {_format_elapsed_s(runtime_summary['adapter_load_s'])}",
    )

    dataset_build_start = perf_counter()
    _log_run(run_label, "phase: build online dataset splits")
    manifest, train, val_splits = build_runtime_splits_by_validation_seed(
        config_path=dataset_config_path,
        device=actual_device,
        batch_size=int(runtime_dataset_batch_size),
        channel_size_override=resolved_channel_size,
        channel_size_policy_override="model_native_exact",
        channel_size_source_override=f"adapter.{adapter.benchmark_sequence_length_source}",
        train_batches=train_batches,
        val_batches=val_batches,
        num_enabled=int(num_enabled),
        validation_seed_values=validation_seed_values,
        validation_seed_offset=validation_seed_offset,
        show_progress_bar=False,
        progress_callback=lambda message: _log_run(run_label, f"dataset: {message}"),
    )
    runtime_summary["dataset_build_s"] = float(perf_counter() - dataset_build_start)
    validation_seed_order = [int(seed_value) for seed_value in manifest["validation_seed_values"]]
    validation_seed_to_generator_seed = {
        int(seed_value): int(generator_seed)
        for seed_value, generator_seed in manifest["validation_seed_to_generator_seed"].items()
    }
    _log_run(
        run_label,
        "dataset ready: "
        f"default_channel_size={manifest['default_channel_size']} "
        f"channel_size={manifest['channel_size']} "
        f"channel_size_policy={manifest['channel_size_policy']} "
        f"num_enabled={manifest['num_enabled']} "
        f"train_batches={manifest['train_batches']} "
        f"val_batches={manifest['val_batches']} "
        f"validation_seeds={validation_seed_order} "
        f"generator_seeds={[validation_seed_to_generator_seed[seed] for seed in validation_seed_order]} "
        f"in {_format_elapsed_s(runtime_summary['dataset_build_s'])}",
    )

    first_seed_value = int(validation_seed_order[0])
    first_val_split = val_splits[first_seed_value]
    adapter_prepare_start = perf_counter()
    _log_run(run_label, "phase: adapter prepare")
    adapter.prepare(manifest=manifest, train_split=train, val_split=first_val_split)
    runtime_summary["adapter_prepare_s"] = float(perf_counter() - adapter_prepare_start)
    _log_run(
        run_label,
        f"adapter prepare done in {_format_elapsed_s(runtime_summary['adapter_prepare_s'])}",
    )
    adapter_runtime_prepare_start = perf_counter()
    adapter.prepare_runtime(device=actual_device)
    runtime_summary["adapter_runtime_prepare_s"] = float(perf_counter() - adapter_runtime_prepare_start)
    _log_run(
        run_label,
        f"adapter runtime ready: device={actual_device} "
        f"encode_batch_size={adapter.default_encode_batch_size} "
        f"in {_format_elapsed_s(runtime_summary['adapter_runtime_prepare_s'])}",
    )

    probe_train = getattr(adapter, "probe_train_split", None) or train
    selected_layers = tuple(int(layer) for layer in (layers or adapter.available_layers))
    if not selected_layers:
        raise ValueError(f"Adapter for {model_name} returned no layers")
    batch_size = int(encode_batch_size or adapter.default_encode_batch_size)
    runtime_summary["feature_encode_batch_size"] = int(batch_size)
    runtime_summary["feature_encode_batch_size_source"] = (
        "cli_override" if encode_batch_size is not None else "adapter_default"
    )

    collect_train_start = perf_counter()
    _log_run(
        run_label,
        f"phase: collect train features across {len(selected_layers)} layers "
        f"(encode_batch_size={batch_size})",
    )
    train_collected = collect_probe_features_by_layer(
        encoder=adapter,
        representation_fn=adapter.make_representation_fn(layers=selected_layers, split="train"),
        layers=selected_layers,
        loader=_make_split_loader(split=probe_train, batch_size=batch_size),
        device=actual_device,
        auto_mixed_precision=adapter.autocast_context(actual_device),
        allow_crops=False,
    )
    runtime_summary["collect_train_s"] = float(perf_counter() - collect_train_start)
    _log_run(
        run_label,
        f"train features ready: samples={train_collected.timings.get('samples')} "
        f"batches={train_collected.timings.get('batches')} "
        f"in {_format_elapsed_s(runtime_summary['collect_train_s'])}",
    )

    val_collected_by_seed: dict[int, object] = {}
    collect_val_total_start = perf_counter()
    collect_val_by_seed_s: dict[str, float] = {}
    _log_run(
        run_label,
        f"phase: collect validation features for {len(validation_seed_order)} validation seeds",
    )
    for index, seed_value in enumerate(validation_seed_order, start=1):
        collect_val_seed_start = perf_counter()
        raw_val_split = val_splits[int(seed_value)]
        if index == 1 and getattr(adapter, "probe_val_split", None) is not None:
            probe_val = adapter.probe_val_split
        else:
            probe_val = adapter.update_probe_val_split(val_split=raw_val_split)
        val_collected_by_seed[int(seed_value)] = collect_probe_features_by_layer(
            encoder=adapter,
            representation_fn=adapter.make_representation_fn(layers=selected_layers, split="val"),
            layers=selected_layers,
            loader=_make_split_loader(split=probe_val, batch_size=batch_size),
            device=actual_device,
            auto_mixed_precision=adapter.autocast_context(actual_device),
            allow_crops=True,
        )
        collect_val_by_seed_s[str(int(seed_value))] = float(perf_counter() - collect_val_seed_start)
        generator_seed = validation_seed_to_generator_seed[int(seed_value)]
        _log_run(
            run_label,
            f"validation features {index}/{len(validation_seed_order)} "
            f"seed={int(seed_value)} generator_seed={int(generator_seed)} "
            f"done in {_format_elapsed_s(collect_val_by_seed_s[str(int(seed_value))])}",
        )
    runtime_summary["collect_val_total_s"] = float(perf_counter() - collect_val_total_start)
    runtime_summary["collect_val_by_validation_seed_s"] = collect_val_by_seed_s
    runtime_summary.update(
        _encoder_forward_runtime_summary(
            train_collected=train_collected,
            val_collected_by_seed=val_collected_by_seed,
        )
    )
    _log_run(
        run_label,
        f"validation feature collection done in {_format_elapsed_s(runtime_summary['collect_val_total_s'])}",
    )

    offline_probe_start = perf_counter()
    _log_run(
        run_label,
        f"phase: run offline probes across {len(selected_layers)} layers "
        f"and {len(validation_seed_order)} validation seeds",
    )
    probe_results_by_validation_seed = offline_probe_run_linear_multihead_by_layer_multi_val_from_collected(
        train_collected=train_collected,
        val_collected_by_seed=val_collected_by_seed,
        num_classes=len(manifest["class_names"]),
        class_names=list(manifest["class_names"]),
        eval_config=probe_config,
        device=actual_device,
        layers_categorical=selected_layers,
        layers_dense=selected_layers,
        layers_confusion=tuple(),
        dense_target_names=list(manifest["dense_target_names"]),
        dense_log_per_target=True,
        probe_seed=probe_seed,
        progress_callback=lambda message: _log_run(run_label, f"probe: {message}"),
    )
    runtime_summary["offline_probe_s"] = float(perf_counter() - offline_probe_start)
    _log_run(
        run_label,
        f"offline probes done in {_format_elapsed_s(runtime_summary['offline_probe_s'])}",
    )
    runtime_summary["layers_evaluated_count"] = int(len(selected_layers))
    runtime_summary["validation_seed_count"] = int(len(validation_seed_order))
    runtime_summary["finished_at_unix"] = float(time())
    runtime_summary["total_wall_s"] = float(perf_counter() - run_start)

    probe_config_payload = dict(probe_config_raw)
    probe_config_payload["probe_seed"] = None if probe_seed is None else int(probe_seed)
    probe_config_payload["learning_rate_scaling"] = probe_learning_rate_scaling_payload(
        probe_config
    )
    payload = build_model_result(
        model_name=spec.name,
        model_slug=spec.slug,
        model_type="foundational",
        model_metadata=model_taxonomy(spec.name).to_payload(),
        checkpoint=spec.checkpoint,
        source=spec.source,
        import_path=spec.import_path,
        dataset_manifest=manifest,
        probe_config=probe_config_payload,
        layers=list(selected_layers),
        adapter_metadata=adapter.adapter_metadata(),
        probe_results_by_validation_seed=probe_results_by_validation_seed,
        validation_seed_values=validation_seed_order,
        validation_seed_to_generator_seed=validation_seed_to_generator_seed,
        probe_seed=probe_seed,
        runtime_summary=runtime_summary,
    )
    out_path = result_output_path(out_dir=out_dir, model_slug=spec.slug, num_enabled=int(num_enabled))
    _log_run(run_label, f"phase: write result JSON to {out_path}")
    write_model_result(out_path=out_path, payload=payload)
    _log_run(
        run_label,
        f"done: wrote {out_path.name} in {_format_elapsed_s(runtime_summary['total_wall_s'])}",
    )
    return out_path


def run_single_model(
    *,
    model_name: str,
    dataset_config_path: Path = DATASET_CONFIG_PATH,
    probe_config_path: Path = PROBE_CONFIG_PATH,
    out_dir: Path = MODEL_RESULTS_ROOT,
    device: torch.device | None = None,
    encode_batch_size: int | None = None,
    layers: list[int] | None = None,
    train_batches: int | None = None,
    val_batches: int | None = None,
    num_enabled_values: list[int] | None = None,
    validation_seed_values: list[int] | None = None,
    validation_seed_offset: int | None = None,
    probe_seed: int | None = 0,
) -> list[Path]:
    requested_num_enabled_values = resolve_requested_num_enabled_values(
        config_path=dataset_config_path,
        requested_num_enabled_values=num_enabled_values,
    )
    out_paths: list[Path] = []
    for active_num_enabled in requested_num_enabled_values:
        out_paths.append(
            run_single_model_for_num_enabled(
                model_name=model_name,
                num_enabled=int(active_num_enabled),
                dataset_config_path=dataset_config_path,
                probe_config_path=probe_config_path,
                out_dir=out_dir,
                device=device,
                encode_batch_size=encode_batch_size,
                layers=layers,
                train_batches=train_batches,
                val_batches=val_batches,
                validation_seed_values=validation_seed_values,
                validation_seed_offset=validation_seed_offset,
                probe_seed=probe_seed,
            )
        )
    return out_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name or slug")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DATASET_CONFIG_PATH,
        help="Dataset config YAML used to build the online Aiono benchmark split",
    )
    parser.add_argument(
        "--probe-config",
        type=Path,
        default=PROBE_CONFIG_PATH,
        help="Probe config YAML",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MODEL_RESULTS_ROOT,
        help="Output directory for JSON result files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=None,
        help="Optional override for representation extraction batch size",
    )
    parser.add_argument(
        "--layer",
        action="append",
        dest="layers",
        type=int,
        default=None,
        help="Optional explicit layer id; can be repeated",
    )
    parser.add_argument(
        "--train-batches",
        type=int,
        default=None,
        help="Optional override for train batch count",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=None,
        help="Optional override for validation batch count per seed",
    )
    parser.add_argument(
        "--num-enabled",
        action="append",
        dest="num_enabled_values",
        type=int,
        default=None,
        help="Optional active num_enabled value override; can be repeated",
    )
    parser.add_argument(
        "--validation-seed-value",
        action="append",
        dest="validation_seed_values",
        type=int,
        default=None,
        help="Optional validation seed value override; can be repeated",
    )
    parser.add_argument(
        "--validation-seed-offset",
        type=int,
        default=None,
        help="Optional validation generator-seed offset override",
    )
    parser.add_argument(
        "--probe-seed",
        type=int,
        default=0,
        help="Fixed probe-training seed reused across validation-seed runs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_paths = run_single_model(
        model_name=str(args.model),
        dataset_config_path=args.dataset_config,
        probe_config_path=args.probe_config,
        out_dir=args.out_dir,
        device=torch.device(str(args.device)),
        encode_batch_size=args.encode_batch_size,
        layers=args.layers,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        num_enabled_values=args.num_enabled_values,
        validation_seed_values=args.validation_seed_values,
        validation_seed_offset=args.validation_seed_offset,
        probe_seed=args.probe_seed,
    )
    for out_path in out_paths:
        print(out_path)


if __name__ == "__main__":
    main()
