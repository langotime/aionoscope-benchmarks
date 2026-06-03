from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from aionoscope_benchmarks.constants import DATASET_CONFIG_PATH, PROBE_CONFIG_PATH, RESULTS_ROOT
from aionoscope_benchmarks.model_registry import create_adapter
from aionoscope_benchmarks.offline_probe import (
    OfflineProbeConfig,
    collect_probe_features_by_layer,
    offline_probe_run_linear_multihead_by_layer_multi_val_from_collected,
)
from aionoscope_benchmarks.results import summarize_categorical, summarize_dense
from aionoscope_benchmarks.run_model import (
    _load_probe_config,
    _make_split_loader,
    _runtime_dataset_batch_size_from_probe_config,
)
from aionoscope_benchmarks.runtime_dataset import build_runtime_splits_by_validation_seed


def _format_elapsed(seconds: float) -> str:
    return f"{seconds:.1f}s"


def _select_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def _parse_csv_floats(raw: str) -> list[float]:
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float")
    return values


def _stat_median(value: object) -> float | None:
    if isinstance(value, dict) and "median" in value:
        median = value["median"]
        return None if median is None else float(median)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _layer_feature_stats(features: torch.Tensor) -> dict[str, float | int | str]:
    values = features.detach().to(dtype=torch.float32).reshape(-1)
    finite = torch.isfinite(values)
    finite_values = values[finite]
    if int(finite_values.numel()) == 0:
        return {
            "dtype": str(features.dtype).replace("torch.", ""),
            "numel": int(values.numel()),
            "finite_fraction": 0.0,
        }
    abs_values = finite_values.abs()
    quantile_values = abs_values
    max_quantile_values = 1_000_000
    if int(quantile_values.numel()) > max_quantile_values:
        step = max(1, int(quantile_values.numel()) // max_quantile_values)
        quantile_values = quantile_values[::step][:max_quantile_values]
    return {
        "dtype": str(features.dtype).replace("torch.", ""),
        "shape": list(features.shape),
        "numel": int(values.numel()),
        "finite_fraction": float(finite.float().mean()),
        "mean": float(finite_values.mean()),
        "std": float(finite_values.std(unbiased=False)),
        "mean_abs": float(abs_values.mean()),
        "max_abs": float(abs_values.max()),
        "p99_abs": float(torch.quantile(quantile_values, 0.99)),
        "p99_abs_sample_size": int(quantile_values.numel()),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _run_probe_sweep(
    *,
    model_name: str,
    head: str,
    num_enabled: int,
    layers: tuple[int, ...],
    learning_rates: list[float],
    train_batches: int | None,
    val_batches: int | None,
    validation_seed_values: list[int],
    encode_batch_size: int | None,
    probe_seed: int,
    device: torch.device,
    output_dir: Path,
    steps: int | None,
    checkpoint_interval: int | None,
) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_config, probe_config_raw = _load_probe_config(PROBE_CONFIG_PATH)
    # LR sweeps test exact optimizer rates; disable benchmark policy scaling here
    # to avoid scaling each candidate value a second time.
    probe_config = replace(probe_config, learning_rate_scaling_enabled=False)
    probe_config_raw["learning_rate_scaling"] = {
        "enabled": False,
        "reason": "diagnostic sweep values are exact optimizer learning rates",
    }
    if steps is not None:
        probe_config = replace(probe_config, steps=int(steps))
        probe_config_raw["steps"] = int(steps)
    if checkpoint_interval is not None:
        probe_config = replace(probe_config, checkpoint_interval=int(checkpoint_interval))
        probe_config_raw["checkpoint_interval"] = int(checkpoint_interval)
    runtime_dataset_batch_size = _runtime_dataset_batch_size_from_probe_config(
        probe_config=probe_config,
        probe_config_raw=probe_config_raw,
    )

    print(
        "diagnostic start: "
        f"model={model_name} head={head} num_enabled={num_enabled} layers={layers} "
        f"learning_rates={learning_rates} validation_seeds={validation_seed_values} device={device}",
        flush=True,
    )
    adapter_start = perf_counter()
    spec, adapter = create_adapter(model_name)
    adapter = adapter.to(device)
    adapter.eval()
    channel_size = int(adapter.exact_benchmark_sequence_length())
    unavailable = sorted(set(layers) - set(int(layer) for layer in adapter.available_layers))
    if unavailable:
        raise ValueError(f"Unavailable layers {unavailable}; available={adapter.available_layers}")
    print(
        f"adapter ready: checkpoint={spec.checkpoint} channel_size={channel_size} "
        f"available_layers={len(adapter.available_layers)} in {_format_elapsed(perf_counter() - adapter_start)}",
        flush=True,
    )

    dataset_start = perf_counter()
    manifest, train, val_splits = build_runtime_splits_by_validation_seed(
        config_path=DATASET_CONFIG_PATH,
        device=device,
        batch_size=int(runtime_dataset_batch_size),
        channel_size_override=channel_size,
        channel_size_policy_override="model_native_exact",
        channel_size_source_override=f"adapter.{adapter.benchmark_sequence_length_source}",
        train_batches=train_batches,
        val_batches=val_batches,
        num_enabled=int(num_enabled),
        validation_seed_values=validation_seed_values,
        show_progress_bar=False,
        progress_callback=lambda message: print(f"dataset: {message}", flush=True),
    )
    print(
        f"dataset ready: train_batches={manifest['train_batches']} val_batches={manifest['val_batches']} "
        f"validation_seeds={manifest['validation_seed_values']} in {_format_elapsed(perf_counter() - dataset_start)}",
        flush=True,
    )

    first_seed = int(validation_seed_values[0])
    adapter.prepare(manifest=manifest, train_split=train, val_split=val_splits[first_seed])
    adapter.prepare_runtime(device=device)
    probe_train = getattr(adapter, "probe_train_split", None) or train
    batch_size = int(encode_batch_size or adapter.default_encode_batch_size)

    collect_train_start = perf_counter()
    train_collected = collect_probe_features_by_layer(
        encoder=adapter,
        representation_fn=adapter.make_representation_fn(layers=layers, split="train"),
        layers=layers,
        loader=_make_split_loader(split=probe_train, batch_size=batch_size),
        device=device,
        auto_mixed_precision=adapter.autocast_context(device),
        allow_crops=False,
    )
    print(f"train features ready in {_format_elapsed(perf_counter() - collect_train_start)}", flush=True)

    val_collected_by_seed = {}
    collect_val_start = perf_counter()
    for index, seed_value in enumerate(validation_seed_values, start=1):
        raw_val_split = val_splits[int(seed_value)]
        if index == 1 and getattr(adapter, "probe_val_split", None) is not None:
            probe_val = adapter.probe_val_split
        else:
            probe_val = adapter.update_probe_val_split(val_split=raw_val_split)
        val_collected_by_seed[int(seed_value)] = collect_probe_features_by_layer(
            encoder=adapter,
            representation_fn=adapter.make_representation_fn(layers=layers, split="val"),
            layers=layers,
            loader=_make_split_loader(split=probe_val, batch_size=batch_size),
            device=device,
            auto_mixed_precision=adapter.autocast_context(device),
            allow_crops=True,
        )
        print(f"validation features ready: seed={int(seed_value)}", flush=True)
    print(f"validation features done in {_format_elapsed(perf_counter() - collect_val_start)}", flush=True)

    feature_stats = {
        str(int(layer)): _layer_feature_stats(train_collected.features_by_layer[int(layer)])
        for layer in layers
    }
    dense_targets = list(manifest["dense_targets"])
    summary_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    payload: dict[str, Any] = {
        "config": {
            "model": spec.name,
            "model_slug": spec.slug,
            "checkpoint": spec.checkpoint,
            "head": str(head),
            "num_enabled": int(num_enabled),
            "layers": [int(layer) for layer in layers],
            "learning_rates": [float(lr) for lr in learning_rates],
            "validation_seed_values": [int(seed) for seed in validation_seed_values],
            "train_batches": int(manifest["train_batches"]),
            "val_batches": int(manifest["val_batches"]),
            "channel_size": int(channel_size),
            "encode_batch_size": int(batch_size),
            "probe_seed": int(probe_seed),
            "base_probe_config": dict(probe_config_raw),
        },
        "adapter": adapter.adapter_metadata(),
        "feature_stats": feature_stats,
        "runs": {},
    }

    for lr in learning_rates:
        run_config: OfflineProbeConfig = replace(
            probe_config,
            learning_rate=float(lr),
            final_learning_rate=float(lr),
        )
        print(f"probe lr={lr:g}: start", flush=True)
        probe_start = perf_counter()
        layers_categorical = layers if head == "categorical" else tuple()
        layers_dense = layers if head == "dense" else tuple()
        results_by_seed = offline_probe_run_linear_multihead_by_layer_multi_val_from_collected(
            train_collected=train_collected,
            val_collected_by_seed=val_collected_by_seed,
            num_classes=len(manifest["class_names"]),
            class_names=list(manifest["class_names"]),
            eval_config=run_config,
            device=device,
            layers_categorical=layers_categorical,
            layers_dense=layers_dense,
            layers_confusion=tuple(),
            dense_target_names=list(manifest["dense_target_names"]),
            dense_log_per_target=True,
            probe_seed=int(probe_seed),
            progress_callback=lambda message, lr=lr: print(f"probe lr={lr:g}: {message}", flush=True),
        )
        elapsed = perf_counter() - probe_start
        first_seed_value = int(validation_seed_values[0])
        if head == "dense":
            dense_by_layer = {
                int(layer): layer_payload
                for layer, layer_payload in results_by_seed[first_seed_value]["dense"].items()
            }
            summary = summarize_dense(dense_by_layer=dense_by_layer, dense_targets=dense_targets)
            macro_r2 = summary["macro_best_layers"]["r2"]
            macro_pearson = summary["macro_best_layers"]["pearson"]
            macro_mse = summary["macro_best_layers"]["mse"]
            summary_row = {
                "model": spec.name,
                "head": str(head),
                "num_enabled": int(num_enabled),
                "lr": float(lr),
                "layers": ",".join(str(int(layer)) for layer in layers),
                "macro_r2_layer": macro_r2["layer"],
                "macro_r2": _stat_median(macro_r2["value"]),
                "macro_pearson_layer": macro_pearson["layer"],
                "macro_pearson": _stat_median(macro_pearson["value"]),
                "macro_mse_layer": macro_mse["layer"],
                "macro_mse": _stat_median(macro_mse["value"]),
                "elapsed_s": float(elapsed),
            }
            summary_rows.append(summary_row)
            print(
                "probe lr={lr:g}: done in {elapsed}; macro_r2={r2:.6g} layer={layer} "
                "macro_pearson={pearson:.6g}".format(
                    lr=lr,
                    elapsed=_format_elapsed(elapsed),
                    r2=float(summary_row["macro_r2"]),
                    layer=summary_row["macro_r2_layer"],
                    pearson=float(summary_row["macro_pearson"]),
                ),
                flush=True,
            )
            for record in summary["oracle_dense_by_target"]:
                target_rows.append(
                    {
                        "model": spec.name,
                        "head": str(head),
                        "num_enabled": int(num_enabled),
                        "lr": float(lr),
                        "target": record["target"],
                        "signal": record["target_signal"],
                        "metric": record["target_metric"],
                        "r2": _stat_median(record["r2"]),
                        "r2_layer": record["r2_layer"],
                        "r2_best_step": _stat_median(record["r2_best_step"]),
                        "pearson": _stat_median(record["pearson"]),
                        "pearson_layer": record["pearson_layer"],
                        "mse": _stat_median(record["mse"]),
                        "mse_layer": record["mse_layer"],
                    }
                )
        elif head == "categorical":
            categorical_by_layer = {
                int(layer): layer_payload
                for layer, layer_payload in results_by_seed[first_seed_value]["categorical"].items()
            }
            summary = summarize_categorical(
                categorical_by_layer=categorical_by_layer,
                class_names=list(manifest["class_names"]),
            )
            best_auc = summary["best_auc"]
            best_auprc = summary["best_auprc"]
            summary_row = {
                "model": spec.name,
                "head": str(head),
                "num_enabled": int(num_enabled),
                "lr": float(lr),
                "layers": ",".join(str(int(layer)) for layer in layers),
                "macro_auc_layer": best_auc["layer"],
                "macro_auc": _stat_median(best_auc["macro_auc"]),
                "macro_auprc_layer": best_auprc["layer"],
                "macro_auprc": _stat_median(best_auprc["macro_auprc"]),
                "elapsed_s": float(elapsed),
            }
            summary_rows.append(summary_row)
            print(
                "probe lr={lr:g}: done in {elapsed}; macro_auc={auc:.6g} layer={auc_layer} "
                "macro_auprc={auprc:.6g} layer={auprc_layer}".format(
                    lr=lr,
                    elapsed=_format_elapsed(elapsed),
                    auc=float(summary_row["macro_auc"]),
                    auc_layer=summary_row["macro_auc_layer"],
                    auprc=float(summary_row["macro_auprc"]),
                    auprc_layer=summary_row["macro_auprc_layer"],
                ),
                flush=True,
            )
            for record in summary["oracle_categorical_by_signal"]:
                signal_rows.append(
                    {
                        "model": spec.name,
                        "head": str(head),
                        "num_enabled": int(num_enabled),
                        "lr": float(lr),
                        "signal": record["signal"],
                        "auroc": _stat_median(record["auroc"]),
                        "auroc_layer": record["auroc_layer"],
                        "auprc": _stat_median(record["auprc"]),
                        "auprc_layer": record["auprc_layer"],
                    }
                )
        else:
            raise ValueError(f"Unsupported head: {head}")
        payload["runs"][f"lr_{lr:g}"] = {
            "elapsed_s": float(elapsed),
            "summary": summary,
        }

    safe_model = str(spec.slug).replace("/", "_")
    prefix = output_dir / f"{safe_model}__num_enabled_{int(num_enabled)}__{head}"
    payload_path = prefix.with_name(prefix.name + "__payload.json")
    payload_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(prefix.with_name(prefix.name + "__summary.csv"), summary_rows)
    _write_csv(prefix.with_name(prefix.name + "__targets.csv"), target_rows)
    _write_csv(prefix.with_name(prefix.name + "__signals.csv"), signal_rows)
    print(f"wrote {payload_path}", flush=True)
    print(f"wrote {prefix.with_name(prefix.name + '__summary.csv')}", flush=True)
    if target_rows:
        print(f"wrote {prefix.with_name(prefix.name + '__targets.csv')}", flush=True)
    if signal_rows:
        print(f"wrote {prefix.with_name(prefix.name + '__signals.csv')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose probe learning-rate sensitivity.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--head", choices=("dense", "categorical"), default="dense")
    parser.add_argument("--num-enabled", type=int, default=1)
    parser.add_argument("--layers", required=True, help="Comma-separated layer ids")
    parser.add_argument("--learning-rates", default="0.01,0.003,0.001,0.0003,0.0001")
    parser.add_argument("--validation-seeds", default="0")
    parser.add_argument("--train-batches", type=int, default=None)
    parser.add_argument("--val-batches", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=None)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_ROOT / "toto2_probe_lr_diagnostics",
    )
    args = parser.parse_args()
    _run_probe_sweep(
        model_name=str(args.model),
        head=str(args.head),
        num_enabled=int(args.num_enabled),
        layers=tuple(_parse_csv_ints(args.layers)),
        learning_rates=_parse_csv_floats(args.learning_rates),
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        validation_seed_values=_parse_csv_ints(args.validation_seeds),
        encode_batch_size=args.encode_batch_size,
        probe_seed=int(args.probe_seed),
        device=_select_device(str(args.device)),
        output_dir=args.output_dir,
        steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
