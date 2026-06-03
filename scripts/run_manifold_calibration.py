from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, time
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from aionoscope_benchmarks.constants import DATASET_CONFIG_PATH, RESULTS_ROOT
from aionoscope_benchmarks.manifold_config import ManifoldEvalConfig
from aionoscope_benchmarks.manifold_eval import (
    compute_manifold_layer_evaluation,
    summarize_layer_metrics,
)
from aionoscope_benchmarks.manifold_slices import build_controlled_manifold_slice
from aionoscope_benchmarks.manifold_viewer import build_viewer
from aionoscope_benchmarks.manifold_viz import write_visualization_bundle
from aionoscope_benchmarks.model_registry import create_adapter, model_taxonomy
from aionoscope_benchmarks.offline_probe import collect_probe_features_by_layer
from aionoscope_benchmarks.runtime_dataset import build_runtime_splits


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(message: str) -> None:
    print(f"[{_utc_timestamp()}] {message}", file=sys.stderr, flush=True)


def _load_config(path: Path | None) -> ManifoldEvalConfig:
    if path is None:
        return ManifoldEvalConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML dict, got {type(raw).__name__}")
    if "manifold_eval" in raw:
        raw = raw["manifold_eval"]
    return ManifoldEvalConfig.from_mapping(raw)


def _selected_layers(
    *,
    available_layers: tuple[int, ...],
    requested_layers: list[int] | None,
    max_layers: int | None,
) -> tuple[int, ...]:
    if requested_layers:
        layers = tuple(int(layer) for layer in requested_layers)
    else:
        layers = tuple(int(layer) for layer in available_layers)
    if max_layers is not None and int(max_layers) > 0 and len(layers) > int(max_layers):
        positions = torch.linspace(0, len(layers) - 1, steps=int(max_layers)).round().to(torch.int64)
        layers = tuple(layers[int(index)] for index in torch.unique(positions).tolist())
    if not layers:
        raise ValueError("selected layers must be non-empty")
    if len(set(layers)) != len(layers):
        raise ValueError(f"selected layers must be unique, got {layers}")
    return layers


def _make_loader(split: dict[str, torch.Tensor], *, batch_size: int) -> DataLoader:
    return DataLoader(
        TensorDataset(split["x"], split["y_cls"], split["y_dense"]),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )


def _split_to_device(split: dict[str, torch.Tensor], *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, non_blocking=device.type == "cuda")
        for key, value in split.items()
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _artifact_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_calibration(
    *,
    config: ManifoldEvalConfig,
    models: tuple[str, ...],
    targets: tuple[str, ...],
    out_root: Path,
    device: torch.device,
    generation_device: torch.device,
    encode_batch_size: int | None,
    layers: list[int] | None,
    max_layers: int | None,
    run_id: str | None,
    skip_plots: bool,
    skip_viewer: bool,
) -> Path:
    config.validate()
    active_run_id = run_id or _artifact_run_id()
    run_root = out_root / active_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "config.json", config.to_payload())
    run_started = perf_counter()
    index_records: list[dict[str, Any]] = []

    for model_name in models:
        model_start = perf_counter()
        _log(f"[{model_name}] load adapter on {device}")
        spec, adapter = create_adapter(model_name)
        adapter = adapter.to(device)
        adapter.eval()
        seq_len = int(adapter.exact_benchmark_sequence_length())
        selected_layers = _selected_layers(
            available_layers=adapter.available_layers,
            requested_layers=layers,
            max_layers=max_layers,
        )
        batch_size = int(encode_batch_size or adapter.default_encode_batch_size)
        _log(
            f"[{model_name}] ready: seq_len={seq_len} layers={list(selected_layers)} "
            f"encode_batch_size={batch_size}"
        )

        first_train = build_controlled_manifold_slice(
            config_path=config.dataset_config_path,
            target_name=targets[0],
            seq_len=seq_len,
            grid_size=int(config.grid_size_1d),
            split="train",
            repeats_per_grid_point=int(config.repeats_per_grid_point),
            seed=0,
            device=generation_device,
            view_grid_mode=config.view_grid_mode,
            view_range_max_abs=config.view_range_max_abs,
            view_log_min_abs=float(config.view_log_min_abs),
        )
        first_val = build_controlled_manifold_slice(
            config_path=config.dataset_config_path,
            target_name=targets[0],
            seq_len=seq_len,
            grid_size=int(config.grid_size_1d),
            split="val",
            repeats_per_grid_point=1,
            seed=1,
            device=generation_device,
            view_grid_mode=config.view_grid_mode,
            view_range_max_abs=config.view_range_max_abs,
            view_log_min_abs=float(config.view_log_min_abs),
        )
        if bool(getattr(adapter, "manifold_requires_balanced_prepare", False)):
            _log(f"[{model_name}] build balanced prepare split for supervised tabular adapter")
            prepare_manifest, prepare_train, prepare_val = build_runtime_splits(
                config_path=config.dataset_config_path,
                device=generation_device,
                batch_size=256,
                channel_size_override=seq_len,
                channel_size_policy_override="model_native_exact",
                channel_size_source_override=f"adapter.{adapter.benchmark_sequence_length_source}",
                num_enabled=1,
            )
            adapter.prepare(
                manifest=prepare_manifest,
                train_split=prepare_train,
                val_split=prepare_val,
            )
            train_representation_split = "manifold_train"
            val_representation_split = "manifold_val"
        else:
            adapter.prepare(
                manifest=first_train.manifest["dataset_manifest"],
                train_split=_split_to_device(first_train.split, device=device),
                val_split=_split_to_device(first_val.split, device=device),
            )
            train_representation_split = "train"
            val_representation_split = "val"
        adapter.prepare_runtime(device=device)

        for target_name in targets:
            target_start = perf_counter()
            _log(f"[{model_name}] target {target_name}: build controlled slices")
            train_slice = (
                first_train
                if target_name == targets[0]
                else build_controlled_manifold_slice(
                    config_path=config.dataset_config_path,
                    target_name=target_name,
                    seq_len=seq_len,
                    grid_size=int(config.grid_size_1d),
                    split="train",
                    repeats_per_grid_point=int(config.repeats_per_grid_point),
                    seed=0,
                    device=generation_device,
                    view_grid_mode=config.view_grid_mode,
                    view_range_max_abs=config.view_range_max_abs,
                    view_log_min_abs=float(config.view_log_min_abs),
                )
            )
            val_slice = (
                first_val
                if target_name == targets[0]
                else build_controlled_manifold_slice(
                    config_path=config.dataset_config_path,
                    target_name=target_name,
                    seq_len=seq_len,
                    grid_size=int(config.grid_size_1d),
                    split="val",
                    repeats_per_grid_point=1,
                    seed=1,
                    device=generation_device,
                    view_grid_mode=config.view_grid_mode,
                    view_range_max_abs=config.view_range_max_abs,
                    view_log_min_abs=float(config.view_log_min_abs),
                )
            )
            _log(f"[{model_name}] target {target_name}: collect train features")
            train_collected = collect_probe_features_by_layer(
                encoder=adapter,
                representation_fn=adapter.make_representation_fn(
                    layers=selected_layers,
                    split=train_representation_split,
                ),
                layers=selected_layers,
                loader=_make_loader(train_slice.split, batch_size=batch_size),
                device=device,
                auto_mixed_precision=adapter.autocast_context(device),
                allow_crops=False,
            )
            _log(f"[{model_name}] target {target_name}: collect val features")
            val_collected = collect_probe_features_by_layer(
                encoder=adapter,
                representation_fn=adapter.make_representation_fn(
                    layers=selected_layers,
                    split=val_representation_split,
                ),
                layers=selected_layers,
                loader=_make_loader(val_slice.split, batch_size=batch_size),
                device=device,
                auto_mixed_precision=adapter.autocast_context(device),
                allow_crops=False,
            )

            target_payload = train_slice.manifest["target"]
            geometry = str(target_payload["geometry"])
            period = target_payload.get("period")
            target_dir = run_root / spec.slug / target_name
            plot_dir = target_dir / "plots"
            by_layer: dict[int, dict[str, Any]] = {}
            visualizations: dict[str, dict[str, str]] = {}
            _log(f"[{model_name}] target {target_name}: compute metrics")
            for layer in selected_layers:
                evaluation = compute_manifold_layer_evaluation(
                    train_features=train_collected.features_by_layer[int(layer)],
                    train_grid_index=train_slice.grid_index,
                    train_coordinates=train_slice.latent_coordinates,
                    val_features=val_collected.features_by_layer[int(layer)],
                    val_coordinates=val_slice.latent_coordinates,
                    geometry=geometry,
                    period=None if period is None else float(period),
                    pca_dim=int(config.pca_dim),
                    geodesic_neighbors=tuple(int(k) for k in config.geodesic_neighbors),
                    plot_max_points=int(config.plot_max_points),
                )
                by_layer[int(layer)] = evaluation.metrics
                if not skip_plots and config.write_plots:
                    stem = f"{spec.slug}__{target_name}__layer_{int(layer)}"
                    visualizations[str(int(layer))] = write_visualization_bundle(
                        out_dir=plot_dir,
                        stem=stem,
                        plot_data=evaluation.plot_data,
                        metrics=evaluation.metrics,
                        title=f"{spec.name} / {target_name} / layer {int(layer)}",
                    )

            summary = summarize_layer_metrics(by_layer=by_layer)
            payload = {
                "schema_version": "manifold_result_v0",
                "plan_issue": "https://github.com/langotime/aionoscope-benchmarks/issues/5",
                "created_at_unix": float(time()),
                "model": {
                    "name": spec.name,
                    "slug": spec.slug,
                    "checkpoint": spec.checkpoint,
                    **model_taxonomy(spec.name).to_payload(),
                    "layers_evaluated": [int(layer) for layer in selected_layers],
                    "adapter": adapter.adapter_metadata(),
                },
                "target": target_payload,
                "config": config.to_payload(),
                "runtime": {
                    "device": str(device),
                    "generation_device": str(generation_device),
                    "process_id": int(os.getpid()),
                },
                "train_slice_manifest": train_slice.manifest,
                "val_slice_manifest": val_slice.manifest,
                "by_layer": {str(layer): metrics for layer, metrics in sorted(by_layer.items())},
                "summary": summary,
                "visualizations": visualizations,
                "timings": {
                    "target_wall_s": float(perf_counter() - target_start),
                    "train_collect": train_collected.timings,
                    "val_collect": val_collected.timings,
                },
            }
            metrics_path = target_dir / "metrics.json"
            _write_json(metrics_path, payload)
            index_records.append(
                {
                    "model": spec.name,
                    "model_slug": spec.slug,
                    "target": target_name,
                    "metrics_json": str(metrics_path),
                    "summary": summary,
                }
            )
            _log(
                f"[{model_name}] target {target_name}: wrote {metrics_path} "
                f"in {perf_counter() - target_start:.1f}s"
            )
        _log(f"[{model_name}] done in {perf_counter() - model_start:.1f}s")

    _write_json(
        run_root / "index.json",
        {
            "schema_version": "manifold_index_v0",
            "run_id": active_run_id,
            "created_at_unix": float(time()),
            "config": config.to_payload(),
            "records": index_records,
            "total_wall_s": float(perf_counter() - run_started),
        },
    )
    if config.write_viewer and not skip_viewer:
        viewer_path = run_root / "index.html"
        build_viewer(artifact_root=run_root, out_path=viewer_path)
        _log(f"viewer: {viewer_path}")
    return run_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Optional manifold eval YAML config")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DATASET_CONFIG_PATH,
        help="Dataset config YAML",
    )
    parser.add_argument("--model", action="append", dest="models", default=None)
    parser.add_argument("--target", action="append", dest="targets", default=None)
    parser.add_argument("--layer", action="append", dest="layers", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--repeats-per-grid-point", type=int, default=None)
    parser.add_argument("--pca-dim", type=int, default=None)
    parser.add_argument("--geodesic-neighbor", action="append", dest="geodesic_neighbors", type=int, default=None)
    parser.add_argument("--view-grid-mode", choices=("linear", "log", "signed_log"), default=None)
    parser.add_argument("--view-range-max-abs", type=float, default=None)
    parser.add_argument("--view-log-min-abs", type=float, default=None)
    parser.add_argument("--plot-max-points", type=int, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--generation-device", type=str, default="cpu")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    out_root = args.out_root or config.artifact_root
    config = replace(
        config,
        dataset_config_path=args.dataset_config,
        artifact_root=out_root,
        models=tuple(args.models) if args.models else config.models,
        targets=tuple(args.targets) if args.targets else config.targets,
        grid_size_1d=int(args.grid_size or config.grid_size_1d),
        repeats_per_grid_point=int(args.repeats_per_grid_point or config.repeats_per_grid_point),
        pca_dim=int(args.pca_dim or config.pca_dim),
        geodesic_neighbors=(
            tuple(int(value) for value in args.geodesic_neighbors)
            if args.geodesic_neighbors
            else config.geodesic_neighbors
        ),
        view_grid_mode=args.view_grid_mode or config.view_grid_mode,
        view_range_max_abs=(
            float(args.view_range_max_abs)
            if args.view_range_max_abs is not None
            else config.view_range_max_abs
        ),
        view_log_min_abs=(
            float(args.view_log_min_abs)
            if args.view_log_min_abs is not None
            else float(config.view_log_min_abs)
        ),
        plot_max_points=int(args.plot_max_points or config.plot_max_points),
        write_plots=bool(config.write_plots and not (args.skip_plots or args.no_plots)),
        write_viewer=bool(config.write_viewer and not args.no_viewer),
    )
    config.validate()
    out_path = run_calibration(
        config=config,
        models=config.models,
        targets=config.targets,
        out_root=out_root,
        device=torch.device(str(args.device)),
        generation_device=torch.device(str(args.generation_device)),
        encode_batch_size=args.encode_batch_size,
        layers=args.layers,
        max_layers=args.max_layers,
        run_id=args.run_id,
        skip_plots=bool(args.skip_plots or args.no_plots),
        skip_viewer=bool(args.no_viewer),
    )
    print(out_path)


if __name__ == "__main__":
    main()
