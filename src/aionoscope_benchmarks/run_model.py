from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .constants import DATASET_CONFIG_PATH, MODEL_RESULTS_ROOT, PROBE_CONFIG_PATH
from .dataset_snapshot import build_runtime_splits
from .model_registry import create_adapter
from .offline_probe import OfflineProbeConfig, offline_probe_run_linear_multihead_by_layer
from .results import build_model_result, write_model_result


def _load_probe_config(path: Path) -> tuple[OfflineProbeConfig, dict[str, object]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML dict, got {type(raw).__name__}")
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
    )
    return config, raw


def run_single_model(
    *,
    model_name: str,
    dataset_config_path: Path = DATASET_CONFIG_PATH,
    probe_config_path: Path = PROBE_CONFIG_PATH,
    out_dir: Path = MODEL_RESULTS_ROOT,
    device: torch.device | None = None,
    encode_batch_size: int | None = None,
    layers: list[int] | None = None,
) -> Path:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    actual_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_config, probe_config_raw = _load_probe_config(probe_config_path)
    manifest, train, val = build_runtime_splits(
        config_path=dataset_config_path,
        device=actual_device,
        batch_size=int(probe_config.batch_size),
    )
    spec, adapter = create_adapter(model_name)
    adapter = adapter.to(actual_device)
    adapter.eval()
    adapter.prepare(manifest=manifest, train_split=train, val_split=val)
    probe_train = getattr(adapter, "probe_train_split", train)
    probe_val = getattr(adapter, "probe_val_split", val)

    selected_layers = tuple(int(layer) for layer in (layers or adapter.available_layers))
    if not selected_layers:
        raise ValueError(f"Adapter for {model_name} returned no layers")
    batch_size = int(encode_batch_size or adapter.default_encode_batch_size)

    train_dataset = TensorDataset(probe_train["x"], probe_train["y_cls"], probe_train["y_dense"])
    val_dataset = TensorDataset(probe_val["x"], probe_val["y_cls"], probe_val["y_dense"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    probe_results = offline_probe_run_linear_multihead_by_layer(
        encoder=adapter,
        train_representation_fn=adapter.make_representation_fn(layers=selected_layers, split="train"),
        val_representation_fn=adapter.make_representation_fn(layers=selected_layers, split="val"),
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(manifest["class_names"]),
        class_names=list(manifest["class_names"]),
        eval_config=probe_config,
        device=actual_device,
        auto_mixed_precision=adapter.autocast_context(actual_device),
        layers_categorical=selected_layers,
        layers_dense=selected_layers,
        layers_confusion=tuple(),
        dense_target_names=list(manifest["dense_target_names"]),
        dense_log_per_target=True,
    )

    payload = build_model_result(
        model_name=spec.name,
        model_slug=spec.slug,
        model_type="foundational",
        checkpoint=spec.checkpoint,
        source=spec.source,
        import_path=spec.import_path,
        dataset_manifest=manifest,
        probe_config=probe_config_raw,
        layers=list(selected_layers),
        adapter_metadata=adapter.adapter_metadata(),
        probe_results=probe_results,
    )
    out_path = out_dir / f"{spec.slug}.json"
    write_model_result(out_path=out_path, payload=payload)
    return out_path


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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = run_single_model(
        model_name=str(args.model),
        dataset_config_path=args.dataset_config,
        probe_config_path=args.probe_config,
        out_dir=args.out_dir,
        device=torch.device(str(args.device)),
        encode_batch_size=args.encode_batch_size,
        layers=args.layers,
    )
    print(out_path)


if __name__ == "__main__":
    main()
