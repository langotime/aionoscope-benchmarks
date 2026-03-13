from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


_DENSE_DIRECTIONS = {
    "mse": "min",
    "mae": "min",
    "r2": "max",
    "pearson": "max",
}


def _layer_key(layer: int) -> str:
    return str(int(layer))


def summarize_categorical(
    *, categorical_by_layer: dict[int, dict[str, object]], class_names: list[str]
) -> dict[str, object]:
    if not categorical_by_layer:
        raise ValueError("categorical_by_layer must be non-empty")

    best_auc_layer = max(
        categorical_by_layer,
        key=lambda layer: float(categorical_by_layer[layer]["best_auc"]["macro_auc"]),
    )
    best_auprc_layer = max(
        categorical_by_layer,
        key=lambda layer: float(categorical_by_layer[layer]["best_auprc"]["macro_auprc"]),
    )
    best_auc = dict(categorical_by_layer[best_auc_layer]["best_auc"])
    best_auprc = dict(categorical_by_layer[best_auprc_layer]["best_auprc"])
    best_auc["layer"] = int(best_auc_layer)
    best_auprc["layer"] = int(best_auprc_layer)

    oracle_by_signal = []
    for signal in class_names:
        auroc_layer = max(
            categorical_by_layer,
            key=lambda layer: float(categorical_by_layer[layer]["best_auc"]["per_class_auc"][signal]),
        )
        auprc_layer = max(
            categorical_by_layer,
            key=lambda layer: float(categorical_by_layer[layer]["best_auprc"]["per_class_auprc"][signal]),
        )
        oracle_by_signal.append(
            {
                "signal": signal,
                "auroc": float(categorical_by_layer[auroc_layer]["best_auc"]["per_class_auc"][signal]),
                "auroc_layer": int(auroc_layer),
                "auprc": float(categorical_by_layer[auprc_layer]["best_auprc"]["per_class_auprc"][signal]),
                "auprc_layer": int(auprc_layer),
            }
        )

    return {
        "best_auc": best_auc,
        "best_auprc": best_auprc,
        "oracle_categorical_by_signal": oracle_by_signal,
    }


def summarize_dense(
    *,
    dense_by_layer: dict[int, dict[str, object]],
    dense_targets: list[dict[str, str]],
) -> dict[str, object]:
    if not dense_by_layer:
        return {
            "oracle_dense_by_target": [],
            "macro_best_layers": {},
        }
    dense_target_names = [str(item["name"]) for item in dense_targets]
    dense_specs = {str(item["name"]): item for item in dense_targets}
    records = []
    for target_name in dense_target_names:
        spec = dense_specs[target_name]
        record: dict[str, object] = {
            "target": target_name,
            "target_signal": str(spec["signal"]),
            "target_metric": str(spec["metric"]),
        }
        for metric_name, direction in _DENSE_DIRECTIONS.items():
            key = f"per_target_{metric_name}"
            if key not in next(iter(dense_by_layer.values())):
                continue
            if direction == "max":
                best_layer = max(
                    dense_by_layer,
                    key=lambda layer: float(dense_by_layer[layer][key][target_name]),
                )
            else:
                best_layer = min(
                    dense_by_layer,
                    key=lambda layer: float(dense_by_layer[layer][key][target_name]),
                )
            record[metric_name] = float(dense_by_layer[best_layer][key][target_name])
            record[f"{metric_name}_layer"] = int(best_layer)
            record[f"{metric_name}_best_step"] = int(
                dense_by_layer[best_layer]["per_target_best_step"][target_name]
            )
        records.append(record)

    macro_best_layers = {}
    for metric_name, direction in _DENSE_DIRECTIONS.items():
        macro_key = f"macro_{metric_name}"
        if macro_key not in next(iter(dense_by_layer.values())):
            continue
        if direction == "max":
            best_layer = max(dense_by_layer, key=lambda layer: float(dense_by_layer[layer][macro_key]))
        else:
            best_layer = min(dense_by_layer, key=lambda layer: float(dense_by_layer[layer][macro_key]))
        macro_best_layers[metric_name] = {
            "layer": int(best_layer),
            "value": float(dense_by_layer[best_layer][macro_key]),
        }
    return {
        "oracle_dense_by_target": records,
        "macro_best_layers": macro_best_layers,
    }


def build_model_result(
    *,
    model_name: str,
    model_slug: str,
    model_type: str,
    checkpoint: str,
    source: str,
    import_path: str,
    dataset_manifest: dict[str, object],
    probe_config: dict[str, object],
    layers: list[int],
    adapter_metadata: dict[str, object],
    probe_results: dict[str, object],
) -> dict[str, object]:
    categorical_by_layer = {
        int(layer): payload for layer, payload in probe_results["categorical"].items()
    }
    dense_by_layer = {int(layer): payload for layer, payload in probe_results["dense"].items()}
    summaries = {
        **summarize_categorical(
            categorical_by_layer=categorical_by_layer,
            class_names=list(dataset_manifest["class_names"]),
        ),
        **summarize_dense(
            dense_by_layer=dense_by_layer,
            dense_targets=list(dataset_manifest["dense_targets"]),
        ),
    }
    return {
        "model": {
            "name": model_name,
            "slug": model_slug,
            "type": model_type,
            "checkpoint": checkpoint,
            "source": source,
            "import_path": import_path,
            "layers_evaluated": [int(layer) for layer in layers],
            "adapter": adapter_metadata,
        },
        "dataset": dataset_manifest,
        "probe_config": probe_config,
        "results": {
            "categorical": {_layer_key(layer): payload for layer, payload in categorical_by_layer.items()},
            "dense": {_layer_key(layer): payload for layer, payload in dense_by_layer.items()},
            "shared": {
                "dense_targets_available": bool(probe_results["dense_targets_available"]),
                "feature_dim": int(probe_results["feature_dim"]),
                "train_size": int(probe_results["train_size"]),
                "val_size": int(probe_results["val_size"]),
                "val_num_crops": int(probe_results["val_num_crops"]),
                "timings": probe_results["timings"],
            },
            "summary": summaries,
        },
    }


def write_model_result(*, out_path: Path, payload: dict[str, object]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model_result(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_dense_values_by_group(
    *,
    oracle_dense_by_target: list[dict[str, object]],
    metric_name: str,
    group_key: str,
    categories: list[str],
) -> list[float]:
    rows = oracle_dense_by_target
    values = []
    for category in categories:
        category_values = [
            float(row[metric_name]) for row in rows if str(row[group_key]) == category
        ]
        if not category_values:
            raise ValueError(
                f"Missing dense values for {group_key}={category!r} metric={metric_name!r}"
            )
        values.append(float(np.median(np.asarray(category_values, dtype=float))))
    return values

