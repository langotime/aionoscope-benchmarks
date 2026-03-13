from __future__ import annotations

import copy
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

_STAT_KEYS = frozenset({"values", "median", "std", "n"})


def _layer_key(layer: int) -> str:
    return str(int(layer))


def _is_numeric(value: object) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_stat_payload(value: object) -> bool:
    return isinstance(value, dict) and _STAT_KEYS.issubset(value.keys())


def _stat_payload(values: list[object], *, path: str) -> dict[str, object]:
    if not values:
        raise ValueError(f"Cannot aggregate empty numeric value list: path={path}")
    arr = np.asarray([float(value) for value in values], dtype=float)
    if not np.all(np.isfinite(arr)):
        bad = [float(value) for value in arr.tolist() if not np.isfinite(float(value))]
        raise ValueError(f"Non-finite numeric values while aggregating {path}: {bad}")
    std = float(np.std(arr, ddof=1)) if int(arr.size) >= 2 else 0.0
    return {
        "values": [float(value) for value in arr.tolist()],
        "median": float(np.median(arr)),
        "std": float(std),
        "n": int(arr.size),
    }


def _stat_median(value: object) -> float:
    if _is_stat_payload(value):
        return float(value["median"])
    if _is_numeric(value):
        return float(value)
    raise TypeError(f"Expected numeric or stat payload, got {type(value).__name__}")


def _aggregate_tree(values: list[object], *, path: str) -> object:
    if not values:
        raise ValueError(f"Cannot aggregate empty value list: path={path}")
    first = values[0]
    if isinstance(first, bool):
        if not all(isinstance(value, bool) and value == first for value in values):
            raise ValueError(f"Boolean values differ while aggregating {path}: {values}")
        return bool(first)
    if _is_numeric(first):
        if not all(_is_numeric(value) for value in values):
            raise ValueError(f"Mixed numeric/non-numeric values while aggregating {path}")
        return _stat_payload(values, path=path)
    if isinstance(first, str):
        if not all(isinstance(value, str) and value == first for value in values):
            raise ValueError(f"String values differ while aggregating {path}: {values}")
        return str(first)
    if first is None:
        if not all(value is None for value in values):
            raise ValueError(f"Mixed None/non-None values while aggregating {path}")
        return None
    if isinstance(first, dict):
        keys = list(first.keys())
        out: dict[str, object] = {}
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(f"Mixed dict/non-dict values while aggregating {path}")
            if set(value.keys()) != set(keys):
                raise ValueError(
                    "Dict keys differ while aggregating "
                    f"{path}: expected={keys} got={list(value.keys())}"
                )
        for key in keys:
            out[key] = _aggregate_tree(
                [value[key] for value in values],
                path=f"{path}.{key}",
            )
        return out
    raise TypeError(f"Unsupported value type while aggregating {path}: {type(first).__name__}")


def _ensure_validation_seed_order(
    *,
    probe_results_by_validation_seed: dict[int, dict[str, object]],
    validation_seed_values: list[int],
) -> list[int]:
    ordered = [int(seed_value) for seed_value in validation_seed_values]
    if set(ordered) != set(int(seed_value) for seed_value in probe_results_by_validation_seed):
        raise ValueError(
            "Validation seed values do not match available probe results. "
            f"expected={ordered} got={sorted(int(seed) for seed in probe_results_by_validation_seed)}"
        )
    return ordered


def _aggregate_layer_payloads(
    *,
    probe_results_by_validation_seed: dict[int, dict[str, object]],
    validation_seed_values: list[int],
    section_key: str,
) -> dict[int, dict[str, object]]:
    ordered_seed_values = _ensure_validation_seed_order(
        probe_results_by_validation_seed=probe_results_by_validation_seed,
        validation_seed_values=validation_seed_values,
    )
    reference = probe_results_by_validation_seed[ordered_seed_values[0]][section_key]
    layer_keys = sorted(int(layer) for layer in reference)
    out: dict[int, dict[str, object]] = {}
    for seed_value in ordered_seed_values:
        current = probe_results_by_validation_seed[seed_value][section_key]
        current_keys = sorted(int(layer) for layer in current)
        if current_keys != layer_keys:
            raise ValueError(
                f"Layer keys differ for section={section_key!r}: expected={layer_keys} got={current_keys}"
            )
    for layer in layer_keys:
        out[int(layer)] = _aggregate_tree(
            [
                probe_results_by_validation_seed[seed_value][section_key][int(layer)]
                for seed_value in ordered_seed_values
            ],
            path=f"{section_key}.{int(layer)}",
        )
    return out


def _aggregate_shared_fields(
    *,
    probe_results_by_validation_seed: dict[int, dict[str, object]],
    validation_seed_values: list[int],
    validation_seed_to_generator_seed: dict[int, int],
    probe_seed: int | None,
) -> dict[str, object]:
    ordered_seed_values = _ensure_validation_seed_order(
        probe_results_by_validation_seed=probe_results_by_validation_seed,
        validation_seed_values=validation_seed_values,
    )
    dense_targets_available = {
        bool(probe_results_by_validation_seed[seed_value]["dense_targets_available"])
        for seed_value in ordered_seed_values
    }
    if len(dense_targets_available) != 1:
        raise ValueError(
            "dense_targets_available differs across validation seeds: "
            f"{sorted(dense_targets_available)}"
        )
    return {
        "dense_targets_available": bool(next(iter(dense_targets_available))),
        "feature_dim": _aggregate_tree(
            [probe_results_by_validation_seed[seed_value]["feature_dim"] for seed_value in ordered_seed_values],
            path="shared.feature_dim",
        ),
        "train_size": _aggregate_tree(
            [probe_results_by_validation_seed[seed_value]["train_size"] for seed_value in ordered_seed_values],
            path="shared.train_size",
        ),
        "val_size": _aggregate_tree(
            [probe_results_by_validation_seed[seed_value]["val_size"] for seed_value in ordered_seed_values],
            path="shared.val_size",
        ),
        "val_num_crops": _aggregate_tree(
            [probe_results_by_validation_seed[seed_value]["val_num_crops"] for seed_value in ordered_seed_values],
            path="shared.val_num_crops",
        ),
        "timings": _aggregate_tree(
            [probe_results_by_validation_seed[seed_value]["timings"] for seed_value in ordered_seed_values],
            path="shared.timings",
        ),
        "probe_seed": None if probe_seed is None else int(probe_seed),
        "validation_seed_values": [int(seed_value) for seed_value in ordered_seed_values],
        "validation_generator_seeds": [
            int(validation_seed_to_generator_seed[int(seed_value)]) for seed_value in ordered_seed_values
        ],
        "validation_seed_to_generator_seed": {
            str(int(seed_value)): int(validation_seed_to_generator_seed[int(seed_value)])
            for seed_value in ordered_seed_values
        },
        "n_validation_runs": int(len(ordered_seed_values)),
    }


def summarize_categorical(
    *, categorical_by_layer: dict[int, dict[str, object]], class_names: list[str]
) -> dict[str, object]:
    if not categorical_by_layer:
        raise ValueError("categorical_by_layer must be non-empty")

    best_auc_layer = max(
        categorical_by_layer,
        key=lambda layer: _stat_median(categorical_by_layer[layer]["best_auc"]["macro_auc"]),
    )
    best_auprc_layer = max(
        categorical_by_layer,
        key=lambda layer: _stat_median(categorical_by_layer[layer]["best_auprc"]["macro_auprc"]),
    )
    best_auc = copy.deepcopy(categorical_by_layer[best_auc_layer]["best_auc"])
    best_auprc = copy.deepcopy(categorical_by_layer[best_auprc_layer]["best_auprc"])
    best_auc["layer"] = int(best_auc_layer)
    best_auprc["layer"] = int(best_auprc_layer)

    oracle_by_signal = []
    for signal in class_names:
        auroc_layer = max(
            categorical_by_layer,
            key=lambda layer: _stat_median(
                categorical_by_layer[layer]["best_auc"]["per_class_auc"][signal]
            ),
        )
        auprc_layer = max(
            categorical_by_layer,
            key=lambda layer: _stat_median(
                categorical_by_layer[layer]["best_auprc"]["per_class_auprc"][signal]
            ),
        )
        oracle_by_signal.append(
            {
                "signal": signal,
                "auroc": copy.deepcopy(
                    categorical_by_layer[auroc_layer]["best_auc"]["per_class_auc"][signal]
                ),
                "auroc_layer": int(auroc_layer),
                "auprc": copy.deepcopy(
                    categorical_by_layer[auprc_layer]["best_auprc"]["per_class_auprc"][signal]
                ),
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
    reference_layer = next(iter(dense_by_layer.values()))
    for target_name in dense_target_names:
        spec = dense_specs[target_name]
        record: dict[str, object] = {
            "target": target_name,
            "target_signal": str(spec["signal"]),
            "target_metric": str(spec["metric"]),
        }
        for metric_name, direction in _DENSE_DIRECTIONS.items():
            key = f"per_target_{metric_name}"
            if key not in reference_layer:
                continue
            if direction == "max":
                best_layer = max(
                    dense_by_layer,
                    key=lambda layer: _stat_median(dense_by_layer[layer][key][target_name]),
                )
            else:
                best_layer = min(
                    dense_by_layer,
                    key=lambda layer: _stat_median(dense_by_layer[layer][key][target_name]),
                )
            record[metric_name] = copy.deepcopy(dense_by_layer[best_layer][key][target_name])
            record[f"{metric_name}_layer"] = int(best_layer)
            record[f"{metric_name}_best_step"] = copy.deepcopy(
                dense_by_layer[best_layer]["per_target_best_step"][target_name]
            )
        records.append(record)

    macro_best_layers = {}
    for metric_name, direction in _DENSE_DIRECTIONS.items():
        macro_key = f"macro_{metric_name}"
        if macro_key not in reference_layer:
            continue
        if direction == "max":
            best_layer = max(dense_by_layer, key=lambda layer: _stat_median(dense_by_layer[layer][macro_key]))
        else:
            best_layer = min(dense_by_layer, key=lambda layer: _stat_median(dense_by_layer[layer][macro_key]))
        macro_best_layers[metric_name] = {
            "layer": int(best_layer),
            "value": copy.deepcopy(dense_by_layer[best_layer][macro_key]),
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
    probe_results_by_validation_seed: dict[int, dict[str, object]],
    validation_seed_values: list[int],
    validation_seed_to_generator_seed: dict[int, int],
    probe_seed: int | None,
    runtime_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    categorical_by_layer = _aggregate_layer_payloads(
        probe_results_by_validation_seed=probe_results_by_validation_seed,
        validation_seed_values=validation_seed_values,
        section_key="categorical",
    )
    dense_by_layer = _aggregate_layer_payloads(
        probe_results_by_validation_seed=probe_results_by_validation_seed,
        validation_seed_values=validation_seed_values,
        section_key="dense",
    )
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
    shared = _aggregate_shared_fields(
        probe_results_by_validation_seed=probe_results_by_validation_seed,
        validation_seed_values=validation_seed_values,
        validation_seed_to_generator_seed=validation_seed_to_generator_seed,
        probe_seed=probe_seed,
    )
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
        "runtime": copy.deepcopy(runtime_summary) if runtime_summary is not None else {},
        "results": {
            "categorical": {_layer_key(layer): payload for layer, payload in categorical_by_layer.items()},
            "dense": {_layer_key(layer): payload for layer, payload in dense_by_layer.items()},
            "shared": shared,
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
            _stat_median(row[metric_name]) for row in rows if str(row[group_key]) == category
        ]
        if not category_values:
            raise ValueError(
                f"Missing dense values for {group_key}={category!r} metric={metric_name!r}"
            )
        values.append(float(np.median(np.asarray(category_values, dtype=float))))
    return values
