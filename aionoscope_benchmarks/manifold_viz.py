from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

_DISTANCE_PLOT_KEYS = ("latent_distance", "linear_distance", "geodesic_distance")
_DISTANCE_META_KEYS = (
    "downsampled",
    "source_grid_points",
    "centroid_grid_points",
    "distance_grid_points",
    "distance_downsampled",
    "distance_plot_indices",
    "plot_grid_points",
    "plot_indices",
    "selected_geodesic_k",
    "geometry",
    "period",
)
_PLOT_DATA_DROP_KEYS = (
    *_DISTANCE_PLOT_KEYS,
    "downsampled",
    "distance_grid_points",
    "distance_downsampled",
    "distance_plot_indices",
    "plot_grid_points",
    "plot_indices",
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")


def _artifact_ref(path: Path, root: Path | None) -> str:
    """Portable artifact reference for JSON. Relative to ``root`` (the manifold
    artifact root) when given, so stored paths survive moving/renaming the tree
    and resolve directly against the data base URL; absolute only as a fallback."""
    if root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(root.resolve())).replace(os.sep, "/")
    except ValueError:
        return os.path.relpath(path.resolve(), root.resolve()).replace(os.sep, "/")


def write_visualization_bundle(
    *,
    out_dir: Path,
    stem: str,
    plot_data: dict[str, Any],
    metrics: dict[str, Any],
    title: str,
    root: Path | None = None,
) -> dict[str, str]:
    del metrics, title
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_data_path = out_dir / f"{stem}_plot_data.json"
    distance_data_path = out_dir / f"{stem}_distance_data.json"
    distance_data = {
        key: plot_data[key]
        for key in _DISTANCE_PLOT_KEYS
        if key in plot_data
    }
    distance_data.update(
        {
            key: plot_data[key]
            for key in _DISTANCE_META_KEYS
            if key in plot_data
        }
    )
    path_data = {
        key: value
        for key, value in plot_data.items()
        if key not in _PLOT_DATA_DROP_KEYS
    }
    path_data["distance_data_json"] = _artifact_ref(distance_data_path, root)
    write_json(plot_data_path, path_data)
    write_json(distance_data_path, distance_data)
    return {
        "plot_data_json": _artifact_ref(plot_data_path, root),
        "distance_data_json": _artifact_ref(distance_data_path, root),
    }
