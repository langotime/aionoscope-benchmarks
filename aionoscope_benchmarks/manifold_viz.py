from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


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


def write_visualization_bundle(
    *,
    out_dir: Path,
    stem: str,
    plot_data: dict[str, Any],
    metrics: dict[str, Any],
    title: str,
) -> dict[str, str]:
    del metrics, title
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_data_path = out_dir / f"{stem}_plot_data.json"
    write_json(plot_data_path, plot_data)
    return {"plot_data_json": str(plot_data_path)}
