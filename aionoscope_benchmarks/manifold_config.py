from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import DATASET_CONFIG_PATH, RESULTS_ROOT


DEFAULT_MANIFOLD_CALIBRATION_MODELS = (
    "MantisV2",
    "LeNEPA-Aiono",
    "Chronos-2",
    "Toto-Open-Base-1.0",
)

DEFAULT_MANIFOLD_CALIBRATION_TARGETS = (
    "sine_phase",
    "sine_frequency_hz",
    "sine_amplitude",
    "spike_time_frac",
    "gaussian_time_frac",
    "linear_trend_slope",
)

DEFAULT_MANIFOLD_GEODESIC_NEIGHBORS = (4, 6, 8)


def _as_str_tuple(value: object, *, default: tuple[str, ...], name: str) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)  # type: ignore[union-attr]
    except TypeError as exc:
        raise ValueError(f"{name} must be a string or iterable of strings") from exc


def _as_int_tuple(value: object, *, default: tuple[int, ...], name: str) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (int(value),)
    try:
        return tuple(int(item) for item in value)  # type: ignore[union-attr]
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer or iterable of integers") from exc


@dataclass(frozen=True)
class ManifoldEvalConfig:
    enabled: bool = True
    mode: str = "controlled_factor_slices"
    protocol_version: str = "calibration_v0"
    models: tuple[str, ...] = DEFAULT_MANIFOLD_CALIBRATION_MODELS
    targets: tuple[str, ...] = DEFAULT_MANIFOLD_CALIBRATION_TARGETS
    num_enabled: int = 1
    grid_size_1d: int = 32
    repeats_per_grid_point: int = 1
    fixed_factor_policy: str = "canonical_non_degenerate"
    nuisance_policy: str = "none_for_canonical_curve"
    validation_policy: str = "half_grid_offset"
    pca_dim: int = 64
    geodesic_neighbors: tuple[int, ...] = DEFAULT_MANIFOLD_GEODESIC_NEIGHBORS
    view_grid_mode: str = "linear"
    view_range_max_abs: float | None = None
    view_log_min_abs: float = 1e-6
    plot_max_points: int = 256
    dataset_config_path: Path = DATASET_CONFIG_PATH
    artifact_root: Path = RESULTS_ROOT / "manifold_calibration"
    write_plots: bool = True
    write_viewer: bool = True

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.mode != "controlled_factor_slices":
            raise ValueError(f"Unsupported manifold eval mode: {self.mode!r}")
        if self.protocol_version != "calibration_v0":
            raise ValueError(
                "Only manifold calibration protocol_version='calibration_v0' is supported, "
                f"got {self.protocol_version!r}"
            )
        if int(self.num_enabled) != 1:
            raise ValueError(f"Calibration requires num_enabled=1, got {self.num_enabled}")
        if int(self.grid_size_1d) < 4:
            raise ValueError(f"grid_size_1d must be >= 4, got {self.grid_size_1d}")
        if int(self.repeats_per_grid_point) < 1:
            raise ValueError(
                "repeats_per_grid_point must be >= 1, "
                f"got {self.repeats_per_grid_point}"
            )
        if not self.models:
            raise ValueError("models must be non-empty")
        if not self.targets:
            raise ValueError("targets must be non-empty")
        if int(self.pca_dim) < 1:
            raise ValueError(f"pca_dim must be >= 1, got {self.pca_dim}")
        if not self.geodesic_neighbors:
            raise ValueError("geodesic_neighbors must be non-empty")
        if any(int(k) < 0 for k in self.geodesic_neighbors):
            raise ValueError(f"geodesic_neighbors must be >= 0, got {self.geodesic_neighbors}")
        if self.view_grid_mode not in {"linear", "log", "signed_log"}:
            raise ValueError(
                "view_grid_mode must be one of ['linear', 'log', 'signed_log'], "
                f"got {self.view_grid_mode!r}"
            )
        if self.view_range_max_abs is not None and float(self.view_range_max_abs) <= 0:
            raise ValueError(
                "view_range_max_abs must be positive when provided, "
                f"got {self.view_range_max_abs!r}"
            )
        if float(self.view_log_min_abs) <= 0:
            raise ValueError(f"view_log_min_abs must be positive, got {self.view_log_min_abs}")
        if int(self.plot_max_points) < 4:
            raise ValueError(f"plot_max_points must be >= 4, got {self.plot_max_points}")
        if self.fixed_factor_policy != "canonical_non_degenerate":
            raise ValueError(
                "Calibration requires fixed_factor_policy='canonical_non_degenerate', "
                f"got {self.fixed_factor_policy!r}"
            )
        if self.nuisance_policy != "none_for_canonical_curve":
            raise ValueError(
                "Calibration requires nuisance_policy='none_for_canonical_curve', "
                f"got {self.nuisance_policy!r}"
            )
        if self.validation_policy != "half_grid_offset":
            raise ValueError(
                "Calibration requires validation_policy='half_grid_offset', "
                f"got {self.validation_policy!r}"
            )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ManifoldEvalConfig:
        if not isinstance(raw, dict):
            raise ValueError(f"Manifold config must be a dict, got {type(raw).__name__}")
        targets_raw = raw.get("targets", DEFAULT_MANIFOLD_CALIBRATION_TARGETS)
        if isinstance(targets_raw, dict):
            targets_raw = targets_raw.get("include", DEFAULT_MANIFOLD_CALIBRATION_TARGETS)
        models_raw = raw.get("models", DEFAULT_MANIFOLD_CALIBRATION_MODELS)
        geodesic_neighbors_raw = raw.get(
            "geodesic_neighbors",
            DEFAULT_MANIFOLD_GEODESIC_NEIGHBORS,
        )
        artifacts = raw.get("artifacts", {})
        if artifacts is None:
            artifacts = {}
        if not isinstance(artifacts, dict):
            raise ValueError(
                "manifold_eval.artifacts must be a dict when provided, "
                f"got {type(artifacts).__name__}"
            )
        config = cls(
            enabled=bool(raw.get("enabled", True)),
            mode=str(raw.get("mode", "controlled_factor_slices")),
            protocol_version=str(raw.get("protocol_version", "calibration_v0")),
            models=_as_str_tuple(
                models_raw,
                default=DEFAULT_MANIFOLD_CALIBRATION_MODELS,
                name="manifold_eval.models",
            ),
            targets=_as_str_tuple(
                targets_raw,
                default=DEFAULT_MANIFOLD_CALIBRATION_TARGETS,
                name="manifold_eval.targets",
            ),
            num_enabled=int(raw.get("num_enabled", 1)),
            grid_size_1d=int(raw.get("grid_size_1d", 32)),
            repeats_per_grid_point=int(raw.get("repeats_per_grid_point", 1)),
            fixed_factor_policy=str(
                raw.get("fixed_factor_policy", "canonical_non_degenerate")
            ),
            nuisance_policy=str(raw.get("nuisance_policy", "none_for_canonical_curve")),
            validation_policy=str(raw.get("validation_policy", "half_grid_offset")),
            pca_dim=int(raw.get("pca_dim", 64)),
            geodesic_neighbors=_as_int_tuple(
                geodesic_neighbors_raw,
                default=DEFAULT_MANIFOLD_GEODESIC_NEIGHBORS,
                name="manifold_eval.geodesic_neighbors",
            ),
            view_grid_mode=str(raw.get("view_grid_mode", "linear")),
            view_range_max_abs=(
                None
                if raw.get("view_range_max_abs") is None
                else float(raw.get("view_range_max_abs"))
            ),
            view_log_min_abs=float(raw.get("view_log_min_abs", 1e-6)),
            plot_max_points=int(raw.get("plot_max_points", 256)),
            dataset_config_path=Path(raw.get("dataset_config_path", DATASET_CONFIG_PATH)),
            artifact_root=Path(artifacts.get("root", RESULTS_ROOT / "manifold_calibration")),
            write_plots=bool(artifacts.get("write_plots", True)),
            write_viewer=bool(artifacts.get("write_viewer", True)),
        )
        config.validate()
        return config

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "protocol_version": self.protocol_version,
            "models": list(self.models),
            "targets": list(self.targets),
            "num_enabled": int(self.num_enabled),
            "grid_size_1d": int(self.grid_size_1d),
            "repeats_per_grid_point": int(self.repeats_per_grid_point),
            "fixed_factor_policy": self.fixed_factor_policy,
            "nuisance_policy": self.nuisance_policy,
            "validation_policy": self.validation_policy,
            "pca_dim": int(self.pca_dim),
            "geodesic_neighbors": [int(k) for k in self.geodesic_neighbors],
            "view_grid_mode": self.view_grid_mode,
            "view_range_max_abs": (
                None if self.view_range_max_abs is None else float(self.view_range_max_abs)
            ),
            "view_log_min_abs": float(self.view_log_min_abs),
            "plot_max_points": int(self.plot_max_points),
            "dataset_config_path": str(self.dataset_config_path),
            "artifact_root": str(self.artifact_root),
            "write_plots": bool(self.write_plots),
            "write_viewer": bool(self.write_viewer),
        }


@dataclass(frozen=True)
class ManifoldTargetGeometry:
    target_name: str
    component: str
    parameter: str
    geometry: str
    source: str
    node_or_view: str
    coordinate_name: str
    period: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "component": self.component,
            "parameter": self.parameter,
            "geometry": self.geometry,
            "source": self.source,
            "node_or_view": self.node_or_view,
            "coordinate_name": self.coordinate_name,
            "period": self.period,
            "notes": list(self.notes),
        }
