"""Fixed-feature and metric-floor baselines for Aionoscope benchmark runs."""

from .features import (
    BASELINE_ALIASES,
    BASELINE_LAYER,
    DEFAULT_FEATURE_BATCH_SIZE,
    PAPER_CRITICAL_BASELINES,
    BaselineSpec,
    baseline_names,
    collect_split_features,
    get_baseline_spec,
    resolve_baseline_names,
)

__all__ = [
    "BASELINE_ALIASES",
    "BASELINE_LAYER",
    "DEFAULT_FEATURE_BATCH_SIZE",
    "PAPER_CRITICAL_BASELINES",
    "BaselineSpec",
    "baseline_names",
    "collect_split_features",
    "get_baseline_spec",
    "resolve_baseline_names",
]
