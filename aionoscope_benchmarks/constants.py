from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_ROOT = REPO_ROOT.parent / "aionoscope"
RESULTS_ROOT = REPO_ROOT / "results"
MODEL_RESULTS_ROOT = RESULTS_ROOT / "models"

BENCHMARK_DEFAULT_CHANNEL_SIZE = 5000

DATASET_CONFIG_PATH = REPO_ROOT / "configs" / "dataset_aiono_basic_components_balanced.yaml"
PROBE_CONFIG_PATH = REPO_ROOT / "configs" / "probe.yaml"
FOUNDATIONAL_MODELS_CONFIG_PATH = REPO_ROOT / "configs" / "models_foundational.yaml"

FOUNDATIONAL_MODELS = (
    "MantisV2",
    "TabPFN",
    "TabICL",
    "MOMENT",
    "TiRex",
    "Chronos2",
    "LeNEPA-Aiono",
    "LeNEPA-CauKer2M",
    "LeNEPA-CauKer2M-20k",
    "TTM",
    "Moirai",
    "Toto",
    "TiViT-H",
    "TiConvNext",
    "NuTime",
    "T-Loss",
)

TARGET_SIGNAL_ORDER = (
    "gaussian_noise",
    "uniform_noise",
    "random_walk_noise",
    "linear_trend",
    "quadratic_trend",
    "log_trend",
    "sigmoid_trend",
    "level_change",
    "spike",
    "gaussian",
    "sine",
    "sawtooth",
    "square",
)

TARGET_METRIC_ORDER = (
    "time_frac",
    "magnitude",
    "std",
    "slope",
    "intercept",
    "a",
    "center",
    "sharpness",
    "offset",
    "frequency_hz",
    "phase",
    "duty_cycle",
    "sigma_sec",
)
