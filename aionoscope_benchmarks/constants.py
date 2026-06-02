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
    "Mantis-8M",
    "MantisPlus",
    "MantisV2",
    "Mantis-UTICA-8M",
    "TabPFN-v2",
    "TabICL-v1",
    "MOMENT-1-Large",
    "TiRex",
    "Chronos-2",
    "LeNEPA-Aiono",
    "LeNEPA-CauKer2M",
    "LeNEPA-CauKer2M-20k",
    "TTM-r2",
    "Time-MoE-50M",
    "Time-MoE-200M",
    "TempoPFN-38M",
    "EIDOS",
    "Timer-Base-84M",
    "Sundial-Base-128M",
    "TimesFM-2.5-200M",
    "Moirai-1.0-R-Small",
    "Moirai-1.0-R-Base",
    "Moirai-1.0-R-Large",
    "Moirai-1.1-R-Small",
    "Moirai-1.1-R-Base",
    "Moirai-1.1-R-Large",
    "Moirai-2.0-R-Small",
    "Moirai-MoE-1.0-R-Small",
    "Moirai-MoE-1.0-R-Base",
    "Kairos-10M",
    "Kairos-23M",
    "Kairos-50M",
    "Reverso-Small-550K",
    "UniShape-ZeroShot",
    "UniShape-FineTune",
    "Toto-Open-Base-1.0",
    "Toto-2.0-4M",
    "Toto-2.0-22M",
    "Toto-2.0-313M",
    "Toto-2.0-1B",
    "Toto-2.0-2.5B",
    "TiViT-H-14-B79K",
    "TiConvNext-XXLarge-AugReg",
    "NuTime-Bias9",
    "T-Loss-CricketX",
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
