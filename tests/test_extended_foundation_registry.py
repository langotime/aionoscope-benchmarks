from __future__ import annotations

from aionoscope_benchmarks.constants import FOUNDATIONAL_MODELS
from aionoscope_benchmarks.model_registry import (
    MODEL_SPECS,
    all_foundational_model_names,
    canonical_model_name,
)


def test_registry_contains_explicit_versions_and_sizes_for_extended_families() -> None:
    expected = {
        "Chronos-2": ("amazon/chronos-2", "chronos"),
        "Mantis-UTICA-8M": ("fegounna/Utica", "mantis"),
        "MOMENT-1-Large": ("AutonLab/MOMENT-1-large", "moment"),
        "NuTime-Bias9": ("checkpoint_bias9.pth", "tivit"),
        "Timer-Base-84M": ("thuml/timer-base-84m", "timemoe"),
        "Sundial-Base-128M": ("thuml/sundial-base-128m", "timemoe"),
        "TTM-r2": ("ibm-granite/granite-timeseries-ttm-r2", "ttm"),
        "Time-MoE-50M": ("Maple728/TimeMoE-50M", "timemoe"),
        "Time-MoE-200M": ("Maple728/TimeMoE-200M", "timemoe"),
        "TimesFM-2.5-200M": ("google/timesfm-2.5-200m-pytorch", "core"),
        "Moirai-1.0-R-Small": ("Salesforce/moirai-1.0-R-small", "moirai"),
        "Moirai-1.0-R-Base": ("Salesforce/moirai-1.0-R-base", "moirai"),
        "Moirai-1.0-R-Large": ("Salesforce/moirai-1.0-R-large", "moirai"),
        "Moirai-1.1-R-Small": ("Salesforce/moirai-1.1-R-small", "moirai"),
        "Moirai-1.1-R-Base": ("Salesforce/moirai-1.1-R-base", "moirai"),
        "Moirai-1.1-R-Large": ("Salesforce/moirai-1.1-R-large", "moirai"),
        "Moirai-2.0-R-Small": ("Salesforce/moirai-2.0-R-small", "moirai"),
        "Moirai-MoE-1.0-R-Small": ("Salesforce/moirai-moe-1.0-R-small", "moirai"),
        "Moirai-MoE-1.0-R-Base": ("Salesforce/moirai-moe-1.0-R-base", "moirai"),
        "TabICL-v1": ("tabicl-classifier-v1-20250208.ckpt", "tabular"),
        "TabPFN-v2": ("Prior-Labs/TabPFN-v2-clf", "tabular"),
        "Kairos-10M": ("mldi-lab/Kairos_10m", "core"),
        "Kairos-23M": ("mldi-lab/Kairos_23m", "core"),
        "Kairos-50M": ("mldi-lab/Kairos_50m", "core"),
        "Reverso-Small-550K": ("shinfxh/reverso", "core"),
        "T-Loss-CricketX": ("CricketX_CausalCNN_encoder.pth", "core"),
        "TiConvNext-XXLarge-AugReg": (
            "laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg",
            "tivit",
        ),
        "TiViT-H-14-B79K": ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "tivit"),
        "Toto-Open-Base-1.0": ("Datadog/Toto-Open-Base-1.0", "toto"),
        "UniShape-ZeroShot": ("pretrained_model_ckpt/unishape_checkpoint_zeroshot.pth", "core"),
        "UniShape-FineTune": ("pretrained_model_ckpt/unishape_checkpoint_finetune.pth", "core"),
    }

    foundational_names = all_foundational_model_names()
    for model_name, (checkpoint, env) in expected.items():
        assert model_name in MODEL_SPECS
        assert MODEL_SPECS[model_name].checkpoint == checkpoint
        assert MODEL_SPECS[model_name].env == env
        assert model_name in FOUNDATIONAL_MODELS
        assert model_name in foundational_names


def test_legacy_aliases_still_resolve_to_canonical_explicit_names() -> None:
    assert canonical_model_name("Chronos2") == "Chronos-2"
    assert canonical_model_name("MOMENT") == "MOMENT-1-Large"
    assert canonical_model_name("Moirai") == "Moirai-1.1-R-Small"
    assert canonical_model_name("NuTime") == "NuTime-Bias9"
    assert canonical_model_name("T-Loss") == "T-Loss-CricketX"
    assert canonical_model_name("TTM") == "TTM-r2"
    assert canonical_model_name("TabICL") == "TabICL-v1"
    assert canonical_model_name("TabPFN") == "TabPFN-v2"
    assert canonical_model_name("TiConvNext") == "TiConvNext-XXLarge-AugReg"
    assert canonical_model_name("TiConvNext-XXLarge-laion2B-s34B-b82K-augreg") == "TiConvNext-XXLarge-AugReg"
    assert canonical_model_name("TiViT-H") == "TiViT-H-14-B79K"
    assert canonical_model_name("TiViT-ViT-H-14-laion2B-s32B-b79K") == "TiViT-H-14-B79K"
    assert canonical_model_name("Time-MoE-Base") == "Time-MoE-50M"
    assert canonical_model_name("Time-MoE-Large") == "Time-MoE-200M"
    assert canonical_model_name("Toto") == "Toto-Open-Base-1.0"
    assert canonical_model_name("UTICA") == "Mantis-UTICA-8M"
    assert "Moirai" not in FOUNDATIONAL_MODELS
    assert "LeNEPA-CauKer2M-20k" in FOUNDATIONAL_MODELS
