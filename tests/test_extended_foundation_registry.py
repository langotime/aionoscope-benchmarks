from __future__ import annotations

from aionoscope_benchmarks.constants import FOUNDATIONAL_MODELS
from aionoscope_benchmarks.model_registry import (
    MODEL_SPECS,
    TRAINING_PARADIGM_DEFINITIONS,
    all_foundational_model_names,
    canonical_model_name,
    model_taxonomy,
)


def test_registry_contains_explicit_versions_and_sizes_for_extended_families() -> None:
    expected = {
        "Chronos-2": ("amazon/chronos-2", "chronos"),
        "Mantis-8M": ("paris-noah/Mantis-8M", "mantis"),
        "MantisPlus": ("paris-noah/MantisPlus", "mantis"),
        "Mantis-UTICA-8M": ("fegounna/Utica", "mantis"),
        "MOMENT-1-Large": ("AutonLab/MOMENT-1-large", "moment"),
        "NuTime-Bias9": ("checkpoint_bias9.pth", "tivit"),
        "TempoPFN-38M": ("AutoML-org/TempoPFN", "tempopfn"),
        "EIDOS": ("external/EIDOS/eidos 1.pt", "timemoe"),
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
        "Toto-2.0-4M": ("Datadog/Toto-2.0-4m", "toto"),
        "Toto-2.0-22M": ("Datadog/Toto-2.0-22m", "toto"),
        "Toto-2.0-313M": ("Datadog/Toto-2.0-313m", "toto"),
        "Toto-2.0-1B": ("Datadog/Toto-2.0-1B", "toto"),
        "Toto-2.0-2.5B": ("Datadog/Toto-2.0-2.5B", "toto"),
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
    assert canonical_model_name("Mantis") == "Mantis-8M"
    assert canonical_model_name("Mantis+") == "MantisPlus"
    assert canonical_model_name("Mantis8M") == "Mantis-8M"
    assert canonical_model_name("MantisV1") == "Mantis-8M"
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
    assert canonical_model_name("TempoPFN") == "TempoPFN-38M"
    assert canonical_model_name("Eidos") == "EIDOS"
    assert canonical_model_name("Toto") == "Toto-Open-Base-1.0"
    assert canonical_model_name("UTICA") == "Mantis-UTICA-8M"
    assert "Moirai" not in FOUNDATIONAL_MODELS
    assert "LeNEPA-CauKer2M-20k" in FOUNDATIONAL_MODELS


def test_model_taxonomy_exposes_family_checkpoint_architecture_and_training_groups() -> None:
    mantis_plus = model_taxonomy("MantisPlus")
    assert mantis_plus.family == "Mantis"
    assert mantis_plus.checkpoint_name == "Plus"
    assert mantis_plus.architecture_backbone == "transformer_full_attention"
    assert mantis_plus.training_paradigm == "representation_ssl"

    lenepa = model_taxonomy("LeNEPA-Aiono")
    assert lenepa.family == "LeNEPA"
    assert lenepa.checkpoint_name == "Aiono"
    assert lenepa.architecture_backbone == "transformer_causal"
    assert lenepa.training_paradigm == "representation_ssl"

    time_moe = model_taxonomy("Time-MoE-200M")
    assert time_moe.family == "Time-MoE"
    assert time_moe.checkpoint_name == "200M"
    assert time_moe.architecture_backbone == "transformer_moe_causal"
    assert time_moe.training_paradigm == "forecasting"

    tempopfn = model_taxonomy("TempoPFN-38M")
    assert tempopfn.family == "TempoPFN"
    assert tempopfn.checkpoint_name == "38M"
    assert tempopfn.architecture_backbone == "linear_rnn"
    assert tempopfn.training_paradigm == "forecasting"

    eidos = model_taxonomy("EIDOS")
    assert eidos.family == "EIDOS"
    assert eidos.checkpoint_name == "EIDOS"
    assert eidos.architecture_backbone == "transformer_causal"
    assert eidos.training_paradigm == "forecasting"

    moirai = model_taxonomy("Moirai-1.1-R-Small")
    assert moirai.architecture_backbone == "transformer_full_attention"
    assert moirai.training_paradigm == "forecasting"

    moirai2 = model_taxonomy("Moirai-2.0-R-Small")
    assert moirai2.architecture_backbone == "transformer_causal"
    assert moirai2.training_paradigm == "forecasting"

    toto2 = model_taxonomy("Toto-2.0-313M")
    assert toto2.family == "Toto"
    assert toto2.checkpoint_name == "2.0-313M"
    assert toto2.architecture_backbone == "transformer_causal"
    assert toto2.training_paradigm == "forecasting"

    tivit = model_taxonomy("TiConvNext-XXLarge-AugReg")
    assert tivit.family == "TiViT"
    assert tivit.architecture_backbone == "vision_convnet"
    assert tivit.training_paradigm == "cross_modal_transfer"


def test_training_paradigms_are_pretraining_based_and_do_not_include_task_finetune() -> None:
    assert "task_finetune" not in TRAINING_PARADIGM_DEFINITIONS

    mantis_utica = model_taxonomy("Mantis-UTICA-8M")
    assert mantis_utica.training_paradigm == "representation_ssl"

    unishape_finetune = model_taxonomy("UniShape-FineTune")
    assert unishape_finetune.training_paradigm == "representation_ssl"
