from __future__ import annotations

import importlib
from dataclasses import dataclass

from .constants import FOUNDATIONAL_MODELS


@dataclass(frozen=True)
class ModelSpec:
    name: str
    slug: str
    source: str
    checkpoint: str
    import_path: str
    env: str
    module: str
    class_name: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "MantisV2": ModelSpec(
        name="MantisV2",
        slug="MantisV2",
        source="https://github.com/vfeofanov/mantis",
        checkpoint="paris-noah/MantisV2",
        import_path="mantis-tsfm",
        env="mantis",
        module="aionoscope_benchmarks.adapters.mantisv2",
        class_name="MantisV2Adapter",
    ),
    "TabPFN": ModelSpec(
        name="TabPFN",
        slug="TabPFN",
        source="https://github.com/PriorLabs/TabPFN",
        checkpoint="Prior-Labs/tabpfn_2_5",
        import_path="tabpfn",
        env="tabular",
        module="aionoscope_benchmarks.adapters.tabpfn",
        class_name="TabPFNAdapter",
    ),
    "TabPFN-TS": ModelSpec(
        name="TabPFN-TS",
        slug="TabPFN-TS",
        source="https://github.com/PriorLabs/tabpfn-time-series",
        checkpoint="tabpfn-v2-regressor-2noar4o2.ckpt",
        import_path="tabpfn_time_series",
        env="tabular",
        module="aionoscope_benchmarks.adapters.tabpfn_ts",
        class_name="TabPFNTSAdapter",
    ),
    "TabICL": ModelSpec(
        name="TabICL",
        slug="TabICL",
        source="https://github.com/soda-inria/tabicl",
        checkpoint="tabicl-classifier-v1-20250208.ckpt",
        import_path="tabicl",
        env="tabular",
        module="aionoscope_benchmarks.adapters.tabicl",
        class_name="TabICLAdapter",
    ),
    "TabICLForecaster": ModelSpec(
        name="TabICLForecaster",
        slug="TabICLForecaster",
        source="https://github.com/soda-inria/tabicl",
        checkpoint="tabicl-regressor-v2-20260212.ckpt",
        import_path="tabicl[forecast]",
        env="tabular",
        module="aionoscope_benchmarks.adapters.tabicl_forecaster",
        class_name="TabICLForecasterAdapter",
    ),
    "MOMENT": ModelSpec(
        name="MOMENT",
        slug="MOMENT",
        source="https://github.com/moment-timeseries-foundation-model/moment",
        checkpoint="AutonLab/MOMENT-1-large",
        import_path="momentfm",
        env="moment",
        module="aionoscope_benchmarks.adapters.moment",
        class_name="MomentAdapter",
    ),
    "TiRex": ModelSpec(
        name="TiRex",
        slug="TiRex",
        source="https://github.com/NX-AI/tirex",
        checkpoint="NX-AI/TiRex",
        import_path="tirex-ts",
        env="tirex",
        module="aionoscope_benchmarks.adapters.tirex",
        class_name="TiRexAdapter",
    ),
    "Chronos2": ModelSpec(
        name="Chronos2",
        slug="Chronos2",
        source="https://github.com/amazon-science/chronos-forecasting",
        checkpoint="amazon/chronos-2",
        import_path="chronos-forecasting",
        env="chronos",
        module="aionoscope_benchmarks.adapters.chronos2",
        class_name="Chronos2Adapter",
    ),
    "LeNEPA-Aiono": ModelSpec(
        name="LeNEPA-Aiono",
        slug="LeNEPA-Aiono",
        source="https://huggingface.co/Natively-TS-Understanding/lenepa-encoder-aiono",
        checkpoint="Natively-TS-Understanding/lenepa-encoder-aiono",
        import_path="published inference.py via huggingface_hub",
        env="core",
        module="aionoscope_benchmarks.adapters.lenepa",
        class_name="LeNEPAAionoAdapter",
    ),
    "LeNEPA-CauKer2M": ModelSpec(
        name="LeNEPA-CauKer2M",
        slug="LeNEPA-CauKer2M",
        source="https://huggingface.co/Natively-TS-Understanding/lenepa-cauker2m-5000-patchnorm-d256-steps200k",
        checkpoint="Natively-TS-Understanding/lenepa-cauker2m-5000-patchnorm-d256-steps200k",
        import_path="published inference.py via huggingface_hub",
        env="core",
        module="aionoscope_benchmarks.adapters.lenepa",
        class_name="LeNEPACauKer2MAdapter",
    ),
    "LeNEPA-CauKer-5k": ModelSpec(
        name="LeNEPA-CauKer-5k",
        slug="LeNEPA-CauKer-5k",
        source="https://huggingface.co/Natively-TS-Understanding/lenepa-cauker2m-5000-patchnorm-d256",
        checkpoint="Natively-TS-Understanding/lenepa-cauker2m-5000-patchnorm-d256",
        import_path="published inference.py via huggingface_hub",
        env="core",
        module="aionoscope_benchmarks.adapters.lenepa",
        class_name="LeNEPACauKer5KAdapter",
    ),
    "TTM": ModelSpec(
        name="TTM",
        slug="TTM",
        source="https://github.com/ibm-granite/granite-tsfm",
        checkpoint="ibm-granite/granite-timeseries-ttm-r2",
        import_path="tsfm_public",
        env="ttm",
        module="aionoscope_benchmarks.adapters.ttm",
        class_name="TTMAdapter",
    ),
    "Moirai": ModelSpec(
        name="Moirai",
        slug="Moirai",
        source="https://github.com/SalesforceAIResearch/uni2ts",
        checkpoint="Salesforce/moirai-1.1-R-small",
        import_path="uni2ts",
        env="moirai",
        module="aionoscope_benchmarks.adapters.moirai",
        class_name="MoiraiAdapter",
    ),
    "Toto": ModelSpec(
        name="Toto",
        slug="Toto",
        source="https://github.com/DataDog/toto",
        checkpoint="Datadog/Toto-Open-Base-1.0",
        import_path="toto-ts",
        env="toto",
        module="aionoscope_benchmarks.adapters.toto",
        class_name="TotoAdapter",
    ),
    "TiViT-H": ModelSpec(
        name="TiViT-H",
        slug="TiViT-H",
        source="https://github.com/ExplainableML/TiViT",
        checkpoint="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        import_path="ExplainableML/TiViT",
        env="tivit",
        module="aionoscope_benchmarks.adapters.tivit_h",
        class_name="TiViTHAdapter",
    ),
    "TiConvNext": ModelSpec(
        name="TiConvNext",
        slug="TiConvNext",
        source="https://github.com/ExplainableML/TiViT",
        checkpoint="laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg",
        import_path="ExplainableML/TiViT",
        env="tivit",
        module="aionoscope_benchmarks.adapters.ticonvnext",
        class_name="TiConvNextAdapter",
    ),
    "NuTime": ModelSpec(
        name="NuTime",
        slug="NuTime",
        source="https://github.com/chenguolin/NuTime",
        checkpoint="checkpoint_bias9.pth",
        import_path="NuTime repo",
        env="tivit",
        module="aionoscope_benchmarks.adapters.nutime",
        class_name="NuTimeAdapter",
    ),
    "T-Loss": ModelSpec(
        name="T-Loss",
        slug="T-Loss",
        source="https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries",
        checkpoint="https://data.lip6.fr/usrlts/",
        import_path="USRLTS repo",
        env="core",
        module="aionoscope_benchmarks.adapters.tloss",
        class_name="TLossAdapter",
    ),
}


def canonical_model_name(name: str) -> str:
    if name in MODEL_SPECS:
        return name
    for key, spec in MODEL_SPECS.items():
        if name == spec.slug:
            return key
    raise KeyError(f"Unknown model name: {name!r}")


def create_adapter(model_name: str):
    key = canonical_model_name(model_name)
    spec = MODEL_SPECS[key]
    module = importlib.import_module(spec.module)
    cls = getattr(module, spec.class_name)
    adapter = cls()
    return spec, adapter


def all_foundational_model_names() -> list[str]:
    return [name for name in FOUNDATIONAL_MODELS if name in MODEL_SPECS]
