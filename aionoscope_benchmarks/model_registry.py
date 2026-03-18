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


@dataclass(frozen=True)
class ModelTaxonomy:
    family: str
    checkpoint_name: str
    architecture_role: str
    architecture_backbone: str
    training_paradigm: str

    def to_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "checkpoint_name": self.checkpoint_name,
            "architecture": {
                "role": self.architecture_role,
                "backbone": self.architecture_backbone,
            },
            "training": {
                "paradigm": self.training_paradigm,
            },
        }


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
    "Mantis-UTICA-8M": ModelSpec(
        name="Mantis-UTICA-8M",
        slug="Mantis-UTICA-8M",
        source="https://github.com/fegounna/Utica",
        checkpoint="fegounna/Utica",
        import_path="mantis-tsfm + huggingface_hub",
        env="mantis",
        module="aionoscope_benchmarks.adapters.utica",
        class_name="MantisUTICA8MAdapter",
    ),
    "TabPFN-v2": ModelSpec(
        name="TabPFN-v2",
        slug="TabPFN-v2",
        source="https://github.com/PriorLabs/TabPFN",
        checkpoint="Prior-Labs/TabPFN-v2-clf",
        import_path="tabpfn",
        env="tabular",
        module="aionoscope_benchmarks.adapters.tabpfn",
        class_name="TabPFNAdapter",
    ),
    "TabICL-v1": ModelSpec(
        name="TabICL-v1",
        slug="TabICL-v1",
        source="https://github.com/soda-inria/tabicl",
        checkpoint="tabicl-classifier-v1-20250208.ckpt",
        import_path="tabicl",
        env="tabular",
        module="aionoscope_benchmarks.adapters.tabicl",
        class_name="TabICLAdapter",
    ),
    "MOMENT-1-Large": ModelSpec(
        name="MOMENT-1-Large",
        slug="MOMENT-1-Large",
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
    "Chronos-2": ModelSpec(
        name="Chronos-2",
        slug="Chronos-2",
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
    "LeNEPA-CauKer2M-20k": ModelSpec(
        name="LeNEPA-CauKer2M-20k",
        slug="LeNEPA-CauKer2M-20k",
        source="https://huggingface.co/Natively-TS-Understanding/lenepa-cauker2m-5000-patchnorm-d256",
        checkpoint="Natively-TS-Understanding/lenepa-cauker2m-5000-patchnorm-d256",
        import_path="published inference.py via huggingface_hub",
        env="core",
        module="aionoscope_benchmarks.adapters.lenepa",
        class_name="LeNEPACauKer2M20KAdapter",
    ),
    "TTM-r2": ModelSpec(
        name="TTM-r2",
        slug="TTM-r2",
        source="https://github.com/ibm-granite/granite-tsfm",
        checkpoint="ibm-granite/granite-timeseries-ttm-r2",
        import_path="tsfm_public",
        env="ttm",
        module="aionoscope_benchmarks.adapters.ttm",
        class_name="TTMAdapter",
    ),
    "Time-MoE-50M": ModelSpec(
        name="Time-MoE-50M",
        slug="Time-MoE-50M",
        source="https://github.com/Time-MoE/Time-MoE",
        checkpoint="Maple728/TimeMoE-50M",
        import_path="transformers",
        env="timemoe",
        module="aionoscope_benchmarks.adapters.timemoe",
        class_name="TimeMoeBaseAdapter",
    ),
    "Time-MoE-200M": ModelSpec(
        name="Time-MoE-200M",
        slug="Time-MoE-200M",
        source="https://github.com/Time-MoE/Time-MoE",
        checkpoint="Maple728/TimeMoE-200M",
        import_path="transformers",
        env="timemoe",
        module="aionoscope_benchmarks.adapters.timemoe",
        class_name="TimeMoeLargeAdapter",
    ),
    "Timer-Base-84M": ModelSpec(
        name="Timer-Base-84M",
        slug="Timer-Base-84M",
        source="https://github.com/thuml/Timer",
        checkpoint="thuml/timer-base-84m",
        import_path="transformers",
        env="timemoe",
        module="aionoscope_benchmarks.adapters.thuml",
        class_name="TimerBase84MAdapter",
    ),
    "Sundial-Base-128M": ModelSpec(
        name="Sundial-Base-128M",
        slug="Sundial-Base-128M",
        source="https://github.com/thuml/Sundial",
        checkpoint="thuml/sundial-base-128m",
        import_path="transformers",
        env="timemoe",
        module="aionoscope_benchmarks.adapters.thuml",
        class_name="SundialBase128MAdapter",
    ),
    "TimesFM-2.5-200M": ModelSpec(
        name="TimesFM-2.5-200M",
        slug="TimesFM-2.5-200M",
        source="https://github.com/google-research/timesfm",
        checkpoint="google/timesfm-2.5-200m-pytorch",
        import_path="timesfm repo",
        env="core",
        module="aionoscope_benchmarks.adapters.timesfm",
        class_name="TimesFM25Adapter",
    ),
    "Moirai-1.0-R-Small": ModelSpec(
        name="Moirai-1.0-R-Small",
        slug="Moirai-1.0-R-Small",
        source="https://github.com/SalesforceAIResearch/uni2ts",
        checkpoint="Salesforce/moirai-1.0-R-small",
        import_path="uni2ts",
        env="moirai",
        module="aionoscope_benchmarks.adapters.moirai",
        class_name="Moirai10RSmallAdapter",
    ),
    "Moirai-1.0-R-Base": ModelSpec(
        name="Moirai-1.0-R-Base",
        slug="Moirai-1.0-R-Base",
        source="https://github.com/SalesforceAIResearch/uni2ts",
        checkpoint="Salesforce/moirai-1.0-R-base",
        import_path="uni2ts",
        env="moirai",
        module="aionoscope_benchmarks.adapters.moirai",
        class_name="Moirai10RBaseAdapter",
    ),
    "Moirai-1.0-R-Large": ModelSpec(
        name="Moirai-1.0-R-Large",
        slug="Moirai-1.0-R-Large",
        source="https://github.com/SalesforceAIResearch/uni2ts",
        checkpoint="Salesforce/moirai-1.0-R-large",
        import_path="uni2ts",
        env="moirai",
        module="aionoscope_benchmarks.adapters.moirai",
        class_name="Moirai10RLargeAdapter",
    ),
    "Moirai-1.1-R-Small": ModelSpec(
        name="Moirai-1.1-R-Small",
        slug="Moirai-1.1-R-Small",
        source="https://github.com/SalesforceAIResearch/uni2ts",
        checkpoint="Salesforce/moirai-1.1-R-small",
        import_path="uni2ts",
        env="moirai",
        module="aionoscope_benchmarks.adapters.moirai",
        class_name="Moirai11RSmallAdapter",
    ),
    "Moirai-1.1-R-Base": ModelSpec(
        name="Moirai-1.1-R-Base",
        slug="Moirai-1.1-R-Base",
        source="https://github.com/SalesforceAIResearch/uni2ts",
        checkpoint="Salesforce/moirai-1.1-R-base",
        import_path="uni2ts",
        env="moirai",
        module="aionoscope_benchmarks.adapters.moirai",
        class_name="Moirai11RBaseAdapter",
    ),
    "Moirai-1.1-R-Large": ModelSpec(
        name="Moirai-1.1-R-Large",
        slug="Moirai-1.1-R-Large",
        source="https://github.com/SalesforceAIResearch/uni2ts",
        checkpoint="Salesforce/moirai-1.1-R-large",
        import_path="uni2ts",
        env="moirai",
        module="aionoscope_benchmarks.adapters.moirai",
        class_name="Moirai11RLargeAdapter",
    ),
    "Moirai-2.0-R-Small": ModelSpec(
        name="Moirai-2.0-R-Small",
        slug="Moirai-2.0-R-Small",
        source="https://github.com/SalesforceAIResearch/uni2ts",
        checkpoint="Salesforce/moirai-2.0-R-small",
        import_path="uni2ts",
        env="moirai",
        module="aionoscope_benchmarks.adapters.moirai",
        class_name="Moirai20RSmallAdapter",
    ),
    "Moirai-MoE-1.0-R-Small": ModelSpec(
        name="Moirai-MoE-1.0-R-Small",
        slug="Moirai-MoE-1.0-R-Small",
        source="https://github.com/SalesforceAIResearch/uni2ts",
        checkpoint="Salesforce/moirai-moe-1.0-R-small",
        import_path="uni2ts",
        env="moirai",
        module="aionoscope_benchmarks.adapters.moirai",
        class_name="MoiraiMoE10RSmallAdapter",
    ),
    "Moirai-MoE-1.0-R-Base": ModelSpec(
        name="Moirai-MoE-1.0-R-Base",
        slug="Moirai-MoE-1.0-R-Base",
        source="https://github.com/SalesforceAIResearch/uni2ts",
        checkpoint="Salesforce/moirai-moe-1.0-R-base",
        import_path="uni2ts",
        env="moirai",
        module="aionoscope_benchmarks.adapters.moirai",
        class_name="MoiraiMoE10RBaseAdapter",
    ),
    "Kairos-10M": ModelSpec(
        name="Kairos-10M",
        slug="Kairos-10M",
        source="https://github.com/foundation-model-research/Kairos",
        checkpoint="mldi-lab/Kairos_10m",
        import_path="Kairos repo + transformers",
        env="core",
        module="aionoscope_benchmarks.adapters.kairos",
        class_name="Kairos10MAdapter",
    ),
    "Kairos-23M": ModelSpec(
        name="Kairos-23M",
        slug="Kairos-23M",
        source="https://github.com/foundation-model-research/Kairos",
        checkpoint="mldi-lab/Kairos_23m",
        import_path="Kairos repo + transformers",
        env="core",
        module="aionoscope_benchmarks.adapters.kairos",
        class_name="Kairos23MAdapter",
    ),
    "Kairos-50M": ModelSpec(
        name="Kairos-50M",
        slug="Kairos-50M",
        source="https://github.com/foundation-model-research/Kairos",
        checkpoint="mldi-lab/Kairos_50m",
        import_path="Kairos repo + transformers",
        env="core",
        module="aionoscope_benchmarks.adapters.kairos",
        class_name="Kairos50MAdapter",
    ),
    "Reverso-Small-550K": ModelSpec(
        name="Reverso-Small-550K",
        slug="Reverso-Small-550K",
        source="https://github.com/shinfxh/reverso",
        checkpoint="shinfxh/reverso",
        import_path="reverso_torch + huggingface_hub",
        env="core",
        module="aionoscope_benchmarks.adapters.reverso",
        class_name="ReversoSmall550KAdapter",
    ),
    "UniShape-ZeroShot": ModelSpec(
        name="UniShape-ZeroShot",
        slug="UniShape-ZeroShot",
        source="https://github.com/qianlima-lab/UniShape",
        checkpoint="pretrained_model_ckpt/unishape_checkpoint_zeroshot.pth",
        import_path="UniShape repo",
        env="core",
        module="aionoscope_benchmarks.adapters.unishape",
        class_name="UniShapeZeroShotAdapter",
    ),
    "UniShape-FineTune": ModelSpec(
        name="UniShape-FineTune",
        slug="UniShape-FineTune",
        source="https://github.com/qianlima-lab/UniShape",
        checkpoint="pretrained_model_ckpt/unishape_checkpoint_finetune.pth",
        import_path="UniShape repo",
        env="core",
        module="aionoscope_benchmarks.adapters.unishape",
        class_name="UniShapeFineTuneAdapter",
    ),
    "Toto-Open-Base-1.0": ModelSpec(
        name="Toto-Open-Base-1.0",
        slug="Toto-Open-Base-1.0",
        source="https://github.com/DataDog/toto",
        checkpoint="Datadog/Toto-Open-Base-1.0",
        import_path="toto-ts",
        env="toto",
        module="aionoscope_benchmarks.adapters.toto",
        class_name="TotoAdapter",
    ),
    "TiViT-H-14-B79K": ModelSpec(
        name="TiViT-H-14-B79K",
        slug="TiViT-H-14-B79K",
        source="https://github.com/ExplainableML/TiViT",
        checkpoint="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        import_path="ExplainableML/TiViT",
        env="tivit",
        module="aionoscope_benchmarks.adapters.tivit_h",
        class_name="TiViTHAdapter",
    ),
    "TiConvNext-XXLarge-AugReg": ModelSpec(
        name="TiConvNext-XXLarge-AugReg",
        slug="TiConvNext-XXLarge-AugReg",
        source="https://github.com/ExplainableML/TiViT",
        checkpoint="laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg",
        import_path="ExplainableML/TiViT",
        env="tivit",
        module="aionoscope_benchmarks.adapters.ticonvnext",
        class_name="TiConvNextAdapter",
    ),
    "NuTime-Bias9": ModelSpec(
        name="NuTime-Bias9",
        slug="NuTime-Bias9",
        source="https://github.com/chenguolin/NuTime",
        checkpoint="checkpoint_bias9.pth",
        import_path="NuTime repo",
        env="tivit",
        module="aionoscope_benchmarks.adapters.nutime",
        class_name="NuTimeAdapter",
    ),
    "T-Loss-CricketX": ModelSpec(
        name="T-Loss-CricketX",
        slug="T-Loss-CricketX",
        source="https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries",
        checkpoint="CricketX_CausalCNN_encoder.pth",
        import_path="USRLTS repo",
        env="core",
        module="aionoscope_benchmarks.adapters.tloss",
        class_name="TLossAdapter",
    ),
}

MODEL_ALIASES: dict[str, str] = {
    "Chronos2": "Chronos-2",
    "MOMENT": "MOMENT-1-Large",
    "Moirai": "Moirai-1.1-R-Small",
    "NuTime": "NuTime-Bias9",
    "T-Loss": "T-Loss-CricketX",
    "TTM": "TTM-r2",
    "TabICL": "TabICL-v1",
    "TabPFN": "TabPFN-v2",
    "TiConvNext": "TiConvNext-XXLarge-AugReg",
    "TiConvNext-XXLarge-laion2B-s34B-b82K-augreg": "TiConvNext-XXLarge-AugReg",
    "TiViT-H": "TiViT-H-14-B79K",
    "TiViT-ViT-H-14-laion2B-s32B-b79K": "TiViT-H-14-B79K",
    "Time-MoE-Base": "Time-MoE-50M",
    "Time-MoE-Large": "Time-MoE-200M",
    "Toto": "Toto-Open-Base-1.0",
    "UTICA": "Mantis-UTICA-8M",
}

MODEL_TAXONOMY: dict[str, ModelTaxonomy] = {
    "MantisV2": ModelTaxonomy(
        family="Mantis",
        checkpoint_name="V2",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="representation_ssl",
    ),
    "Mantis-UTICA-8M": ModelTaxonomy(
        family="Mantis",
        checkpoint_name="UTICA-8M",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="task_finetune",
    ),
    "TabPFN-v2": ModelTaxonomy(
        family="TabPFN",
        checkpoint_name="v2",
        architecture_role="tabular",
        architecture_backbone="tabular_transformer",
        training_paradigm="tabular_supervised",
    ),
    "TabICL-v1": ModelTaxonomy(
        family="TabICL",
        checkpoint_name="v1",
        architecture_role="tabular",
        architecture_backbone="tabular_transformer",
        training_paradigm="tabular_supervised",
    ),
    "MOMENT-1-Large": ModelTaxonomy(
        family="MOMENT",
        checkpoint_name="1-Large",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="representation_ssl",
    ),
    "TiRex": ModelTaxonomy(
        family="TiRex",
        checkpoint_name="TiRex",
        architecture_role="encoder",
        architecture_backbone="slstm",
        training_paradigm="representation_ssl",
    ),
    "Chronos-2": ModelTaxonomy(
        family="Chronos",
        checkpoint_name="2",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "LeNEPA-Aiono": ModelTaxonomy(
        family="LeNEPA",
        checkpoint_name="Aiono",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="representation_ssl",
    ),
    "LeNEPA-CauKer2M": ModelTaxonomy(
        family="LeNEPA",
        checkpoint_name="CauKer2M",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="representation_ssl",
    ),
    "LeNEPA-CauKer2M-20k": ModelTaxonomy(
        family="LeNEPA",
        checkpoint_name="CauKer2M-20k",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="representation_ssl",
    ),
    "TTM-r2": ModelTaxonomy(
        family="TTM",
        checkpoint_name="r2",
        architecture_role="encoder",
        architecture_backbone="mlp_mixer",
        training_paradigm="forecasting",
    ),
    "Time-MoE-50M": ModelTaxonomy(
        family="Time-MoE",
        checkpoint_name="50M",
        architecture_role="decoder",
        architecture_backbone="transformer_moe",
        training_paradigm="forecasting",
    ),
    "Time-MoE-200M": ModelTaxonomy(
        family="Time-MoE",
        checkpoint_name="200M",
        architecture_role="decoder",
        architecture_backbone="transformer_moe",
        training_paradigm="forecasting",
    ),
    "Timer-Base-84M": ModelTaxonomy(
        family="Timer",
        checkpoint_name="Base-84M",
        architecture_role="decoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "Sundial-Base-128M": ModelTaxonomy(
        family="Sundial",
        checkpoint_name="Base-128M",
        architecture_role="decoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "TimesFM-2.5-200M": ModelTaxonomy(
        family="TimesFM",
        checkpoint_name="2.5-200M",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "Moirai-1.0-R-Small": ModelTaxonomy(
        family="Moirai",
        checkpoint_name="1.0-R-Small",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "Moirai-1.0-R-Base": ModelTaxonomy(
        family="Moirai",
        checkpoint_name="1.0-R-Base",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "Moirai-1.0-R-Large": ModelTaxonomy(
        family="Moirai",
        checkpoint_name="1.0-R-Large",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "Moirai-1.1-R-Small": ModelTaxonomy(
        family="Moirai",
        checkpoint_name="1.1-R-Small",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "Moirai-1.1-R-Base": ModelTaxonomy(
        family="Moirai",
        checkpoint_name="1.1-R-Base",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "Moirai-1.1-R-Large": ModelTaxonomy(
        family="Moirai",
        checkpoint_name="1.1-R-Large",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "Moirai-2.0-R-Small": ModelTaxonomy(
        family="Moirai",
        checkpoint_name="2.0-R-Small",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "Moirai-MoE-1.0-R-Small": ModelTaxonomy(
        family="Moirai",
        checkpoint_name="MoE-1.0-R-Small",
        architecture_role="encoder",
        architecture_backbone="transformer_moe",
        training_paradigm="forecasting",
    ),
    "Moirai-MoE-1.0-R-Base": ModelTaxonomy(
        family="Moirai",
        checkpoint_name="MoE-1.0-R-Base",
        architecture_role="encoder",
        architecture_backbone="transformer_moe",
        training_paradigm="forecasting",
    ),
    "Kairos-10M": ModelTaxonomy(
        family="Kairos",
        checkpoint_name="10M",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "Kairos-23M": ModelTaxonomy(
        family="Kairos",
        checkpoint_name="23M",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "Kairos-50M": ModelTaxonomy(
        family="Kairos",
        checkpoint_name="50M",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "Reverso-Small-550K": ModelTaxonomy(
        family="Reverso",
        checkpoint_name="Small-550K",
        architecture_role="encoder",
        architecture_backbone="hybrid_sequence_model",
        training_paradigm="forecasting",
    ),
    "UniShape-ZeroShot": ModelTaxonomy(
        family="UniShape",
        checkpoint_name="ZeroShot",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="representation_ssl",
    ),
    "UniShape-FineTune": ModelTaxonomy(
        family="UniShape",
        checkpoint_name="FineTune",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="task_finetune",
    ),
    "Toto-Open-Base-1.0": ModelTaxonomy(
        family="Toto",
        checkpoint_name="Open-Base-1.0",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="forecasting",
    ),
    "TiViT-H-14-B79K": ModelTaxonomy(
        family="TiViT",
        checkpoint_name="H-14-B79K",
        architecture_role="encoder",
        architecture_backbone="vision_transformer",
        training_paradigm="cross_modal_transfer",
    ),
    "TiConvNext-XXLarge-AugReg": ModelTaxonomy(
        family="TiViT",
        checkpoint_name="ConvNext-XXLarge-AugReg",
        architecture_role="encoder",
        architecture_backbone="vision_convnet",
        training_paradigm="cross_modal_transfer",
    ),
    "NuTime-Bias9": ModelTaxonomy(
        family="NuTime",
        checkpoint_name="Bias9",
        architecture_role="encoder",
        architecture_backbone="transformer",
        training_paradigm="representation_ssl",
    ),
    "T-Loss-CricketX": ModelTaxonomy(
        family="T-Loss",
        checkpoint_name="CricketX",
        architecture_role="encoder",
        architecture_backbone="causal_cnn",
        training_paradigm="representation_ssl",
    ),
}

_MISSING_TAXONOMY = set(MODEL_SPECS) - set(MODEL_TAXONOMY)
_EXTRA_TAXONOMY = set(MODEL_TAXONOMY) - set(MODEL_SPECS)
if _MISSING_TAXONOMY or _EXTRA_TAXONOMY:
    raise RuntimeError(
        "MODEL_TAXONOMY must cover the canonical registry exactly: "
        f"missing={sorted(_MISSING_TAXONOMY)} extra={sorted(_EXTRA_TAXONOMY)}"
    )


def canonical_model_name(name: str) -> str:
    alias = MODEL_ALIASES.get(name)
    if alias is not None:
        return alias
    if name in MODEL_SPECS:
        return name
    for key, spec in MODEL_SPECS.items():
        if name == spec.slug:
            return key
    raise KeyError(f"Unknown model name: {name!r}")


def model_taxonomy(name: str) -> ModelTaxonomy:
    key = canonical_model_name(name)
    try:
        return MODEL_TAXONOMY[key]
    except KeyError as error:
        raise KeyError(f"Missing model taxonomy for canonical name: {key!r}") from error


def create_adapter(model_name: str):
    key = canonical_model_name(model_name)
    spec = MODEL_SPECS[key]
    module = importlib.import_module(spec.module)
    cls = getattr(module, spec.class_name)
    adapter = cls()
    return spec, adapter


def all_foundational_model_names() -> list[str]:
    return [name for name in FOUNDATIONAL_MODELS if name in MODEL_SPECS]
