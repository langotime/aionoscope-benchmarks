# Benchmark Models List

## Foundational Methods With Official Pretrained Weights

Naming policy for this file and the code registry:

- when a family has multiple official releases, benchmark entries use the exact official version and size in the canonical name, such as `TimesFM-2.5-200M` or `Moirai-1.1-R-Small`;
- prefer the official upstream repo plus the official Hugging Face checkpoint when one exists;
- if the only official published checkpoint is shipped directly in the repo, document that explicitly as a repo-hosted exception instead of treating it as an implicit fallback.

Under that policy, the benchmark currently does **not** include `Timer-S1`, `Reverso` (2.6M), or `Reverso-Nano`, because an official benchmarkable Hugging Face checkpoint was not identified for those exact variants. For univariate zero-shot forecasting, the official Timer repo uses the published `thuml/timer-base-84m` checkpoint as the `Timer-XL` checkpoint, so the benchmark keeps the exact published checkpoint name `Timer-Base-84M` instead of adding a second `Timer-XL` alias entry. The currently benchmarked official entries are the ones listed below.

- `MantisV2`
  - Source code: `https://github.com/vfeofanov/mantis`
  - Official checkpoint: `https://huggingface.co/paris-noah/MantisV2`
  - Import from: `mantis-tsfm` package; the PyPI docs map `MantisV2` to the `mantis.architecture` module and the `paris-noah/MantisV2` checkpoint.
  - Benchmark exact length: `512` samples. The official package says Mantis accepts any sequence length divisible by `32`, and recommends `512` because the pretrained model was trained at that length.
  - Paper note: this file keeps only the bare non-ensembled, non-fine-tuned `MantisV2` model.

- `Mantis-UTICA-8M`
  - Source code: `https://github.com/fegounna/Utica`
  - Official checkpoint: `https://huggingface.co/fegounna/Utica`
  - Import from: `mantis-tsfm`; the official UTICA README reuses `mantis.architecture.Mantis8M`, downloads `pytorch_model.bin` from `fegounna/Utica`, and loads it with `strict=False`.
  - Benchmark exact length: `512` samples, matching the official UTICA README resize target.
  - Note: this is a different training of the legacy Mantis-8M backbone, not an alias of `MantisV2`.

- `TabPFN`
  - Source code: `https://github.com/PriorLabs/TabPFN`
  - Official checkpoint: current official weights are hosted in `https://huggingface.co/Prior-Labs/tabpfn_2_5`. The current docs name the default classifier checkpoint `tabpfn-v2.5-classifier-v2.5_default.ckpt`.
  - Import from: `tabpfn`; use `from tabpfn import TabPFNClassifier`.
  - Benchmark exact length: `128` samples. The benchmark generates exact-length tabular fallback waveforms instead of downsampling longer waveforms inside the adapter.
  - Paper note: the MantisV2 paper pins `tabpfn==2.2.1`, so if exact paper reproduction matters you should pin that package version explicitly. For the `>10` class workaround, the paper points to `https://github.com/PriorLabs/tabpfn-extensions` rather than a core `tabpfn` API.

- `TabICL`
  - Source code: `https://github.com/soda-inria/tabicl`
  - Official checkpoint: the package auto-downloads official checkpoints from the project releases. The repository documents `tabicl-classifier-v1-20250208.ckpt` as the original ICML 2025 paper checkpoint, and also lists newer `v1.1` and `v2` checkpoints.
  - Import from: `tabicl`; use `from tabicl import TabICLClassifier`.
  - Benchmark exact length: `128` samples. The benchmark generates exact-length tabular fallback waveforms instead of downsampling longer waveforms inside the adapter.
  - Paper note: the MantisV2 paper pins `tabicl==0.1.3` and says it uses default parameters. If exact reproduction matters, pin both the package version and checkpoint explicitly because the project default checkpoint has changed across releases.

- `MOMENT`
  - Source code: `https://github.com/moment-timeseries-foundation-model/moment`
  - Official checkpoint: `https://huggingface.co/AutonLab/MOMENT-1-large`
  - Import from: `momentfm`; the official README uses `from momentfm import MOMENTPipeline`.
  - Benchmark exact length: `512` samples, matching the checkpoint config `seq_len`.
  - Paper note: the paper uses `MOMENT-1-large` and then truncates inference to the 10th Transformer layer for its own zero-shot comparison.

- `TiRex`
  - Source code: `https://github.com/NX-AI/tirex`
  - Official checkpoint: the official package examples load the public model id `NX-AI/TiRex`.
  - Import from: `tirex-ts`; the official package examples use `from tirex import load_model`.
  - Benchmark exact length: `2048` samples, matching the model's training context length.
  - Paper note: the paper pins `tirex-ts==1.1.1` and then reads the 5th layer for its own zero-shot comparison.

- `Chronos2`
  - Source code: `https://github.com/amazon-science/chronos-forecasting`
  - Official checkpoint: `https://huggingface.co/amazon/chronos-2`
  - Import from: `chronos-forecasting`; the official docs use `from chronos import Chronos2Pipeline`.
  - Benchmark exact length: `8192` samples, matching the checkpoint context length.
  - Paper note: the paper pins `chronos-forecasting==2.0.0` and then reads the 4th layer for its own zero-shot comparison.

- `LeNEPA-Aiono`
  - Source code: `https://huggingface.co/Natively-TS-Understanding/lenepa-encoder-aiono`
  - Official checkpoint: `https://huggingface.co/Natively-TS-Understanding/lenepa-encoder-aiono`
  - Import from: the published Hugging Face `inference.py` bundle; the benchmark loads that self-contained file via `huggingface_hub` and calls `load_lenepa_encoder(...)`.
  - Note: this is the balanced Aiono encoder-only export with a fixed `[B, 1, 5000]` input contract, `patch_size=25`, and `8` transformer blocks. In the benchmark, layer `0` is the tokenizer output and layer `8` is the post-final-layer-norm output.

- `LeNEPA-CauKer2M`
  - Source code: `https://huggingface.co/Natively-TS-Understanding/lenepa-cauker2m-5000-patchnorm-d256-steps200k`
  - Official checkpoint: `https://huggingface.co/Natively-TS-Understanding/lenepa-cauker2m-5000-patchnorm-d256-steps200k`
  - Import from: the published Hugging Face `inference.py` bundle; the benchmark loads that self-contained file via `huggingface_hub` and calls `load_lenepa_encoder(...)`.
  - Note: this is the CauKer2M encoder-only export with per-patch normalization inside the tokenizer, a fixed `[B, 1, 5000]` input contract, and `8` transformer blocks. In the benchmark, layer `0` is the tokenizer output and layer `8` is the post-final-layer-norm output.

- `LeNEPA-CauKer2M-20k`
  - Source code: `https://huggingface.co/Natively-TS-Understanding/lenepa-cauker2m-5000-patchnorm-d256`
  - Official checkpoint: `https://huggingface.co/Natively-TS-Understanding/lenepa-cauker2m-5000-patchnorm-d256`
  - Import from: the published Hugging Face `inference.py` bundle; the benchmark loads that self-contained file via `huggingface_hub` and calls `load_lenepa_encoder(...)`.
  - Note: this is the non-steps-suffixed CauKer encoder export with per-patch normalization inside the tokenizer, a fixed `[B, 1, 5000]` input contract, `patch_size=8`, and `8` transformer blocks. In the benchmark, layer `0` is the tokenizer output and layer `8` is the post-final-layer-norm output.

- `TTM (TinyTimeMixers, latest r2.1 release)`
  - Source code: `https://github.com/ibm-granite/granite-tsfm`
  - Official checkpoint: `https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2`
  - Import from: `granite-tsfm` / `tsfm_public`; use the IBM TTM classes from `tsfm_public` such as `TinyTimeMixerForPrediction`. Do not plan around `transformers` import for this model.
  - Benchmark exact length: `512` samples, matching the backbone patching sequence length.
  - Note: the current latest public TTM release is `r2.1` within the `granite-timeseries-ttm-r2` model card, not the older `r1` / `v1` release.

- `Time-MoE-Base`
  - Source code: `https://github.com/Time-MoE/Time-MoE`
  - Official checkpoint: `https://huggingface.co/Maple728/TimeMoE-50M`
  - Import from: `transformers`; use `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`. The upstream repo explicitly pins `transformers==4.40.1`.
  - Benchmark exact length: `4096` samples, matching the checkpoint `max_position_embeddings`.
  - Note: this is the official `50M` checkpoint. The benchmark applies the upstream per-series z-score normalization, then mean-pools the decoder token stream. Layer `0` is the input embedding stream and layer `12` is the post-final-norm decoder output.

- `Time-MoE-Large`
  - Source code: `https://github.com/Time-MoE/Time-MoE`
  - Official checkpoint: `https://huggingface.co/Maple728/TimeMoE-200M`
  - Import from: `transformers`; use `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`. The upstream repo explicitly pins `transformers==4.40.1`.
  - Benchmark exact length: `4096` samples, matching the checkpoint `max_position_embeddings`.
  - Note: this is the official `200M` checkpoint. The benchmark applies the upstream per-series z-score normalization, then mean-pools the decoder token stream. Layer `0` is the input embedding stream and layer `12` is the post-final-norm decoder output.

- `Timer-Base-84M`
  - Source code: `https://github.com/thuml/Timer`
  - Official checkpoint: `https://huggingface.co/thuml/timer-base-84m`
  - Import from: `transformers`; use `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`. The official model card pins `transformers==4.40.1`.
  - Benchmark exact length: `2880` samples, matching the official Timer model-card context length.
  - Note: this is the official published checkpoint used by the upstream `Timer-XL` univariate zero-shot forecasting flow. The benchmark keeps the published checkpoint name instead of adding a second `Timer-XL` alias row.

- `Sundial-Base-128M`
  - Source code: `https://github.com/thuml/Sundial`
  - Official checkpoint: `https://huggingface.co/thuml/sundial-base-128m`
  - Import from: `transformers`; use `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.
  - Benchmark exact length: `2880` samples, matching the official Sundial quickstart lookback length.
  - Note: this is the currently published official HF Sundial checkpoint.

- `TimesFM-2.5-200M`
  - Source code: `https://github.com/google-research/timesfm`
  - Official checkpoint: `https://huggingface.co/google/timesfm-2.5-200m-pytorch`
  - Import from: the official TimesFM repo; use `TimesFM_2p5_200M_torch.from_pretrained(...)`.
  - Benchmark exact length: `16384` samples, matching the official TimesFM 2.5 context limit.
  - Note: the benchmark intentionally names the exact published generation and size because the TimesFM family has multiple official versions (`1.0`, `2.0`, `2.5`).

- `Moirai-1.0-R-Small`
  - Source code: `https://github.com/SalesforceAIResearch/uni2ts`
  - Official checkpoint: `https://huggingface.co/Salesforce/moirai-1.0-R-small`
  - Import from: `uni2ts`; use the official `MoiraiModule` / `MoiraiForecast` API.
  - Benchmark exact length: `512` samples, matching `max_seq_len`.

- `Moirai-1.0-R-Base`
  - Source code: `https://github.com/SalesforceAIResearch/uni2ts`
  - Official checkpoint: `https://huggingface.co/Salesforce/moirai-1.0-R-base`
  - Import from: `uni2ts`; use the official `MoiraiModule` / `MoiraiForecast` API.
  - Benchmark exact length: `512` samples, matching `max_seq_len`.

- `Moirai-1.0-R-Large`
  - Source code: `https://github.com/SalesforceAIResearch/uni2ts`
  - Official checkpoint: `https://huggingface.co/Salesforce/moirai-1.0-R-large`
  - Import from: `uni2ts`; use the official `MoiraiModule` / `MoiraiForecast` API.
  - Benchmark exact length: `512` samples, matching `max_seq_len`.

- `Moirai-1.1-R-Small`
  - Source code: `https://github.com/SalesforceAIResearch/uni2ts`
  - Official checkpoint: `https://huggingface.co/Salesforce/moirai-1.1-R-small`
  - Import from: `uni2ts`; use the official `MoiraiModule` / `MoiraiForecast` API.
  - Benchmark exact length: `512` samples, matching `max_seq_len`.

- `Moirai-1.1-R-Base`
  - Source code: `https://github.com/SalesforceAIResearch/uni2ts`
  - Official checkpoint: `https://huggingface.co/Salesforce/moirai-1.1-R-base`
  - Import from: `uni2ts`; use the official `MoiraiModule` / `MoiraiForecast` API.
  - Benchmark exact length: `512` samples, matching `max_seq_len`.

- `Moirai-1.1-R-Large`
  - Source code: `https://github.com/SalesforceAIResearch/uni2ts`
  - Official checkpoint: `https://huggingface.co/Salesforce/moirai-1.1-R-large`
  - Import from: `uni2ts`; use the official `MoiraiModule` / `MoiraiForecast` API.
  - Benchmark exact length: `512` samples, matching `max_seq_len`.

- `Moirai-2.0-R-Small`
  - Source code: `https://github.com/SalesforceAIResearch/uni2ts`
  - Official checkpoint: `https://huggingface.co/Salesforce/moirai-2.0-R-small`
  - Import from: `uni2ts`; use the official `Moirai2Module` / `Moirai2Forecast` API.
  - Benchmark exact length: `512` samples, matching `max_seq_len`.

- `Moirai-MoE-1.0-R-Small`
  - Source code: `https://github.com/SalesforceAIResearch/uni2ts`
  - Official checkpoint: `https://huggingface.co/Salesforce/moirai-moe-1.0-R-small`
  - Import from: `uni2ts`; use the official `MoiraiMoEModule` / `MoiraiMoEForecast` API.
  - Benchmark exact length: `512` samples, matching `max_seq_len`.

- `Moirai-MoE-1.0-R-Base`
  - Source code: `https://github.com/SalesforceAIResearch/uni2ts`
  - Official checkpoint: `https://huggingface.co/Salesforce/moirai-moe-1.0-R-base`
  - Import from: `uni2ts`; use the official `MoiraiMoEModule` / `MoiraiMoEForecast` API.
  - Benchmark exact length: `512` samples, matching `max_seq_len`.

- `Kairos-10M`
  - Source code: `https://github.com/foundation-model-research/Kairos`
  - Official checkpoint: `https://huggingface.co/mldi-lab/Kairos_10m`
  - Import from: the official Kairos repo; load `KairosModel.from_pretrained(...)` after registering the upstream config/model classes.
  - Benchmark exact length: `2048` samples, matching the checkpoint context length.

- `Kairos-23M`
  - Source code: `https://github.com/foundation-model-research/Kairos`
  - Official checkpoint: `https://huggingface.co/mldi-lab/Kairos_23m`
  - Import from: the official Kairos repo; load `KairosModel.from_pretrained(...)` after registering the upstream config/model classes.
  - Benchmark exact length: `2048` samples, matching the checkpoint context length.

- `Kairos-50M`
  - Source code: `https://github.com/foundation-model-research/Kairos`
  - Official checkpoint: `https://huggingface.co/mldi-lab/Kairos_50m`
  - Import from: the official Kairos repo; load `KairosModel.from_pretrained(...)` after registering the upstream config/model classes.
  - Benchmark exact length: `2048` samples, matching the checkpoint context length.

- `Reverso-Small-550K`
  - Source code: `https://github.com/shinfxh/reverso`
  - Official checkpoint: `https://huggingface.co/shinfxh/reverso`
  - Import from: the official repo's `reverso_torch` implementation plus the official Hugging Face files under `checkpoints/reverso_small/`.
  - Benchmark exact length: `2048` samples, matching the published `reverso_small/args.json`.
  - Note: the official repo also documents `Reverso` and `Reverso-Nano`, but only `Reverso-Small` currently ships a benchmarkable checkpoint.

- `UniShape-ZeroShot`
  - Source code: `https://github.com/qianlima-lab/UniShape`
  - Official checkpoint: repo-hosted official checkpoint `pretrained_model_ckpt/unishape_checkpoint_zeroshot.pth`
  - Import from: the official UniShape repo; the benchmark reuses the published multiscale tokenization and transformer code directly.
  - Benchmark exact length: `512` samples, matching the repository-published resized series length used by the zero-shot script.
  - Note: this is the documented repo-hosted exception to the usual Hugging Face checkpoint rule.

- `UniShape-FineTune`
  - Source code: `https://github.com/qianlima-lab/UniShape`
  - Official checkpoint: repo-hosted official checkpoint `pretrained_model_ckpt/unishape_checkpoint_finetune.pth`
  - Import from: the official UniShape repo; the benchmark reuses the published multiscale tokenization and transformer code directly.
  - Benchmark exact length: `512` samples, matching the repository-published resized series length used by the fine-tune workflow.
  - Note: this is the documented repo-hosted exception to the usual Hugging Face checkpoint rule.

- `Toto`
  - Source code: `https://github.com/DataDog/toto`
  - Official checkpoint: `https://huggingface.co/Datadog/Toto-Open-Base-1.0`
  - Import from: `toto-ts`; the official quick-start example uses `from toto.inference.forecaster import TotoForecaster` and `from toto.data.util.dataset import MaskedTimeseries`.
  - Benchmark exact length: `4096` samples, matching the official Toto Open Base quick-start context length and remaining divisible by the model patch size `64`.
  - Note: this is Datadog's observability-focused open-weights time-series foundation model.

- `TiViT-H`
  - Source code: `https://github.com/ExplainableML/TiViT`
  - Official checkpoint: `https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
  - Import from: clone the `ExplainableML/TiViT` repo; the paper does not point to a separate pip package for TiViT itself.
  - Benchmark exact length: `5000` samples. The wrapper is length-agnostic in the time domain, so the benchmark keeps the default exact length with no adapter-side padding or cropping.
  - Paper note: this is the TiViT wrapper around a CLIP ViT-H backbone, and the paper reads the 14th layer for its own zero-shot comparison.

- `TiConvNext`
  - Source code: `https://github.com/ExplainableML/TiViT`
  - Official checkpoint: `https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg`
  - Import from: clone the `ExplainableML/TiViT` repo; the paper does not point to a separate pip package for TiConvNext itself.
  - Benchmark exact length: `5000` samples. The wrapper is length-agnostic in the time domain, so the benchmark keeps the default exact length with no adapter-side padding or cropping.
  - Paper note: this is the TiViT pipeline with a CLIP ConvNext backbone, and the paper reads the 15th layer for its own zero-shot comparison.

- `NuTime`
  - Source code: `https://github.com/chenguolin/NuTime`
  - Official checkpoint: `https://github.com/chenguolin/NuTime/blob/main/ckpt/checkpoint_bias9.pth`
  - Import from: clone the `NuTime` repo; the paper does not cite a separate pip package. The paper also points to `configs/demo_ft_epilepsy.json` for the architecture config it uses.
  - Benchmark exact length: `176` samples, matching `transform_size`.
  - Paper note: the paper does not use NuTime's adapter, because that would require fine-tuning and would break the frozen-encoder setup.

- `T-Loss`
  - Source code: `https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries`
  - Official checkpoint: the official repo states that pretrained models are downloadable from `https://data.lip6.fr/usrlts/`.
  - Import from: clone the repo; there is no published pip package. The repo exposes code through its repository modules and scripts rather than a documented package import.
  - Benchmark exact length: `5000` samples. The encoder itself is length-agnostic, so the benchmark keeps the default exact length with no adapter-side padding or cropping.
  - Paper note: in the MantisV2 bibliography this baseline corresponds to Franceschi et al. 2019, titled *Unsupervised scalable representation learning for multivariate time series*.

## Non-Foundational Methods Without Official Pretrained Weights

- `Catch22+`
  - Source code: `https://github.com/DynamicsAndNeuralSystems/pycatch22` for the Python wrapper, and `https://github.com/DynamicsAndNeuralSystems/catch22` for the core library.
  - Official checkpoint: no official public pretrained checkpoint exists.
  - Import from: `pycatch22`; use `import pycatch22`.
  - Paper note: `Catch22+` itself is not an upstream package. It is the paper's local augmentation `Catch22 + Stats`, where `Stats` means patchwise mean/std over 8 non-overlapping patches. You have to implement that augmentation around `pycatch22`.

- `CNN`
  - Source code: no official public source repository found for the exact Zhao et al. 2017 `CNN` paper cited by MantisV2.
  - Official checkpoint: no official public pretrained checkpoint found.
  - Import from: implement locally in this repo with `torch.nn.Conv1d`; no official upstream import target identified.
  - Note: use a 1D time-series CNN, not a 2D/image CNN. `https://github.com/hfawaz/dl-4-tsc` can be used only as architecture reference if needed, but the planned implementation target is a local `Conv1d` baseline.

- `FCN`
  - Source code: `https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline`
  - Official checkpoint: no official public pretrained checkpoint found.
  - Import from: clone the repo; no official pip package is published for these baselines.
  - Paper note: this repository is the original authors' codebase for the Wang et al. 2017 deep-learning baselines and contains the FCN implementation.

- `MLP`
  - Source code: `https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline`
  - Official checkpoint: no official public pretrained checkpoint found.
  - Import from: implement locally in this repo with PyTorch `nn.Linear` layers; no official upstream import target identified for a packaged baseline.
  - Note: use a classical fully connected MLP for flattened time-series inputs. The Wang et al. 2017 baseline repo can be used only as architecture reference if needed, but the planned implementation target is a local PyTorch MLP baseline.

- `ResNet`
  - Source code: `https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline`
  - Official checkpoint: no official public pretrained checkpoint found.
  - Import from: clone the repo; no official pip package is published for these baselines.
  - Paper note: this repository is the original authors' codebase for the Wang et al. 2017 deep-learning baselines and contains the ResNet implementation.

- `DTW`
  - Source code: no single official source repository or package import was identified for the `DTW` baseline as cited in the MantisV2 paper; it is an algorithmic baseline rather than a pretrained model.
  - Official checkpoint: no checkpoint exists.
  - Import from: no official import target identified.
  - Practical fallback: use a standard implementation from a time-series library if you need a runnable baseline, but that would be a reproduction rather than an official upstream model import.

- `TS2Vec`
  - Source code: `https://github.com/zhihanyue/ts2vec`
  - Official checkpoint: no official public pretrained checkpoint found.
  - Import from: clone the repo; the official README uses `from ts2vec import TS2Vec`.
  - Paper note: results are imported from prior work rather than rerun inside the MantisV2 paper.

- `TNC`
  - Source code: `https://github.com/sanatonek/TNC_representation_learning`
  - Official checkpoint: no official public pretrained checkpoint found.
  - Import from: clone the repo; the code is organized under the local `tnc/` package rather than a published PyPI package.
  - Paper note: results are imported from prior work rather than rerun inside the MantisV2 paper.

- `TS-TCC`
  - Source code: `https://github.com/emadeldeen24/TS-TCC`
  - Official checkpoint: no official public pretrained checkpoint found.
  - Import from: clone the repo; no official pip package is advertised by the authors.
  - Paper note: results are imported from prior work rather than rerun inside the MantisV2 paper.
