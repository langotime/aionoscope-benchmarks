# Documentation

## What This Repo Does

This repo benchmarks frozen foundational time-series models on the balanced Aiono basic-components contract built from the sibling `aiono` library. Each run:

- rebuilds a deterministic finite benchmark split in memory;
- resolves the model's exact benchmark sequence length before dataset generation;
- extracts frozen representations from one or more model layers;
- trains linear probes for component classification and dense parameter regression;
- writes one JSON artifact per model into `results/models/`;
- optionally visualizes those artifacts in `results/dashboard.html`.

## Environment Layout

Base development uses `uv` and the editable sibling dependency declared in `pyproject.toml`.

```bash
uv sync
```

The full foundational sweep uses multiple pinned virtual environments because the model families do not share one compatible dependency set. The repo already encodes those interpreter paths in `scripts/run_foundational_sequential.py`.

`aionoscope_benchmarks.run_model` now performs a preflight dependency check before dataset generation or adapter loading. In particular, categorical probe metrics require `torchmetrics` in the active model-specific environment. If a model env drifts out of sync, repair it directly, for example:

```bash
uv pip install --python .venv-chronos/bin/python 'torchmetrics>=1.9,<2'
```

The `tabular` environment now also needs the forecasting extras for the new
forecast-derived tabular adapters:

```bash
uv pip install --python .venv-tabular/bin/python 'tabpfn-time-series' 'tabicl[forecast]'
```

The base `core` environment now also hosts the official `TimesFM`, `Kairos`,
`Reverso`, and `UniShape` integrations. `Kairos` follows the upstream repo's
published newer `transformers` stack, and `UniShape` needs `fastai` for the
official model code:

```bash
uv pip install --python .venv/bin/python 'transformers>=4.56,<4.57' 'jaxtyping>=0.3,<0.4' 'fastai<3'
```

Time-MoE, Timer, and Sundial use the official remote-code checkpoints and the
upstream THUML / Time-MoE published `transformers` pin. Install that in the
dedicated `.venv-timemoe` env:

```bash
uv pip install --python .venv-timemoe/bin/python 'transformers==4.40.1'
```

When `flash-attn` is available in `.venv-timemoe`, the adapter prefers
`flash_attention_2` on CUDA. Otherwise it falls back to eager attention and keeps
the default encode batch size at `1`.

## Common Commands

Run one model in the current environment:

```bash
uv run python -m aionoscope_benchmarks.run_model --model TiRex
```

Run a few models in the current environment:

```bash
uv run python -m aionoscope_benchmarks.run_many --model TiRex --model Chronos-2
```

Run all foundational models with the per-model pinned interpreters:

```bash
uv run python scripts/run_foundational_sequential.py
```

Be aware that exact model-native sequence lengths can materially increase RAM use because
the benchmark still materializes finite train and validation tensors in memory before
feature collection. `Chronos-2` is the heaviest case because it now runs at exact length
`8192`.

Serve the static dashboard:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/results/dashboard.html`.

## Config Files

### Dataset config

`configs/dataset_aiono_basic_components_balanced.yaml` defines the benchmark contract:

- sampling frequency and the default reference sequence length;
- the sequence-length policy (`model_native_exact` in the current benchmark);
- component library and `num_enabled`;
- training seed;
- ordered validation seed values and validation seed offset;
- train and validation batch counts;
- dense target definitions.

Changing this file changes the benchmark contract and must be treated as a benchmark-definition change, not a casual tuning knob.

`run_model.py` does not blindly use the config default length. It loads the adapter
first, resolves that model's exact benchmark sequence length, and passes that resolved
length into the runtime split builder. The manifest written to each JSON artifact stores:

- `default_channel_size`: the reference length from the YAML config;
- `channel_size`: the exact resolved length used to generate the run;
- `channel_size_policy`: the active policy string;
- `channel_size_source`: where the resolved exact length came from.

### Versioned periodic contract

The dataset config is now explicitly versioned:

- `benchmark_family: aiono_basic_components`
- `benchmark_version: v1`

`v1` semantics are resolved in the shared `aiono` library, not reimplemented in each
consumer. The benchmark repo calls
`resolve_aiono_basic_components_periodic_contract(...)` and writes the resolved result
into the runtime manifest.

The baseline invariants for `v1` are:

- `sampling_frequency = 500 Hz` everywhere
- `frequency_hz = auto` by default
- `sawtooth_min_points_per_period = 5`
- `square_min_points_in_shorter_plateau = 2`
- square-wave recoverability depends on the configured `square_duty_cycle` range

For a resolved sequence length `L`, the shared resolver computes:

- `duration_sec = (L - 1) / 500`
- `f_min_full_period = min_full_periods / duration_sec`
- `f_max_nyquist = nyquist_fraction * 500 / 2`
- `sawtooth_high = min(f_max_nyquist, 500 / sawtooth_min_points_per_period)`
- `square_high = min(f_max_nyquist, 500 * shorter_plateau_fraction / square_min_points_in_shorter_plateau)`

That means the lower bound rises on short exact-length adapters because the signal still
needs at least one full period to make `frequency_hz` recoverable. The benchmark no
longer uses one hidden hard-coded `0.2..6.0 Hz` range for every exact sequence length.

The manifest exposes the resolved periodic task through fields such as:

- `benchmark_family`
- `benchmark_version`
- `baseline_sampling_frequency_hz`
- `duration_sec`
- `periodic_frequency_mode`
- `periodic_frequency_resolution_source`
- `sine_frequency_hz_resolved_low/high`
- `sawtooth_frequency_hz_resolved_low/high`
- `square_frequency_hz_resolved_low/high`
- `square_duty_cycle_min/max`
- `square_frequency_hz_recoverability_upper_bound`
- `periodic_sampler_specs`

`periodic_sampler_specs` is the quickest way to inspect the full canonical periodic
sampler contract, including amplitude, phase, offset, resolved frequency bounds, and
square `duty_cycle`.

### Probe config

`configs/probe.yaml` controls linear probe training:

- number of steps;
- batch size;
- optimizer hyperparameters;
- gradient clipping;
- checkpoint interval used for selecting best probe checkpoints.

### Foundational model list

`configs/models_foundational.yaml` is the human-facing sweep list. The code-level source of truth for metadata and adapters is `aionoscope_benchmarks/model_registry.py`.
Canonical benchmark names must include the exact official version and size whenever a
family publishes multiple checkpoints. The sweep therefore uses names such as
`TimesFM-2.5-200M`, `Moirai-1.1-R-Small`, `Moirai-MoE-1.0-R-Base`,
`Toto-Open-Base-1.0`, and `Mantis-UTICA-8M` instead of ambiguous family-only
labels. When the only official published artifact is identified by a stable
variant token rather than a size marker, that token stays in the canonical name,
for example `NuTime-Bias9` or `T-Loss-CricketX`.
Registry entries are expected to point at the official upstream repo and, when one
exists, the official Hugging Face checkpoint. `UniShape` is the current explicit
documented exception: the official repo ships the published checkpoints directly under
`pretrained_model_ckpt/`, and no official Hugging Face checkpoint is used for that
integration.

The two LeNEPA entries (`LeNEPA-Aiono` and `LeNEPA-CauKer2M`) run in the base `core`
environment. On first use, their adapters download the published `inference.py`,
`lenepa_encoder_config.json`, and `lenepa_encoder.safetensors` files from the
checkpoint repository on Hugging Face and then reuse the local cache on subsequent runs.
They expose zero-indexed benchmark layers `0..8`; layer `0` is the tokenizer output and
layer `8` is the post-final-layer-norm encoder output, matching the published export
contract before mean pooling.
More generally, adapters that expose a distinct embedding stream use layer `0` for that
embedding and number transformer-style encoder blocks from `1`.
`Time-MoE-50M` and `Time-MoE-200M` also reserve layer `0` for their input
embedding stream. They load the official `Maple728/TimeMoE-50M` and
`Maple728/TimeMoE-200M` checkpoints through `transformers` remote code, pin the
exact benchmark length to the checkpoint `max_position_embeddings=4096`, apply the
official per-series z-score normalization to the exact benchmark waveform, and
mean-pool the causal token stream across time. Their final benchmark layer is the
post-final-norm decoder output.
`Timer-Base-84M` and `Sundial-Base-128M` run at the official `2880`-sample quickstart
lookback length and expose their remote-code decoder hidden states through the same
layer-0-plus-block-output convention. For univariate zero-shot forecasting, the
official Timer repo uses `thuml/timer-base-84m` as the published `Timer-XL`
checkpoint, so the benchmark keeps the exact published checkpoint name
`Timer-Base-84M` rather than introducing a second alias entry.
`TimesFM-2.5-200M` uses the official Google PyTorch checkpoint, keeps the exact context
length at `16384`, applies the upstream running-stat ReVIN prefill normalization used by
TimesFM decode, and mean-pools patch tokens across the tokenizer stream plus each
stacked transformer block.
The explicit `Moirai-*` and `Moirai-MoE-*` entries all pin exact context length to the
checkpoint `max_seq_len` and use the official `uni2ts` forecast helper packing path
before mean-pooling observed non-prediction tokens. The benchmark no longer keeps a
separate legacy `results/models/Moirai.json` artifact; current Moirai results live only
under the explicit versioned `Moirai-*` and `Moirai-MoE-*` names.
`Kairos-10M`, `Kairos-23M`, and `Kairos-50M` use the official Kairos repo code against
the official `mldi-lab/*` checkpoints, pin exact context length to `2048`, and expose
encoder hidden states after the official adaptive patching pipeline.
`Reverso-Small-550K` uses the official `shinfxh/reverso` Hugging Face repo together
with the official pure-PyTorch `reverso_torch` implementation from the upstream repo.
`UniShape-ZeroShot` and `UniShape-FineTune` use the official repo-hosted checkpoints and
the official multiscale tokenization code at the repository-published resized length of
`512`.
`Mantis-UTICA-8M` reuses the official `mantis-tsfm` `Mantis8M` backbone with the
official `fegounna/Utica` weights and uses the official README resize target of `512`.
`TabPFN-v2` names the current checked-in official classifier artifact, `TabICL-v1`
names the original ICML paper checkpoint, and the repo-hosted singleton checkpoints
keep their published variant tokens in the canonical benchmark names, such as
`NuTime-Bias9` and `T-Loss-CricketX`.

The current exact benchmark lengths are:

- `TimesFM-2.5-200M`: `16384`
- `Chronos-2`: `8192`
- `LeNEPA-Aiono`: `5000`
- `LeNEPA-CauKer2M`: `5000`
- `LeNEPA-CauKer2M-20k`: `5000`
- `MantisV2`: `512`
- `Mantis-UTICA-8M`: `512`
- `Moirai-1.0-R-Small`: `512`
- `Moirai-1.0-R-Base`: `512`
- `Moirai-1.0-R-Large`: `512`
- `Moirai-1.1-R-Small`: `512`
- `Moirai-1.1-R-Base`: `512`
- `Moirai-1.1-R-Large`: `512`
- `Moirai-2.0-R-Small`: `512`
- `Moirai-MoE-1.0-R-Small`: `512`
- `Moirai-MoE-1.0-R-Base`: `512`
- `MOMENT-1-Large`: `512`
- `NuTime-Bias9`: `176`
- `T-Loss-CricketX`: `5000`
- `Timer-Base-84M`: `2880`
- `Sundial-Base-128M`: `2880`
- `Time-MoE-50M`: `4096`
- `Time-MoE-200M`: `4096`
- `TTM-r2`: `512`
- `Kairos-10M`: `2048`
- `Kairos-23M`: `2048`
- `Kairos-50M`: `2048`
- `Reverso-Small-550K`: `2048`
- `UniShape-ZeroShot`: `512`
- `UniShape-FineTune`: `512`
- `TabICL-v1`: `128`
- `TabPFN-v2`: `128`
- `TiConvNext-XXLarge-AugReg`: `5000`
- `TiRex`: `2048`
- `TiViT-H-14-B79K`: `5000`
- `Toto-Open-Base-1.0`: `4096`

## Validation Seed Semantics

The benchmark distinguishes:

- validation seed values: user-facing identifiers stored in result payloads;
- validation generator seeds: actual dataset generator seeds derived by adding `validation_seed_offset`.

The train seed must not overlap with any validation generator seed. The manifest stored in the JSON records both the ordered validation seed values and the derived generator seeds.

## Result Artifacts

Each model run writes `results/models/<slug>.json`.

High-level structure:

- `model`: model identity, source, checkpoint, layers evaluated, adapter metadata, plus explicit dashboard taxonomy fields under `family`, `checkpoint_name`, `architecture.backbone`, and `training.paradigm`;
- `dataset`: the benchmark manifest used to build the train and validation splits, including the config default length and the exact resolved length used for the run;
- `probe_config`: probe hyperparameters plus the fixed `probe_seed`;
- `runtime`: wall-clock and device metadata plus explicit encoder forward train/validation/total timings;
- `results.categorical`: per-layer multi-label classification outputs;
- `results.dense`: per-layer dense regression outputs;
- `results.shared`: validation-run aggregates shared across sections;
- `results.summary`: best-layer summaries and oracle-per-target summaries.

Numeric values aggregated across validation seeds use the payload:

```json
{
  "values": [0.91, 0.92, 0.90],
  "median": 0.91,
  "std": 0.01,
  "n": 3
}
```

The dashboard expects this schema for aggregated runs.

### Dashboard Taxonomy Classes

The dashboard taxonomy is benchmark-path based. Do not classify a model from its
paper title alone; classify it from the token stream that the benchmark adapter
actually exposes and pools.

Architecture classes:

- `transformer_full_attention`: time-series transformer path whose pooled states come from full-context self-attention over the benchmark context. Padding or group masks are allowed, but there is no causal time mask on the pooled token stream. Use this for encoder-style paths such as `Chronos-2`, `Kairos-*`, `MOMENT-1-Large`, `Moirai-1.x-*`, `NuTime-Bias9`, `Toto-Open-Base-1.0`, `MantisV2`, `Mantis-UTICA-8M`, and `UniShape-*`.
- `transformer_causal`: time-series transformer path whose pooled states come from causal masking or a decoder-only token stream. Use this for `LeNEPA-*`, `Timer-Base-84M`, `Sundial-Base-128M`, `TimesFM-2.5-200M`, and `Moirai-2.0-R-Small`.
- `transformer_moe_causal`: causal transformer path with sparse mixture-of-experts routing. Use this for `Time-MoE-*` and `Moirai-MoE-*`.
- `tabular_transformer`: transformer-style tabular classifier operating on flattened benchmark features.
- `vision_transformer`: image-first ViT-style backbone reused as a frozen benchmark encoder.
- `vision_convnet`: image-first convolutional backbone reused as a frozen benchmark encoder.
- `slstm`: structured/stateful LSTM backbone.
- `mlp_mixer`: token/channel mixing MLP backbone.
- `hybrid_sequence_model`: mixed sequence backbone that combines multiple modeling primitives and does not fit a narrower class cleanly.
- `causal_cnn`: purely causal convolutional encoder.

Training-paradigm classes:

- `forecasting`: benchmarked encoder/backbone pretrained primarily for forecasting / next-step prediction.
- `representation_ssl`: benchmarked encoder/backbone pretrained with self-supervised or self-distillation representation learning instead of a task-specific supervised head.
- `cross_modal_transfer`: checkpoint pretrained in another modality and transferred into this benchmark as a frozen encoder.
- `tabular_supervised`: supervised tabular classifier reused on flattened benchmark features.

`model.training.paradigm` is a pretraining taxonomy, not a checkpoint-packaging
taxonomy. When adding new models, classify the benchmarked encoder path by the
dominant upstream pretraining recipe and never add a separate fine-tune bucket. If
an official released checkpoint is a downstream fine-tune of a self-supervised
encoder, keep `representation_ssl` and document the downstream fine-tuned checkpoint
under the model's source/checkpoint notes or adapter metadata instead. The same rule
applies to forecasting-pretrained backbones: the training class stays `forecasting`
even if the upstream release also provides a supervised adaptation stage.

When a model family contains both encoder and decoder components, classify it by the
path the benchmark actually reads. For example, `Kairos-*` is stored as
`transformer_full_attention` because the adapter pools encoder states, while
`TimesFM-2.5-200M` is stored as `transformer_causal` because the adapter reads the
decoder-only causal transformer stack.

## Troubleshooting Periodic Metric Mismatches

If two evaluations use the same checkpoint but disagree on dense periodic metrics,
check the dataset manifest before treating the delta as a model change.

First inspect:

- `benchmark_family`
- `benchmark_version`
- `sampling_frequency`
- `channel_size`
- `duration_sec`
- resolved `*_frequency_hz_resolved_low/high`
- `square_duty_cycle_min/max`

The most common failure mode is semantic drift rather than checkpoint drift:

- different `sampling_frequency`
- different exact sequence length
- different `frequency_hz` resolution mode
- different square `duty_cycle` range

For the current `aiono_basic_components/v1` contract, a hidden `sampling_frequency`
mismatch is considered a benchmark bug, not a valid alternate evaluation.

## Dashboard Contract

`results/dashboard.html` reads the checked-in or newly generated JSON files directly in the browser. Keep these constraints in mind when changing result payloads:

- file discovery now tries root-relative `/models/list.txt` first, then relative `models/` directory listing; if neither works, the UI reports the discovery failure and loads no model files;
- layer ids are serialized as JSON object keys;
- summary fields such as `best_auc`, `best_auprc`, macro best `r2`, and macro best `pearson` are read by the UI;
- sidebar controls are grouped into independent HTML disclosure sections; only the `Model selector` section starts expanded by default, and multiple sections may remain open simultaneously;
- the `Filter models` panel narrows the model-selector list by checkpoint-name search and by selected classes from the current shared color mode; changing the color mode clears any active class-filter selection;
- the shared color selector reads canonical JSON metadata rather than inferring groups from filenames: `model.family`, `model.checkpoint_name`, `model.architecture.backbone`, and `model.training.paradigm`;
- the selection-aware bubble chart only plots currently enabled models and allows any supported bubble metric on the `x` axis, `y` axis, or bubble size; inference uses `runtime.encoder_forward_total_s`; parameter axes can now switch between total registered model parameters and cumulative parameters through the furthest plotted best layer from the active layer-aware bubble metrics; parameter-count axes render on a log scale; and older JSONs may still fall back to `results.shared.timings.collect_*.*forward_s` for inference mode;
- adapters that do not expose a registered PyTorch encoder may leave `model.adapter.parameter_count` as `null`; the dashboard must treat that as unavailable metadata rather than inventing a count;
- total parameter metadata lives in `model.adapter.parameter_count` and the explicit alias `model.adapter.parameter_count_total`; cumulative representation-path counts live in `model.adapter.parameter_count_prefix_by_layer`; `TabPFN-v2` and `TabICL-v1` still report the single official backbone count rather than multiplying by one-vs-rest classifier replicas;
- grouped dense metrics depend on the target signal and target metric labels stored in the JSON;
- browser charts may clip values for display, but raw metrics must remain preserved in the JSON.

If you change the output schema in Python, update the dashboard in the same task.

## Adapter Workflow

To add a new model:

1. Add a `ModelSpec` entry in `aionoscope_benchmarks/model_registry.py`.
2. Implement a `FrozenTimeSeriesAdapter` subclass in `aionoscope_benchmarks/adapters/`.
3. Pick the environment name that can actually import the model stack.
4. Assign explicit taxonomy fields in `MODEL_TAXONOMY`, using the benchmark-path rules above for `model.architecture.backbone` and `model.training.paradigm`.
5. Expose an exact benchmark sequence length, honest `available_layers`, and stable `adapter_metadata()`.
6. Make the adapter fail fast on sequence-length mismatch rather than silently cropping, padding, or waveform-resampling benchmark inputs.
7. Run at least one benchmark invocation and verify that a JSON result is produced and the dashboard can read it.

Use `prepare()` and `update_probe_val_split()` only for benchmark-facing preprocessing that the adapter genuinely needs.
Forecast-derived adapters should prefer keeping the full exact waveform as context and
adding a deterministic query row internally over burning one real benchmark timestep as
forecast horizon.

If a model repo already publishes a self-contained inference bundle, prefer importing
that bundle through `huggingface_hub` instead of copying the upstream inference code into
this repo.

## Repo Conventions

- Keep benchmark orchestration in Python.
- Treat `results/models/*.json` as generated artifacts, not hand-edited source files.
- Keep heavyweight third-party code under `external/` isolated from the benchmark package itself.
- Prefer fail-fast behavior with explicit errors when a required environment, external repo, or checkpoint is missing.
- When changing benchmark contracts, result schema, or adapter behavior, review `README.md`, `ARCHITECTURE.md`, and this document together.
