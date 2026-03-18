# Architecture

## Purpose

`aionoscope-benchmarks` is a separate benchmark runner for evaluating frozen time-series representations on a fixed Aionoscope/Aiono synthetic benchmark contract. It is not the upstream signal-generation library. The repo owns benchmark orchestration, adapter integration, offline probe evaluation, result aggregation, and browser-side visualization.

## Stable Design Decisions

### Separate benchmark repo

The benchmark code lives outside the upstream `aionoscope` library and depends on it as a sibling editable dependency. This keeps benchmark-specific code, heavyweight model integrations, and result artifacts isolated from the core generator library.

### Runtime dataset materialization

`aionoscope_benchmarks/runtime_dataset.py` rebuilds the finite train split and validation splits in memory from the dataset YAML on every run. The resulting manifest is part of the output JSON and is the reproducibility contract for the evaluated split.

### Versioned benchmark semantics

The benchmark contract is versioned. The current family is `aiono_basic_components/v1`.

Changing any of the following requires a benchmark-version change rather than a silent
config tweak:

- baseline `sampling_frequency`
- periodic recoverability rules
- resolved-bound semantics for `frequency_hz: auto`
- dense-target definitions

Results are only comparable within the same benchmark family/version.

### Shared periodic resolver in `aiono`

Periodic benchmark semantics are resolved in the upstream `aiono` library, not
reimplemented separately in this repo. `runtime_dataset.py` calls the shared
`resolve_aiono_basic_components_periodic_contract(...)` helper and records the resolved
bounds in the dataset manifest.

This keeps benchmark generation, examples, and downstream consumers on one source of
truth for:

- baseline `sampling_frequency = 500 Hz`
- `frequency_hz: auto`
- waveform-specific recoverability rules
- square duty-cycle-aware upper bounds

### Model-native exact sequence lengths

The benchmark no longer uses one shared sequence length for every model. `run_model.py`
instantiates the adapter first, resolves that model's exact benchmark sequence length,
and only then asks Aiono to materialize the finite train and validation splits. The
dataset manifest stores both the config default length and the exact resolved length
used for the run.

### Fixed validation-seed protocol

The benchmark contract is a fixed train split plus a fixed ordered set of validation seed values. Validation seed values are user-facing identifiers; generator seeds are derived by adding an offset so train and validation generation do not overlap. Aggregated metrics are computed over the ordered validation-seed set and stored explicitly in the result payload.

### Adapter boundary for foundational models

Every benchmarked model is wrapped by a `FrozenTimeSeriesAdapter` implementation in `aionoscope_benchmarks/adapters/`. The adapter contract is:

- expose an exact benchmark sequence length;
- expose `available_layers`;
- optionally preprocess the benchmark split in `prepare()` and `update_probe_val_split()`;
- return one representation tensor per requested layer from `forward_layer_dict()`;
- report stable adapter metadata for the result JSON.

The benchmark pipeline treats all models through this adapter boundary rather than through model-specific probe code paths.
When a model is published as a self-contained Hugging Face inference bundle, the adapter
may download and import that published bundle directly via `huggingface_hub` instead of
depending on a separate pip package or vendored external repo.
Forecasting wrappers still fit this boundary when the adapter keeps the full exact
benchmark waveform as context, appends a deterministic next-step query row internally,
and cuts a frozen representation at a pre-head query-state boundary instead of using
forecast scalars as probe features.
When an adapter can expose an explicit embedding stream, layer `0` is reserved for that
embedding representation and subsequent encoder blocks start at layer `1`.
Adapters must fail fast on input-length mismatch. Benchmark adapters must not silently
crop, pad, or waveform-resample the generated Aiono sequence to fit the model.

### Offline probes operate on collected features

`run_model.py` first collects layerwise representations for the full finite train and validation splits, then runs linear probes over those frozen features. Feature collection and probe training are intentionally separated so per-layer probe evaluation does not recompute model embeddings.

### Result JSON is the canonical artifact

One JSON file in `results/models/` is the canonical output for one model run. The JSON stores:

- model identity and adapter metadata;
- model taxonomy for dashboard grouping: family/checkpoint split plus architecture and training labels;
- dataset manifest and probe configuration;
- per-layer categorical and dense probe metrics;
- aggregated validation-run statistics (`values`, `median`, `std`, `n`);
- shared runtime and validation-seed metadata;
- summary selections such as best layers and oracle-per-target views.
- explicit adapter metadata for both total model parameter count and cumulative representation-path parameter counts by layer.

The dataset manifest is intentionally rich enough to diagnose semantic drift. It now
includes benchmark family/version markers plus the resolved periodic-frequency contract
for the exact sequence length used in that run.

The browser dashboard is a consumer of this JSON schema, not an independent source of truth.

### Dashboard taxonomy is explicit and benchmark-path based

The dashboard taxonomy stored in each result JSON is intentionally explicit:

- `model.family`: the published model family name shared across checkpoints;
- `model.checkpoint_name`: the checkpoint or variant token within that family;
- `model.architecture.backbone`: the architecture class used for dashboard grouping;
- `model.training.paradigm`: the coarse training-paradigm class used for dashboard grouping.

`model.architecture.backbone` is defined by the representation path used inside the
benchmark adapter, not by loose paper terminology. This matters for models that
ship both encoder and decoder components: the classification must follow the token
stream that the benchmark actually pools.

Current architecture classes:

- `transformer_full_attention`: time-series transformer path with full-context self-attention over the benchmark context. Padding or group masks are allowed, but there is no causal time mask on the pooled token stream. This includes encoder-style paths such as `Chronos-2`, `Kairos-*`, `MOMENT-1-Large`, `Moirai-1.x-*`, `NuTime-Bias9`, `Toto-Open-Base-1.0`, `MantisV2`, `Mantis-UTICA-8M`, and `UniShape-*`.
- `transformer_causal`: time-series transformer path with causal masking or decoder-only token states. This includes `LeNEPA-*`, `Timer-Base-84M`, `Sundial-Base-128M`, `TimesFM-2.5-200M`, and `Moirai-2.0-R-Small`.
- `transformer_moe_causal`: causal transformer path with sparse mixture-of-experts routing. This includes `Time-MoE-*` and `Moirai-MoE-*`.
- `tabular_transformer`: transformer-style tabular classifier operating on flattened benchmark features.
- `vision_transformer`: image-first ViT-style backbone reused as a frozen benchmark encoder.
- `vision_convnet`: image-first convolutional backbone reused as a frozen benchmark encoder.
- `slstm`: structured/stateful LSTM backbone.
- `mlp_mixer`: token/channel mixing MLP backbone.
- `hybrid_sequence_model`: mixed sequence backbone that combines multiple modeling primitives and does not fit a narrower class cleanly.
- `causal_cnn`: purely causal convolutional encoder.

Current training-paradigm classes:

- `forecasting`: checkpoint trained primarily for forecasting / next-step prediction.
- `representation_ssl`: checkpoint trained with self-supervised representation learning instead of a task-specific supervised head.
- `task_finetune`: checkpoint released as an official task-fine-tuned model.
- `cross_modal_transfer`: checkpoint pretrained in another modality and transferred into this benchmark as a frozen encoder.
- `tabular_supervised`: supervised tabular classifier reused on flattened benchmark features.

### Multi-environment execution is intentional

The foundational model stack spans incompatible dependency sets. The repo therefore supports multiple pinned virtual environments such as `.venv`, `.venv-tabular`, `.venv-timemoe`, `.venv-moirai`, `.venv-mantis2`, and `.venv-tivit`. `scripts/run_foundational_sequential.py` dispatches each model into the interpreter mapped from its registry entry. This is part of the architecture, not a temporary workaround.

## Execution Model

1. Load probe config and dataset config YAML.
2. Instantiate the requested adapter from the model registry.
3. Resolve the adapter's exact benchmark sequence length.
4. Build the in-memory benchmark train split and validation splits plus a dataset manifest.
5. Let the adapter prepare any model-specific benchmark views or cached state.
6. Collect frozen features for all requested layers on the train split.
7. Collect frozen features for all requested layers on each validation split.
8. Train and evaluate linear probes on the collected features.
9. Aggregate metrics across validation seeds into the JSON result schema.
10. Persist one model JSON artifact for later comparison and dashboard visualization.

## Main Components

### Configs

- `configs/dataset_aiono_basic_components_balanced.yaml`: dataset contract for the balanced Aiono basic-components benchmark.
- `configs/probe.yaml`: linear probe training and evaluation hyperparameters.
- `configs/models_foundational.yaml`: foundational model list used by sweep scripts.

### Runtime split builder

`aionoscope_benchmarks/runtime_dataset.py` builds the benchmark splits and manifest from the dataset YAML and the sibling `aiono` package.

### Model registry and adapters

`aionoscope_benchmarks/model_registry.py` maps canonical model names to source metadata, environment names, and adapter classes. Adapters implement the stable representation-extraction interface.
Canonical model names are versioned and size-qualified whenever the upstream family
publishes multiple official checkpoints. The registry therefore prefers entries such as
`TimesFM-2.5-200M`, `Moirai-1.1-R-Small`, `Toto-Open-Base-1.0`, or `Mantis-UTICA-8M`
over ambiguous family-only labels. When an upstream repo-hosted singleton checkpoint is
identified by a published variant token instead of a size marker, that token is part of
the canonical name as well, for example `NuTime-Bias9` or `T-Loss-CricketX`.
Registry entries are expected to point at the official upstream repository and, when one
exists, the official Hugging Face checkpoint. If the only official published checkpoint
is shipped directly inside the upstream repo, that exception must be documented
explicitly in the top-level docs instead of being treated as an implicit fallback.

### Offline probe engine

`aionoscope_benchmarks/offline_probe.py` stages collected features and trains the linear multi-head probes used for both multi-label classification and dense regression.

### Result assembly

`aionoscope_benchmarks/results.py` aggregates metrics across validation runs, computes summary selections, and writes the canonical JSON result payload.

### Entry points

- `aionoscope_benchmarks/run_model.py`: run one model and write one JSON file.
- `aionoscope_benchmarks/run_many.py`: run a list of models in the current environment.
- `scripts/run_foundational_sequential.py`: sweep the full foundational set across multiple pinned environments.

### Visualization

`results/dashboard.html` is a static browser dashboard that reads `results/models/*.json` and visualizes the stored metrics. It must stay compatible with the JSON schema produced by `results.py`, including the explicit runtime encoder-forward totals, the total adapter parameter counts, the cumulative through-layer parameter metadata used by the bubble chart controls, and the canonical model taxonomy fields used by the shared color-palette selector across model chips, bubble charts, radar panels, and layer curves.

## Architectural Invariants

- Benchmark changes must preserve the clear split between dataset generation, adapter integration, offline probing, result aggregation, and visualization.
- The train split and validation splits must remain reproducible from config plus seed information stored in the manifest.
- The benchmark must never silently drift away from the versioned `aiono` contract for periodic semantics.
- Validation aggregation must not silently reorder or drop validation seeds.
- Adapters may customize preprocessing, but they must still expose a stable layerwise representation interface, emit truthful adapter metadata, and reject benchmark inputs with the wrong sequence length.
- Result schema changes must be reflected in `README.md`, `DOCUMENTATION.md`, and `results/dashboard.html` in the same task.
- The dashboard must remain a pure reader of result artifacts; benchmark computations belong in Python, not in browser-only logic.
