# Architecture

## Purpose

`aionoscope-benchmarks` is a separate benchmark runner for evaluating frozen time-series representations on a fixed Aionoscope/Aiono synthetic benchmark contract. It is not the upstream signal-generation library. The repo owns benchmark orchestration, adapter integration, offline probe evaluation, result aggregation, and browser-side visualization.

## Stable Design Decisions

### Separate benchmark repo

The benchmark code lives outside the upstream `aionoscope` library and depends on it as a sibling editable dependency. This keeps benchmark-specific code, heavyweight model integrations, and result artifacts isolated from the core generator library.

### Runtime dataset materialization

The benchmark does not rely on checked-in `train.pt` or `val.pt` snapshots. `src/aionoscope_benchmarks/dataset_snapshot.py` rebuilds the finite train split and validation splits in memory from the dataset YAML on every run. The resulting manifest is part of the output JSON and is the reproducibility contract for the evaluated split.

### Model-native exact sequence lengths

The benchmark no longer uses one shared sequence length for every model. `run_model.py`
instantiates the adapter first, resolves that model's exact benchmark sequence length,
and only then asks Aiono to materialize the finite train and validation splits. The
dataset manifest stores both the config default length and the exact resolved length
used for the run.

### Fixed validation-seed protocol

The benchmark contract is a fixed train split plus a fixed ordered set of validation seed values. Validation seed values are user-facing identifiers; generator seeds are derived by adding an offset so train and validation generation do not overlap. Aggregated metrics are computed over the ordered validation-seed set and stored explicitly in the result payload.

### Adapter boundary for foundational models

Every benchmarked model is wrapped by a `FrozenTimeSeriesAdapter` implementation in `src/aionoscope_benchmarks/adapters/`. The adapter contract is:

- expose an exact benchmark sequence length;
- expose `available_layers`;
- optionally preprocess the benchmark split in `prepare()` and `update_probe_val_split()`;
- return one representation tensor per requested layer from `forward_layer_dict()`;
- report stable adapter metadata for the result JSON.

The benchmark pipeline treats all models through this adapter boundary rather than through model-specific probe code paths.
When a model is published as a self-contained Hugging Face inference bundle, the adapter
may download and import that published bundle directly via `huggingface_hub` instead of
depending on a separate pip package or vendored external repo.
When an adapter can expose an explicit embedding stream, layer `0` is reserved for that
embedding representation and subsequent encoder blocks start at layer `1`.
Adapters must fail fast on input-length mismatch. Benchmark adapters must not silently
crop, pad, or waveform-resample the generated Aiono sequence to fit the model.

### Offline probes operate on collected features

`run_model.py` first collects layerwise representations for the full finite train and validation splits, then runs linear probes over those frozen features. Feature collection and probe training are intentionally separated so per-layer probe evaluation does not recompute model embeddings.

### Result JSON is the canonical artifact

One JSON file in `results/models/` is the canonical output for one model run. The JSON stores:

- model identity and adapter metadata;
- dataset manifest and probe configuration;
- per-layer categorical and dense probe metrics;
- aggregated validation-run statistics (`values`, `median`, `std`, `n`);
- shared runtime and validation-seed metadata;
- summary selections such as best layers and oracle-per-target views.

The browser dashboard is a consumer of this JSON schema, not an independent source of truth.

### Multi-environment execution is intentional

The foundational model stack spans incompatible dependency sets. The repo therefore supports multiple pinned virtual environments such as `.venv`, `.venv-tivit`, and `.venv-tabular`. `scripts/run_foundational_sequential.py` dispatches each model into the interpreter mapped from its registry entry. This is part of the architecture, not a temporary workaround.

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

- `configs/dataset_toyts_basic_components_balanced.yaml`: dataset contract for the balanced ToyTS basic-components benchmark.
- `configs/probe.yaml`: linear probe training and evaluation hyperparameters.
- `configs/models_foundational.yaml`: foundational model list used by sweep scripts.

### Runtime split builder

`src/aionoscope_benchmarks/dataset_snapshot.py` builds the benchmark splits and manifest from the dataset YAML and the sibling `aiono` package.

### Model registry and adapters

`src/aionoscope_benchmarks/model_registry.py` maps canonical model names to source metadata, environment names, and adapter classes. Adapters implement the stable representation-extraction interface.

### Offline probe engine

`src/aionoscope_benchmarks/offline_probe.py` stages collected features and trains the linear multi-head probes used for both multi-label classification and dense regression.

### Result assembly

`src/aionoscope_benchmarks/results.py` aggregates metrics across validation runs, computes summary selections, and writes the canonical JSON result payload.

### Entry points

- `src/aionoscope_benchmarks/run_model.py`: run one model and write one JSON file.
- `src/aionoscope_benchmarks/run_many.py`: run a list of models in the current environment.
- `scripts/run_foundational_sequential.py`: sweep the full foundational set across multiple pinned environments.

### Visualization

`results/dashboard.html` is a static browser dashboard that reads `results/models/*.json` and visualizes the stored metrics. It must stay compatible with the JSON schema produced by `results.py`.

## Architectural Invariants

- Benchmark changes must preserve the clear split between dataset generation, adapter integration, offline probing, result aggregation, and visualization.
- The train split and validation splits must remain reproducible from config plus seed information stored in the manifest.
- Validation aggregation must not silently reorder or drop validation seeds.
- Adapters may customize preprocessing, but they must still expose a stable layerwise representation interface, emit truthful adapter metadata, and reject benchmark inputs with the wrong sequence length.
- Result schema changes must be reflected in `README.md`, `DOCUMENTATION.md`, and `results/dashboard.html` in the same task.
- The dashboard must remain a pure reader of result artifacts; benchmark computations belong in Python, not in browser-only logic.
