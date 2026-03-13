# Documentation

## What This Repo Does

This repo benchmarks frozen foundational time-series models on the balanced ToyTS basic-components contract built from the sibling `aiono` library. Each run:

- rebuilds a deterministic finite benchmark split in memory;
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

## Common Commands

Run one model in the current environment:

```bash
uv run python -m aionoscope_benchmarks.run_model --model TiRex
```

Run a few models in the current environment:

```bash
uv run python -m aionoscope_benchmarks.run_many --model TiRex --model Chronos2
```

Run all foundational models with the per-model pinned interpreters:

```bash
uv run python scripts/run_foundational_sequential.py
```

Serve the static dashboard:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/results/dashboard.html`.

## Config Files

### Dataset config

`configs/dataset_toyts_basic_components_balanced.yaml` defines the benchmark contract:

- sampling frequency and sequence length;
- component library and `num_enabled`;
- training seed;
- ordered validation seed values and validation seed offset;
- train and validation batch counts;
- dense target definitions.

Changing this file changes the benchmark contract and must be treated as a benchmark-definition change, not a casual tuning knob.

### Probe config

`configs/probe.yaml` controls linear probe training:

- number of steps;
- batch size;
- optimizer hyperparameters;
- gradient clipping;
- checkpoint interval used for selecting best probe checkpoints.

### Foundational model list

`configs/models_foundational.yaml` is the human-facing sweep list. The code-level source of truth for metadata and adapters is `src/aionoscope_benchmarks/model_registry.py`.

The two LeNEPA entries (`LeNEPA-Aiono` and `LeNEPA-CauKer2M`) run in the base `core`
environment. On first use, their adapters download the published `inference.py`,
`lenepa_encoder_config.json`, and `lenepa_encoder.safetensors` files from the
checkpoint repository on Hugging Face and then reuse the local cache on subsequent runs.
They expose zero-indexed benchmark layers `0..8`; layer `0` is the tokenizer output and
layer `8` is the post-final-layer-norm encoder output, matching the published export
contract before mean pooling.
More generally, adapters that expose a distinct embedding stream use layer `0` for that
embedding and number transformer-style encoder blocks from `1`.

## Validation Seed Semantics

The benchmark distinguishes:

- validation seed values: user-facing identifiers stored in result payloads;
- validation generator seeds: actual dataset generator seeds derived by adding `validation_seed_offset`.

The train seed must not overlap with any validation generator seed. The manifest stored in the JSON records both the ordered validation seed values and the derived generator seeds.

## Result Artifacts

Each model run writes `results/models/<slug>.json`.

High-level structure:

- `model`: model identity, source, checkpoint, layers evaluated, and adapter metadata;
- `dataset`: the benchmark manifest used to build the train and validation splits;
- `probe_config`: probe hyperparameters plus the fixed `probe_seed`;
- `runtime`: wall-clock and device metadata;
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

## Dashboard Contract

`results/dashboard.html` reads the checked-in or newly generated JSON files directly in the browser. Keep these constraints in mind when changing result payloads:

- layer ids are serialized as JSON object keys;
- summary fields such as `best_auc`, `best_auprc`, macro best `r2`, and macro best `pearson` are read by the UI;
- grouped dense metrics depend on the target signal and target metric labels stored in the JSON;
- browser charts may clip values for display, but raw metrics must remain preserved in the JSON.

If you change the output schema in Python, update the dashboard in the same task.

## Adapter Workflow

To add a new model:

1. Add a `ModelSpec` entry in `src/aionoscope_benchmarks/model_registry.py`.
2. Implement a `FrozenTimeSeriesAdapter` subclass in `src/aionoscope_benchmarks/adapters/`.
3. Pick the environment name that can actually import the model stack.
4. Expose honest `available_layers` and stable `adapter_metadata()`.
5. Run at least one benchmark invocation and verify that a JSON result is produced and the dashboard can read it.

Use `prepare()` and `update_probe_val_split()` only for benchmark-facing preprocessing that the adapter genuinely needs.

If a model repo already publishes a self-contained inference bundle, prefer importing
that bundle through `huggingface_hub` instead of copying the upstream inference code into
this repo.

## Repo Conventions

- Keep benchmark orchestration in Python.
- Treat `results/models/*.json` as generated artifacts, not hand-edited source files.
- Keep heavyweight third-party code under `external/` isolated from the benchmark package itself.
- Prefer fail-fast behavior with explicit errors when a required environment, external repo, or checkpoint is missing.
- When changing benchmark contracts, result schema, or adapter behavior, review `README.md`, `ARCHITECTURE.md`, and this document together.
