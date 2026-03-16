# Documentation

## What This Repo Does

This repo benchmarks frozen foundational time-series models on the balanced ToyTS basic-components contract built from the sibling `aiono` library. Each run:

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

Be aware that exact model-native sequence lengths can materially increase RAM use because
the benchmark still materializes finite train and validation tensors in memory before
feature collection. `Chronos2` is the heaviest case because it now runs at exact length
`8192`.

Serve the static dashboard:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/results/dashboard.html`.

## Config Files

### Dataset config

`configs/dataset_toyts_basic_components_balanced.yaml` defines the benchmark contract:

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

- `benchmark_family: toyts_basic_components`
- `benchmark_version: v1`

`v1` semantics are resolved in the shared `aiono` library, not reimplemented in each
consumer. The benchmark repo calls
`resolve_toyts_basic_components_periodic_contract(...)` and writes the resolved result
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

The current exact benchmark lengths are:

- `Chronos2`: `8192`
- `LeNEPA-Aiono`: `5000`
- `LeNEPA-CauKer2M`: `5000`
- `MantisV2`: `512`
- `Moirai`: `512`
- `MOMENT`: `512`
- `NuTime`: `176`
- `TabICLForecaster`: `4096`
- `T-Loss`: `5000`
- `TTM`: `512`
- `TabICL`: `128`
- `TabPFN`: `128`
- `TabPFN-TS`: `4096`
- `TiConvNext`: `5000`
- `TiRex`: `2048`
- `TiViT-H`: `5000`
- `Toto`: `4096`

The two new forecasting-derived tabular adapters follow a different extraction path
from the older `TabPFN` / `TabICL` fallbacks:

- `TabPFN-TS` uses the official `tabpfn-time-series` feature engineering stack on the
  full exact `4096`-sample waveform, generates a synthetic `prediction_length=1`
  next-step query row, fits `TabPFNRegressor`, and uses the averaged official
  `get_embeddings(..., data_source="test")` output for that query row as `layer 0`.
- `TabICLForecaster` uses the official `TabICLForecaster` preprocessing stack on the
  full exact `4096`-sample waveform, generates the same synthetic one-step query row,
  and then exposes forecast-query row states from the raw `TabICL` model:
  `layer 0` is the row-interaction output and `layers 1..12` are the ICL block outputs.

Because both adapters currently require a separate forecast-table fit per benchmark
sample, they cap both probe train and probe val subsets at `128` samples. That cap is
reported in adapter metadata and is an explicit benchmark policy, not a hidden fallback.

## Validation Seed Semantics

The benchmark distinguishes:

- validation seed values: user-facing identifiers stored in result payloads;
- validation generator seeds: actual dataset generator seeds derived by adding `validation_seed_offset`.

The train seed must not overlap with any validation generator seed. The manifest stored in the JSON records both the ordered validation seed values and the derived generator seeds.

## Result Artifacts

Each model run writes `results/models/<slug>.json`.

High-level structure:

- `model`: model identity, source, checkpoint, layers evaluated, and adapter metadata;
- `dataset`: the benchmark manifest used to build the train and validation splits, including the config default length and the exact resolved length used for the run;
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

For the current `toyts_basic_components/v1` contract, a hidden `sampling_frequency`
mismatch is considered a benchmark bug, not a valid alternate evaluation.

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
4. Expose an exact benchmark sequence length, honest `available_layers`, and stable `adapter_metadata()`.
5. Make the adapter fail fast on sequence-length mismatch rather than silently cropping, padding, or waveform-resampling benchmark inputs.
6. Run at least one benchmark invocation and verify that a JSON result is produced and the dashboard can read it.

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
