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

Time-MoE uses the official remote-code checkpoints and the upstream repo's
published `transformers` pin. Install that in the dedicated Time-MoE env:

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

## Cloudflare Pages Deployment

For Cloudflare Pages, publish `results/` as the build output directory. In that
layout the dashboard is served as `/dashboard.html`, the model JSON files live
under `/models/`, and the dashboard first tries `/models/list.txt` before it
falls back to directory listing.

Cloudflare Pages may auto-detect Python from the repository root because this
repo contains `pyproject.toml`. If you leave dependency installation enabled,
Pages will try `pip install .` before your build command and fail on this repo's
benchmark-specific Python constraints and local editable dependency layout.

Build command:

```bash
find results/models -maxdepth 1 -type f -name '*.json' -printf '%f\n' | LC_ALL=C sort > results/models/list.txt
```

Build output directory:

```text
results
```

Environment variable:

```text
SKIP_DEPENDENCY_INSTALL=true
```

The generated `results/models/list.txt` should contain one JSON filename per
line, for example `Chronos2.json`. `results/dashboard.html` normalizes those
entries back to `models/<file>.json` at runtime. When you serve the repo root
locally with `python -m http.server 8000`, `/models/list.txt` usually does not
exist, so the dashboard automatically falls back to the relative
`results/models/` directory listing instead.

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

The two LeNEPA entries (`LeNEPA-Aiono` and `LeNEPA-CauKer2M`) run in the base `core`
environment. On first use, their adapters download the published `inference.py`,
`lenepa_encoder_config.json`, and `lenepa_encoder.safetensors` files from the
checkpoint repository on Hugging Face and then reuse the local cache on subsequent runs.
They expose zero-indexed benchmark layers `0..8`; layer `0` is the tokenizer output and
layer `8` is the post-final-layer-norm encoder output, matching the published export
contract before mean pooling.
More generally, adapters that expose a distinct embedding stream use layer `0` for that
embedding and number transformer-style encoder blocks from `1`.
`Time-MoE-Base` and `Time-MoE-Large` also reserve layer `0` for their input
embedding stream. They load the official `Maple728/TimeMoE-50M` and
`Maple728/TimeMoE-200M` checkpoints through `transformers` remote code, pin the
exact benchmark length to the checkpoint `max_position_embeddings=4096`, apply the
official per-series z-score normalization to the exact benchmark waveform, and
mean-pool the causal token stream across time. Their final benchmark layer is the
post-final-norm decoder output.

The current exact benchmark lengths are:

- `Chronos2`: `8192`
- `LeNEPA-Aiono`: `5000`
- `LeNEPA-CauKer2M`: `5000`
- `MantisV2`: `512`
- `Moirai`: `512`
- `MOMENT`: `512`
- `NuTime`: `176`
- `T-Loss`: `5000`
- `Time-MoE-Base`: `4096`
- `Time-MoE-Large`: `4096`
- `TTM`: `512`
- `TabICL`: `128`
- `TabPFN`: `128`
- `TiConvNext`: `5000`
- `TiRex`: `2048`
- `TiViT-H`: `5000`
- `Toto`: `4096`

## Validation Seed Semantics

The benchmark distinguishes:

- validation seed values: user-facing identifiers stored in result payloads;
- validation generator seeds: actual dataset generator seeds derived by adding `validation_seed_offset`.

The train seed must not overlap with any validation generator seed. The manifest stored in the JSON records both the ordered validation seed values and the derived generator seeds.

## Result Artifacts

Each model run writes `results/models/<slug>.json`.

High-level structure:

- `model`: model identity, source, checkpoint, layers evaluated, and adapter metadata, including stored parameter counts when the adapter exposes registered PyTorch parameters;
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
- the selection-aware bubble chart only plots currently enabled models and allows any supported bubble metric on the `x` axis, `y` axis, or bubble size; inference uses `runtime.encoder_forward_total_s`, parameter count uses `model.adapter.parameter_count`, parameter-count axes render on a log scale, and older JSONs may still fall back to `results.shared.timings.collect_*.*forward_s` for inference mode;
- adapters that do not expose a registered PyTorch encoder may leave `model.adapter.parameter_count` as `null`; the dashboard must treat that as unavailable metadata rather than inventing a count;
- grouped dense metrics depend on the target signal and target metric labels stored in the JSON;
- browser charts may clip values for display, but raw metrics must remain preserved in the JSON.

If you change the output schema in Python, update the dashboard in the same task.

## Adapter Workflow

To add a new model:

1. Add a `ModelSpec` entry in `aionoscope_benchmarks/model_registry.py`.
2. Implement a `FrozenTimeSeriesAdapter` subclass in `aionoscope_benchmarks/adapters/`.
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
