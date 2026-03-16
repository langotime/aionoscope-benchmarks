# Aionoscope Benchmarks

Separate benchmark repo for running frozen-feature offline probes on the balanced
Aiono basic-components dataset against foundational models listed in
`benchmark_models_list.md`.

Current scope:

- foundational models only
- balanced Aiono basic-components offline probes
- one JSON result per model
- interactive browser dashboard from those JSON results

Additional docs:

- `ARCHITECTURE.md` for the stable benchmark design and execution model
- `DOCUMENTATION.md` for operational details, result schema expectations, and workflow notes

## Benchmark Identity

The current checked-in benchmark contract is the versioned family `aiono_basic_components/v1`.

Stable semantics for this family:

- baseline `sampling_frequency = 500 Hz`
- benchmark family/version stored in every runtime manifest and result JSON
- periodic `frequency_hz = auto` by default, resolved from sequence length plus waveform-specific recoverability rules in the shared `aiono` resolver

Absolute metric comparisons are only valid within the same benchmark family/version.

## Dataset In Plain English

You do not need to know Aiono internals to read these benchmark results. In this repo,
Aiono is simply the synthetic signal generator used to create the evaluation set.

For this benchmark, each example is:

- one single-channel time series
- generated at `500 Hz` sampling rate
- materialized at the exact benchmark sequence length for the current model, not at one fixed global length
- produced by rendering a small number of primitive signal components into one final observed waveform

In Aiono terms, the final observed waveform is the `mix` view. In plain English, that
just means "take the enabled components, render them, and add them together into one
1D signal."

The family no longer hard-codes one shared periodic frequency range for every model.
Instead, `aiono.resolve_aiono_basic_components_periodic_contract(...)` resolves
`frequency_hz` from the exact sequence length used for that model:

- `sine` uses a Nyquist-based upper bound;
- `sawtooth` uses a points-per-period recoverability upper bound;
- `square` uses a duty-cycle-aware shorter-plateau upper bound.

Those resolved bounds are written into the dataset manifest so downstream analysis can
tell which periodic task was actually evaluated.

### Exact sequence lengths per model

The benchmark now resolves sequence length from the adapter before dataset generation.
Adapters are expected to fail fast on any length mismatch rather than silently cropping,
padding, or waveform-resampling the benchmark data.

Current exact lengths:

- `8192`: `Chronos2`
- `5000`: `LeNEPA-Aiono`, `LeNEPA-CauKer2M`, `LeNEPA-CauKer2M-20k`, `TiViT-H`, `TiConvNext`, `T-Loss`
- `4096`: `Toto`
- `2048`: `TiRex`
- `512`: `MantisV2`, `MOMENT`, `TTM`, `Moirai`
- `176`: `NuTime`
- `128`: `TabPFN`, `TabICL`

That means signal duration is now model-dependent. At `500 Hz`, durations range from
`0.256` seconds for the tabular fallbacks up to `16.384` seconds for `Chronos2`.

### What gets mixed into each signal

Every sample enables exactly `2` components (`num_enabled=2`) chosen from this library
of primitive building blocks:

- Baseline: `constant`, which acts like a flat offset or baseline level
- Noise: `gaussian_noise`, `uniform_noise`, `random_walk_noise`
- Trends: `linear_trend`, `quadratic_trend`, `log_trend`, `sigmoid_trend`
- Periodic patterns: `sine`, `sawtooth`, `square`
- Local events: `spike`, `level_change`, `gaussian` (a localized Gaussian bump)

That means a signal can look like, for example:

- a sine wave plus Gaussian noise
- a square wave plus a linear trend
- a spike plus a baseline shift
- a level change plus random-walk noise

Because `constant` is one of the selectable components, some samples are effectively
"one structured pattern + baseline offset", while others are mixtures of two
non-constant patterns.

### Why the dataset is called "balanced"

"Balanced" here means the component sampler is unweighted across the component list.
There is no manual class-weight skew that makes a few signal types dominate the dataset.
This gives a cleaner probe benchmark for comparing representations across many different
signal families.

Each benchmark run regenerates the same finite evaluation split on the fly from
Aiono using fixed seeds, fixed batch counts, and the current model's exact benchmark
sequence length. The split is materialized only in RAM for the current run.

Those fixed settings are:

- sequence length policy: exact per-model, resolved before dataset generation
- training seed: `0`
- validation seed values: `0..9`
- validation generator-seed offset: `+100`, so the actual validation generator seeds are `100..109`
- training set: `256` batches x `256` samples = `65,536` examples
- validation set per seed: `256` batches x `256` samples = `65,536` examples

The `+100` offset is deliberate: it keeps the validation generator seeds disjoint from the
training seed range while still preserving the human-facing validation seed pool `0..9`.

### What the models have to predict

The benchmark is not just "classify the whole waveform."

It evaluates two kinds of readout from frozen features:

- Multi-label classification: predict which of the `14` component types are present in the signal.
- Dense regression: predict the underlying generative parameters when they exist.

Examples of dense targets include:

- event timing and amplitude for `spike`, `level_change`, and `gaussian`
- noise scale for the noise processes
- slope / intercept / curvature for trends
- amplitude / frequency / phase / offset for periodic waves
- duty cycle for `square`

The dense benchmark contains `34` regression targets in total. Reported classification
metrics are also grouped into the human-readable families `noise`, `trend`, `periodic`,
and `events`.

In the radar plots, dense regression quality is shown in two aggregations:

- by component type: for example `spike`, `sine`, `gaussian_noise`, `linear_trend`
- by parameter type: for example `amplitude`, `frequency_hz`, `time_frac`, `slope`

## Typical Workflow

Minimal core workflow:

```bash
uv sync
uv run python -m aionoscope_benchmarks.run_many --model all
```

`run_model` and `run_many` build the balanced Aiono split at runtime via
`aiono.datasets.SynthBatchIterableDataset`. `run_model` loads the adapter first,
resolves that model's exact benchmark sequence length, then keeps the finite train/val tensors
only in process memory for that benchmark run.

Interactive browser view:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/results/dashboard.html`. The page loads
`results/models/*.json`, computes one selection-aware bubble chart plus the same
6 radar panels and 4 layer curves in browser-side JavaScript with Apache
ECharts, and lets you filter models plus switch the best-layer selector between
`AUROC`, `AUPRC`, `R2`, and `Pearson`.
When the JSON contains repeated validation runs, the dashboard plots the
per-layer / per-category median and shows sample standard deviation (`ddof=1`)
in tooltips, with shaded `± std` bands on the layer curves.

For the full foundational sweep used in the current results, use
`scripts/run_foundational_sequential.py`. It dispatches each model into the
environment pinned for that model family, which is necessary because the
foundational stack spans incompatible dependency sets.

## Foundational Benchmark Snapshot (2026-03-13)

The first full foundational offline-probe benchmark used the LeNEPA/Aiono balanced
basic-components offline-probe contract, evaluated all available layers for deep
models, and was executed sequentially.

Artifacts:

*   Per-model JSON results: `results/models/*.json`
*   Interactive dashboard: `results/dashboard.html`
*   Sequential runner: `scripts/run_foundational_sequential.py`

Results summary:

The table below is the original single-validation-seed snapshot from `2026-03-13`.
The benchmark code now targets the `10`-seed validation pool described above, but the
checked-in JSON snapshot has not been rerun yet, so these rows should be treated as
legacy `n=1` reference numbers.
They also predate the current model-native exact-sequence-length contract; the historical
snapshot used one shared `5000`-sample dataset for every model.
They also predate the current embedding-aware layer numbering, where some adapters now
reserve layer `0` for the embedding stream, so the checked-in best-layer ids are legacy
indices rather than current ones.

The foundational registry now also includes `LeNEPA-Aiono`, `LeNEPA-CauKer2M`,
and `LeNEPA-CauKer2M-20k`. Those entries were added after this checked-in snapshot,
so they do not yet have checked-in JSON artifacts or rows in the table below.
Embedding-aware adapters now reserve layer `0` for the embedding stream. For both
LeNEPA adapters, benchmark layers run from `0..8`: layer `0` is the tokenizer
/ embedding output, layers `1..7` are intermediate transformer-block outputs, and layer
`8` is the encoder output after the published final layer norm before mean pooling.

| Model | Layers tested | Best AUROC layer | Macro AUROC | Best AUPRC layer | Macro AUPRC | Best R2 layer | Macro R2 | Best Pearson layer | Macro Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Chronos2` | 12 | 9 | 0.9845 | 9 | 0.9420 | 11 | 0.3064 | 5 | 0.5239 |
| `MOMENT` | 24 | 23 | 0.9016 | 23 | 0.7473 | 22 | 0.2932 | 10 | 0.4511 |
| `MantisV2` | 6 | 2 | 0.9699 | 2 | 0.9005 | 1 | 0.5170 | 2 | 0.7084 |
| `Moirai` | 6 | 5 | 0.8454 | 5 | 0.6142 | 0 | 0.0711 | 5 | 0.2717 |
| `NuTime` | 6 | 5 | 0.8908 | 5 | 0.6843 | 1 | 0.3871 | 2 | 0.6026 |
| `T-Loss` | 12 | 10 | 0.8943 | 10 | 0.7158 | 10 | 0.4912 | 10 | 0.6864 |
| `TTM` | 15 | 13 | 0.8224 | 12 | 0.5349 | 3 | 0.1176 | 14 | 0.2371 |
| `TabICL` | 1 | 0 | 0.8199 | 0 | 0.5513 | 0 | 0.0801 | 0 | 0.2305 |
| `TabPFN` | 1 | 0 | 0.8693 | 0 | 0.6354 | 0 | 0.0772 | 0 | 0.2332 |
| `TiConvNext` | 40 | 38 | 0.9908 | 25 | 0.9598 | 34 | 0.4580 | 33 | 0.5901 |
| `TiRex` | 12 | 5 | 0.9675 | 5 | 0.8890 | 11 | 0.3614 | 3 | 0.5778 |
| `TiViT-H` | 32 | 16 | 0.9898 | 26 | 0.9570 | 31 | 0.0880 | 31 | 0.5094 |
| `Toto` | 12 | 8 | 0.9774 | 8 | 0.9178 | 7 | 0.5985 | 7 | 0.7383 |

Benchmark gotchas:

*   The interactive charts use absolute values, not step deltas.
*   The benchmark code now generates Aiono data online at runtime. It does not depend on a checked-in `results/datasets/*` snapshot; each run rebuilds the same deterministic finite split in memory from the YAML config and seeds.
*   The current benchmark contract is versioned as `aiono_basic_components/v1`. Compare absolute metrics only against runs with the same family/version and inspect the manifest for resolved periodic bounds before interpreting dense periodic deltas.
*   The validation protocol now evaluates the same fixed train split against validation seed values `0..9`, mapped to generator seeds `100..109` so train and validation seeds do not overlap.
*   Probe training randomness is held fixed with `probe_seed=0` across those validation runs, so the reported spread reflects the validation-seed sweep rather than probe reinitialization noise.
*   Every metric value in the new JSON schema stores the full validation-seed array plus `median/std`, and the dashboard visualizes those aggregated values. The historical checked-in snapshot predates this schema and also predates the versioned periodic contract, so it behaves like `n=1` legacy data rather than a current `v1` reference.
*   The dashboard lets you choose the layer selector per model: `best_auc.layer`, `best_auprc.layer`, best macro `R2`, or best macro `Pearson`. Those best-layer choices are taken from the metric medians across validation runs. Do not confuse those selector views with the oracle-over-layer summaries stored in the JSON payloads.
*   The selection-aware bubble chart uses macro AUPRC on `x`, macro `R2` on `y`, and encoder forward time for bubble size. It only shows currently enabled models. That size comes from `results.shared.timings.collect_train.forward_s` plus the sum of `results.shared.timings.collect_val.forward_s.values`, not from `runtime.total_wall_s`.
*   The dense regression panels are labeled `Regression By Component Type` and `Regression By Parameter Type`, and they use absolute `R2` and Pearson, not `MSE`. `MSE` varied too much across target types to be useful on a shared dashboard scale.
*   Negative `R2` values are clipped to `0` in the dashboard charts only so the useful range stays readable. Tooltips still expose the raw `R2` median and its standard deviation, and the JSON files keep the raw values.
*   `TabPFN` and `TabICL` are not natural frozen layerwise encoders. In this benchmark they use fallback adapters that expose a synthetic single `layer 0`, run on exact `128`-sample waveforms as tabular features, train one-vs-rest classifiers, and cap both probe train and probe val subsets at `2048` samples.
*   `TabPFN` could not use the gated `v2.5` checkpoint in an unattended run, so the benchmark falls back to the public `v2` model.
*   Model-native exact lengths can materially increase RAM usage because the benchmark still materializes finite train/validation tensors in memory. `Chronos2` is the most extreme case because its exact context length is `8192`.
*   `TTM` peaks at different layers for different selectors, notably AUROC vs AUPRC (`13` vs `12`).
*   The TiViT image-backbone wrappers are the slowest and heaviest foundational runs on the full balanced online-generated split, especially `TiConvNext`.
