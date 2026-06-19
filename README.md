# Aionoscope Benchmarks

This file is human-facing onboarding. Agents should update it when public-facing workflow or scope changes, but they should gather repository context from `AGENTS.md`, `docs/index.md`, `ARCHITECTURE.md`, and `DOCUMENTATION.md` instead.

Separate benchmark repo for running frozen-feature offline probes on the balanced
Aiono basic-components dataset against foundational models listed in
`benchmark_models_list.md`.

Current scope:

- foundational models plus explicit calibration baselines
- balanced Aiono basic-components offline probes
- one JSON result per benchmark run
- interactive browser dashboard from those JSON results
- standalone manifold artifacts for inspecting representation geometry

Canonical benchmark model names now include explicit version and size whenever an
upstream family publishes multiple official checkpoints, and they keep published
variant tokens when that is the only stable official identifier for a repo-hosted or
model-card artifact. The registry and model docs also point to the official upstream
repo and, when available, the official Hugging Face checkpoint rather than an
unofficial mirror or alias.

Additional docs:

- `docs/index.md` for the repo knowledge map and task-oriented doc index
- `ARCHITECTURE.md` for the stable benchmark design and execution model
- `DOCUMENTATION.md` for operational details, result schema expectations, and workflow notes
- `docs/planning.md` for the GitHub-issue planning workflow

Canonical local validation commands:

```bash
uv run python -m aionoscope_benchmarks.repo_checks
uv run python -m aionoscope_benchmarks.dashboard_smoke
uv run pytest
```

Run paper-critical baseline calibrations for one exact sequence length and
interference regime:

```bash
uv run python -m aionoscope_benchmarks.run_baseline \
  --baseline paper-critical \
  --channel-size 512 \
  --num-enabled 2
```

Baseline artifacts use the same `results/models/<slug>__num_enabled_<k>.json`
schema as model artifacts with `model.type = "baseline"`. The dashboard has a
run-type filter so baselines do not mix into the model leaderboard by default.

Run a small manifold inspection:

```bash
uv run python scripts/run_manifold_calibration_sequential.py \
  --model MantisV2 \
  --model LeNEPA-Aiono \
  --target sine_phase \
  --target spike_time_frac \
  --max-layers 4
```

This writes generated manifold JSON artifacts under `results/manifolds/`,
including the viewer `manifest.json`, per-target `metrics.json`,
`plot_data_json`, and `distance_data_json` files. That generated directory is
ignored by Git. The
checked-in hosted viewer is `results/manifolds.html`; Cloudflare Pages deploys
it from the Git-backed `aionoscope-benchmarks` Pages project. When served from
localhost or `file:`, the viewer reads local JSON from `results/manifolds/`;
when served from Pages, it fetches the large JSON corpus from Cloudflare R2 at
`https://manifolds-data.aionoscope.langotime.ai/manifolds/v20260619T143710Z/`.
See `docs/manifold-r2-pages.md` before changing the hosted manifold setup.

The old checked-in article pages were moved to `https://blog.langotime.ai/`.
Cloudflare Pages keeps the legacy benchmark URLs alive through
`results/_redirects`; do not add `results/about-*.html` article pages back to
this repository.

The LeNEPA-CauKer2M training-checkpoint sweep is generated separately so each
checkpoint keeps its own subdirectory and JSON identity:

```bash
uv run python scripts/run_lenepa_cauker2m_checkpoint_manifolds.py \
  --start-index 1 \
  --end-index 220
```

Those artifacts live under
`results/manifolds/LeNEPA-CauKer2M/ckpt_<step>/<target>/metrics.json`; each
payload stores `model.checkpoint_index`, `model.checkpoint_step`, and
`model.checkpoint_path`. The viewer shows a compact checkpoint picker only for
models that have checkpoint-sweep records.

The viewer is an Apache ECharts review page; it reads stored JSON artifacts,
shows axis labels and metric tooltips, and is not loaded by the main benchmark
dashboard. It loads `manifest.json` first, then fetches each selected
model/target `metrics.json` on demand for layerwise scalar metrics. Centroid
manifolds render in 2D or 3D PCA (`echarts-gl`), and a
comparison bar lets you pin up to four selections: the "Metrics across layers"
panel overlays them as coloured per-(model, target) curves, while Centroid path
/ Distance scatter / Distance heatmap show one side-by-side panel per pinned
selection (same-target centroids are Procrustes-aligned to a shared frame).
Scatter and heatmap distance matrices are loaded from `distance_data_json` only
when their collapsed block is opened. In a git worktree without the model
`.venv-*` directories, add `--env-root /path/to/checkout/with/model/envs`.

## Benchmark Identity

The current checked-in benchmark contract is the versioned family `aiono_basic_components/v2`.

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

- `16384`: `TimesFM-2.5-200M`
- `8192`: `Chronos-2`
- `5000`: `LeNEPA-Aiono`, `LeNEPA-CauKer2M`, `LeNEPA-CauKer2M-20k`, `TiViT-H-14-B79K`, `TiConvNext-XXLarge-AugReg`, `T-Loss-CricketX`
- `4096`: `Toto-Open-Base-1.0`, `Time-MoE-50M`, `Time-MoE-200M`
- `3072`: `TempoPFN-38M`
- `2880`: `Timer-Base-84M`, `Sundial-Base-128M`
- `2048`: `TiRex`, `Kairos-10M`, `Kairos-23M`, `Kairos-50M`, `Reverso-Small-550K`
- `512`: `EIDOS`, `Mantis-8M`, `MantisPlus`, `MantisV2`, `Mantis-UTICA-8M`, `MOMENT-1-Large`, `TTM-r2`, `Moirai-1.0-R-Small`, `Moirai-1.0-R-Base`, `Moirai-1.0-R-Large`, `Moirai-1.1-R-Small`, `Moirai-1.1-R-Base`, `Moirai-1.1-R-Large`, `Moirai-2.0-R-Small`, `Moirai-MoE-1.0-R-Small`, `Moirai-MoE-1.0-R-Base`, `UniShape-ZeroShot`, `UniShape-FineTune`, `Toto-2.0-4M`, `Toto-2.0-22M`, `Toto-2.0-313M`, `Toto-2.0-1B`, `Toto-2.0-2.5B`
- `176`: `NuTime-Bias9`
- `128`: `TabPFN-v2`, `TabICL-v1`

That means signal duration is now model-dependent. At `500 Hz`, durations range from
`0.256` seconds for the tabular fallbacks up to `32.766` seconds for `TimesFM-2.5-200M`.

### What gets mixed into each signal

The current `v2` benchmark evaluates three explicit interference regimes per model:
`num_enabled=1`, `2`, and `3`. Each JSON artifact corresponds to one active
`num_enabled` value, and the dataset manifest also records the full configured
`num_enabled_values=[1, 2, 3]` contract.

Enabled components are chosen from this library of primitive building blocks:

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
"one structured pattern + baseline offset", while others are mixtures of multiple
non-constant patterns. The `1/2/3`-component sweep is meant to expose how each model
behaves as interference increases.

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

By default, `run_model` and `run_many` fan out across all configured
`num_enabled_values`, so one model produces three JSON files:
`results/models/<slug>__num_enabled_1.json`,
`results/models/<slug>__num_enabled_2.json`, and
`results/models/<slug>__num_enabled_3.json`.
Use `--num-enabled` to run only a subset, for example:

```bash
uv run python -m aionoscope_benchmarks.run_model --model TiRex --num-enabled 2
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
ECharts, and lets you filter benchmark runs, switch the best-layer selector between
`AUROC`, `AUPRC`, `R2`, and `Pearson`, and choose any supported bubble metric
for the chart `x` axis, `y` axis, or bubble size, including macro `AUROC`,
macro `AUPRC`, macro `R2`, macro `Pearson`, encoder forward time, and stored
model parameter count. The same sidebar also exposes one shared color palette
mode for model chips and every chart: by exact model checkpoint, model family,
architecture type, or coarse training paradigm. When
either bubble-chart axis is set to model parameter count, the dashboard uses a
log scale for that axis automatically. The left sidebar groups controls into
independent collapsible panels; only `Model selector` is expanded by default,
and you can keep any number of sections open at once. The `Enabled components`
panel filters the visible `num_enabled` runs before they reach the selector or
plots. The `Model selector` panel now combines checkpoint-name search with
independent class-selection toggles over the remaining benchmark runs. Those
toggles directly change which runs are selected for display, and they can group
by model family, architecture type, or training paradigm independently of the
shared color palette. The plots no longer append `num_enabled` to chart labels:
layer curves encode it as `1 = solid`, `2 = dashed`, `3 = dotted`, and the
bubble map uses plain / striped / dotted fills for the same run variants.
When the JSON contains repeated validation runs, the dashboard plots the
per-layer / per-category median and shows sample standard deviation (`ddof=1`)
in tooltips, with shaded `± std` bands on the layer curves.
When published with `results/` as the site root, the dashboard first tries
`/models/list.txt`, then falls back to `models/` directory listing. If neither
works, it now reports a discovery error instead of using a hard-coded manifest.
While loading many result artifacts, the dashboard fetches JSONs with bounded
concurrency, retries transient per-file failures, and enforces a per-file
timeout. The `Visible runs` summary shows live progress as
`Loading... loaded/total` until loading completes.
`results/models/list.txt` is only for deployment environments that need an explicit
manifest and should not exist in the dev checkout.

For the full foundational sweep used in the current results, use
`scripts/run_foundational_sequential.py`. It dispatches each model into the
environment pinned for that model family, which is necessary because the
foundational stack spans incompatible dependency sets.
`Time-MoE-50M`, `Time-MoE-200M`, `EIDOS`, `Timer-Base-84M`, and `Sundial-Base-128M`
use the dedicated `.venv-timemoe` interpreter because those official remote-code
checkpoints and local EIDOS runtime require the published `transformers==4.40.1`
stack (plus `einops` for EIDOS).
`TempoPFN-38M` uses `.venv-tempopfn` because the published TempoPFN stack adds
`triton==3.2.0`, `flash-linear-attention`, and the self-contained Hugging Face
repo snapshot code path on top of the base benchmark dependencies.
The explicit `Moirai-*` family uses `.venv-moirai`, and `Mantis-8M`,
`MantisPlus`, `MantisV2`, plus `Mantis-UTICA-8M` use `.venv-mantis2`.

## Foundational Benchmark Snapshot (2026-03-13)

The first full foundational offline-probe benchmark used the LeNEPA/Aiono balanced
basic-components offline-probe contract, evaluated all available layers for deep
models, and was executed sequentially.

Artifacts:

*   Active `v2` output directory: `results/models/<slug>__num_enabled_<k>.json`
*   Historical `v1` reference: git history only
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
Those historical JSONs now live only in git history; the default `results/models/`
discovery path is reserved for one coherent `v2` corpus with explicit `num_enabled`
run suffixes.

The foundational registry now also includes `LeNEPA-Aiono`, `LeNEPA-CauKer2M`,
`LeNEPA-CauKer2M-20k`, `Time-MoE-50M`, `Time-MoE-200M`, `TempoPFN-38M`,
`EIDOS`, `Timer-Base-84M`, `Sundial-Base-128M`, `TimesFM-2.5-200M`, the explicit
`Moirai-*` / `Moirai-MoE-*` variants, `Kairos-*`, `Reverso-Small-550K`, `UniShape-*`, `Mantis-8M`,
`MantisPlus`, and `Mantis-UTICA-8M`.
Current reruns for those entries will live as separate per-run JSON artifacts under
`results/models/<slug>__num_enabled_<k>.json`,
but the table below remains the original `2026-03-13` single-seed snapshot and was not
regenerated for the expanded registry. For univariate zero-shot forecasting, the
official Timer workflow uses `thuml/timer-base-84m` as the published `Timer-XL`
checkpoint, so the benchmark keeps the exact published checkpoint name
`Timer-Base-84M`. The old standalone `results/models/Moirai.json` artifact has been
retired; historical Moirai outputs now live only in git history, and current Moirai
reruns will use the explicit versioned `Moirai-*` and
`Moirai-MoE-*` names plus the `num_enabled` suffix.
Embedding-aware adapters now reserve layer `0` for the embedding stream. For both
LeNEPA adapters, benchmark layers run from `0..8`: layer `0` is the tokenizer
/ embedding output, layers `1..7` are intermediate transformer-block outputs, and layer
`8` is the encoder output after the published final layer norm before mean pooling.

| Model | Layers tested | Best AUROC layer | Macro AUROC | Best AUPRC layer | Macro AUPRC | Best R2 layer | Macro R2 | Best Pearson layer | Macro Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Chronos-2` | 12 | 9 | 0.9845 | 9 | 0.9420 | 11 | 0.3064 | 5 | 0.5239 |
| `MOMENT-1-Large` | 24 | 23 | 0.9016 | 23 | 0.7473 | 22 | 0.2932 | 10 | 0.4511 |
| `MantisV2` | 6 | 2 | 0.9699 | 2 | 0.9005 | 1 | 0.5170 | 2 | 0.7084 |
| `Moirai` | 6 | 5 | 0.8454 | 5 | 0.6142 | 0 | 0.0711 | 5 | 0.2717 |
| `NuTime-Bias9` | 6 | 5 | 0.8908 | 5 | 0.6843 | 1 | 0.3871 | 2 | 0.6026 |
| `T-Loss-CricketX` | 12 | 10 | 0.8943 | 10 | 0.7158 | 10 | 0.4912 | 10 | 0.6864 |
| `TTM-r2` | 15 | 13 | 0.8224 | 12 | 0.5349 | 3 | 0.1176 | 14 | 0.2371 |
| `TabICL-v1` | 1 | 0 | 0.8199 | 0 | 0.5513 | 0 | 0.0801 | 0 | 0.2305 |
| `TabPFN-v2` | 1 | 0 | 0.8693 | 0 | 0.6354 | 0 | 0.0772 | 0 | 0.2332 |
| `TiConvNext-XXLarge-AugReg` | 40 | 38 | 0.9908 | 25 | 0.9598 | 34 | 0.4580 | 33 | 0.5901 |
| `TiRex` | 12 | 5 | 0.9675 | 5 | 0.8890 | 11 | 0.3614 | 3 | 0.5778 |
| `TiViT-H-14-B79K` | 32 | 16 | 0.9898 | 26 | 0.9570 | 31 | 0.0880 | 31 | 0.5094 |
| `Toto-Open-Base-1.0` | 12 | 8 | 0.9774 | 8 | 0.9178 | 7 | 0.5985 | 7 | 0.7383 |

Benchmark gotchas:

*   The interactive charts use absolute values, not step deltas.
*   The benchmark code now generates Aiono data online at runtime. It does not depend on a checked-in `results/datasets/*` snapshot; each run rebuilds the same deterministic finite split in memory from the YAML config and seeds.
*   The current benchmark contract is versioned as `aiono_basic_components/v2`. Compare absolute metrics only against runs with the same family/version and inspect the manifest for resolved periodic bounds before interpreting dense periodic deltas.
*   The validation protocol now evaluates the same fixed train split against validation seed values `0..9`, mapped to generator seeds `100..109` so train and validation seeds do not overlap.
*   Probe training randomness is held fixed with `probe_seed=0` across those validation runs, so the reported spread reflects the validation-seed sweep rather than probe reinitialization noise.
*   Every metric value in the new JSON schema stores the full validation-seed array plus `median/std`, and the dashboard visualizes those aggregated values. The historical checked-in snapshot predates this schema and also predates the versioned periodic contract, so it behaves like `n=1` legacy data rather than a current `v2` reference.
*   The dashboard lets you choose the layer selector per run: `best_auc.layer`, `best_auprc.layer`, best macro `R2`, or best macro `Pearson`. Those best-layer choices are taken from the metric medians across validation runs. Do not confuse those selector views with the oracle-over-layer summaries stored in the JSON payloads.
*   The dashboard now includes a dedicated `Enabled components` filter. Unchecked `num_enabled` values disappear from the run selector, legends, and plots, and any hidden runs are removed from the active selection state instead of being restored later.
*   The selection-aware bubble chart only shows currently enabled runs. Any supported bubble metric can drive the `x` axis, `y` axis, or bubble size: macro `AUROC`, macro `AUPRC`, macro `R2`, macro `Pearson`, explicit encoder forward time, or stored model parameter count. Parameter-count axes switch to log scale automatically so large and small models remain comparable on the same chart. When a bubble axis or size uses model parameters, the UI now exposes a parameter-scope selector: either total registered model parameters or cumulative parameters through the furthest plotted best layer across the currently selected layer-aware bubble metrics. Chart labels now stay on the canonical model name while `num_enabled` is encoded visually: `1 = solid/plain`, `2 = dashed/striped`, `3 = dotted/dotted-fill`.
*   Current benchmark JSONs also store explicit dashboard taxonomy under `model.family`, `model.checkpoint_name`, `model.architecture.backbone`, and `model.training.paradigm`. For time-series transformers, the architecture class is benchmark-path based: `transformer_causal` means the pooled token stream is causal / decoder-style, `transformer_full_attention` means the pooled token stream can attend across the full context, and `transformer_moe_causal` is the same causal definition with sparse MoE routing. `model.training.paradigm` is intentionally pretraining-based: classify the benchmarked encoder path by its dominant upstream pretraining recipe, not by whether the published checkpoint was later fine-tuned for a downstream task. The shared color-mode selector uses those stored fields instead of inferring family or architecture groups from filenames in browser-only code.
*   New benchmark JSONs store encoder forward timing explicitly in `runtime.encoder_forward_total_s` and related train/validation breakdown fields. The dashboard falls back to `results.shared.timings.collect_*.*forward_s` only for older JSONs that predate the explicit runtime fields.
*   New benchmark JSONs store total parameter counts in `model.adapter.parameter_count` plus the explicit alias `model.adapter.parameter_count_total`, and also store cumulative representation-path counts in `model.adapter.parameter_count_prefix_by_layer`. The dashboard uses those prefix counts for the new through-best-layer parameter scope. For `TabPFN-v2` and `TabICL-v1`, the total count still comes from the single official backbone model exposed by the fitted public classifier API, and is not multiplied by one-vs-rest labels or estimator replicas.
*   The dense regression panels are labeled `Regression By Component Type` and `Regression By Parameter Type`, and they use absolute `R2` and Pearson, not `MSE`. `MSE` varied too much across target types to be useful on a shared dashboard scale.
*   Negative `R2` values are clipped to `0` in the dashboard charts only so the useful range stays readable. Tooltips still expose the raw `R2` median and its standard deviation, and the JSON files keep the raw values.
*   `TabPFN-v2` and `TabICL-v1` are not natural frozen layerwise encoders. In this benchmark they use fallback adapters that expose a synthetic single `layer 0`, run on exact `128`-sample waveforms as tabular features, train one-vs-rest classifiers, and cap both probe train and probe val subsets at `2048` samples.
*   `TabPFN-v2` names the actual checked-in classifier artifact and points at the official `Prior-Labs/TabPFN-v2-clf` checkpoint. The official `tabpfn_2_5` family exists separately, but switching the canonical benchmark entry would require a fresh rerun.
*   Model-native exact lengths can materially increase RAM usage because the benchmark still materializes finite train/validation tensors in memory. `Chronos-2` is the most extreme case because its exact context length is `8192`.
*   `TTM-r2` peaks at different layers for different selectors, notably AUROC vs AUPRC (`13` vs `12`).
*   The TiViT image-backbone wrappers are the slowest and heaviest foundational runs on the full balanced online-generated split, especially `TiConvNext-XXLarge-AugReg`.
