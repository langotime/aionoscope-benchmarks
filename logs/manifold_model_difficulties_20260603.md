# Manifold Benchmark Model Difficulties

Date: 2026-06-03 UTC

Scope: `aionoscope-benchmarks-manifold`, combined manifold run
`results/manifold_calibration/manifold_grid1024_opt_combined`.

Benchmark shape used for the sweep:

- Grid size: `1024`
- Geodesic neighbor: `8`
- Linear targets: `sine_phase`, `sine_frequency_hz`, `sine_amplitude`,
  `spike_time_frac`, `gaussian_time_frac`, `linear_trend_slope`
- Signed-log target: `linear_trend_slope__signed_log`
- Main summary file: `logs/manifold_missing_sweep_20260602T144054Z.json`

## TabPFN-v2

Both linear and signed-log manifold runs failed before writing target metrics.

Reason: the current TabPFN adapter is a supervised tabular fallback adapter. It
trains one-vs-rest classifiers from the provided `train_split["y_cls"]`. A
controlled manifold slice intentionally activates only the target component, so
`y_cls` contains no negative samples for that component and no positive samples
for most other components. The adapter raises:

`TabPFN one-vs-rest fit needs both positive and negative samples`.

This is not a CUDA/runtime problem. The model needs a manifold-specific path:
fit TabPFN on a normal balanced runtime split, then evaluate probabilities on
each controlled manifold slice.

Mitigation added on 2026-06-03: the manifold runner now prepares supervised
tabular adapters on a normal balanced runtime split (`num_enabled=1`), and the
TabPFN adapter exposes a dynamic manifold representation path for controlled
evaluation batches.

Status after targeted rerun: all six linear targets and the signed-log
`linear_trend_slope` target completed and were merged into the combined
artifact.

## TabICL-v1

Both linear and signed-log manifold runs failed for the same structural reason
as TabPFN-v2.

The adapter is also a supervised tabular one-vs-rest fallback adapter and cannot
fit its classifiers from a single-component controlled manifold slice. It raises:

`TabICL one-vs-rest fit needs both positive and negative samples`.

Like TabPFN-v2, this should be handled by fitting on a normal balanced runtime
split and evaluating the fitted classifiers on the controlled slices.

Mitigation added on 2026-06-03: the same balanced-prepare plus dynamic
controlled-evaluation path was added for TabICL.

Status after targeted rerun: all six linear targets and the signed-log
`linear_trend_slope` target completed and were merged into the combined
artifact.

## MOMENT-1-Large

The initial linear run wrote these targets successfully:

- `sine_phase`
- `sine_frequency_hz`
- `sine_amplitude`

It failed while processing `spike_time_frac` with:

`numpy.linalg.LinAlgError: SVD did not converge`

The signed-log `linear_trend_slope` run completed successfully.

Mitigation added on 2026-06-03: `maybe_pca()` now sanitizes non-finite feature
values and falls back to raw sanitized features if sklearn PCA raises
`LinAlgError`. The fallback records `pca_error` in the PCA payload instead of
stopping the whole target.

Status after targeted rerun: all missing MOMENT linear targets were completed
and merged into the combined artifact:

- `spike_time_frac`
- `gaussian_time_frac`
- `linear_trend_slope`

The combined viewer was rebuilt after merging these targets.

## Toto-2.0 models

All Toto-2 versions ran successfully:

- `Toto-2.0-4M`
- `Toto-2.0-22M`
- `Toto-2.0-313M`
- `Toto-2.0-1B`
- `Toto-2.0-2.5B`

Difficulties were runtime and numerical warnings, not fatal failures.

Runtime scaled steeply with depth. Approximate observed full-run timings:

- `Toto-2.0-4M`: linear about 5.2 min, signed-log about 1.1 min
- `Toto-2.0-22M`: linear about 7.7 min, signed-log about 1.5 min
- `Toto-2.0-313M`: linear about 33.9 min, signed-log about 5.7 min
- `Toto-2.0-1B`: linear about 57.1 min, signed-log about 10.4 min
- `Toto-2.0-2.5B`: linear about 102.9 min, signed-log about 16.6 min

Several Toto-2 targets, especially `sine_amplitude`, emitted sklearn PCA
warnings like:

`RuntimeWarning: invalid value encountered in divide`

These warnings did not stop the runs; metrics were written and merged.

## TiViT / TiConvNext / NuTime

The remaining image-time-series style models completed, but used the `tivit`
environment and emitted dependency warnings:

- `TiViT-H-14-B79K`
- `TiConvNext-XXLarge-AugReg`
- `NuTime-Bias9`

Observed warnings included deprecated `timm.models.layers` imports and sklearn
PCA variance warnings for some targets. These were non-fatal.

## T-Loss-CricketX

Completed successfully. It emitted a PyTorch warning about deprecated
`torch.nn.utils.weight_norm`, but the warning was non-fatal and all target
metrics were merged.

## Environment Issues Resolved During Setup

The manifold repo did not initially have a ready Toto-2 environment. A local
`.venv-toto` was prepared, then the sweep used the sibling
`../aionoscope-benchmarks` environment root because it already contained the
per-model `.venv-*` environments.

Toto-2 required Python 3.12 and extra packages from the DataDog Toto repository.
The local setup also needed a compatible `torchvision` version and
`setuptools<81` to keep `pkg_resources` imports working in the Toto stack.

The manifold repo also needed an ignored `external -> ../aionoscope-benchmarks/external`
symlink so adapters could resolve sibling external checkouts/checkpoints without
copying them into this repo.

## Final Combined Artifact Check

After the MOMENT and tabular reruns, the combined artifact contains:

- Registry models: `45`
- Artifact model directories: `45`
- Metrics files: `315`
- Missing model/target records: `0`

Both viewers were rebuilt:

- `results/manifold_calibration/manifold_grid1024_opt_combined/index.html`
- `results/manifold_calibration/index.html`
