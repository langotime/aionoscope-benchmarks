# Plan: Expand Basic-Components Benchmark to 1/2/3 Enabled Components

## Goal

Extend the balanced `basic_components` benchmark so every model is evaluated under three explicit interference regimes with `num_enabled = 1`, `2`, and `3`. The update must keep the benchmark contract explicit, preserve reproducibility metadata, make the dashboard filterable by `num_enabled`, expand tests, and update documentation in the same task.

## Current State

- `configs/dataset_aiono_basic_components_balanced.yaml` hard-codes `aiono.basic_components.num_enabled: 2`.
- `aionoscope_benchmarks/runtime_dataset.py` reads exactly one integer `num_enabled`, stores it in the dataset manifest, and builds exactly one pipeline for that active count.
- `aionoscope_benchmarks/run_model.py` writes one JSON file per model to `results/models/<model-slug>.json`, so repeated runs for the same model would overwrite one another.
- `results/dashboard.html` uses `result.model.name` as the effective identity in selection and hover state, so multiple JSON files for the same model but different benchmark conditions would collide.
- Top-level docs and checked-in result artifacts describe the current benchmark as `aiono_basic_components/v1` with exactly two enabled components.

## Locked Decisions

These decisions are resolved and should be treated as the implementation contract.

1. Benchmark versioning
   - This change will produce a new benchmark version: `aiono_basic_components/v2`.
   - `v1` remains the historical "exactly two enabled components" contract.
   - `v2` is the new "three explicit interference regimes with enabled-component counts 1/2/3" contract.

2. Benchmark shape
   - We will not mix `1/2/3` component counts inside one sampled dataset.
   - We will run three explicit benchmark runs per model, one per active `num_enabled`.
   - This preserves the current "one JSON artifact per benchmark run" model and avoids a large nested result schema.

3. Dataset config shape
   - Replace the scalar config field with an explicit list, for example `num_enabled_values: [1, 2, 3]`.
   - Each concrete run still uses one scalar active `num_enabled`.
   - Each produced manifest will store both:
     - `num_enabled`: the active count for that JSON artifact;
     - `num_enabled_values`: the full configured count set for the `v2` contract.

4. Runner behavior
   - `run_model.py` will fan out across all configured `num_enabled_values` by default.
   - `run_model.py` will also accept an explicit `--num-enabled` override so local testing can run only a subset of the configured counts.
   - `run_many.py` and `scripts/run_foundational_sequential.py` will inherit the same behavior and will treat each `(model, num_enabled)` pair as a distinct run target.

5. Artifact naming
   - Output filenames will be unique per model and active component count.
   - The canonical filename pattern will be `results/models/<model-slug>__num_enabled_<k>.json`.
   - We will not keep writing unsuffixed `results/models/<model-slug>.json` for `v2`.

6. Dashboard identity
   - The dashboard will stop using `model.name` as the only stable identity.
   - The internal run key will be a composite identifier built from benchmark metadata already present in the JSON, specifically `model.slug + benchmark_family + benchmark_version + num_enabled`.
   - Canonical model naming remains unchanged. The interference regime is benchmark-run metadata, not model taxonomy.

7. Dashboard filtering behavior
   - Add a dedicated sidebar filter for enabled-component counts.
   - The filter shows the available `num_enabled` values as explicit toggles; for the checked-in `v2` corpus that means `1`, `2`, and `3`.
   - All values are selected by default.
   - This filter is applied before the model selector and before plot construction.
   - If a `num_enabled` value is unchecked:
     - runs with that value disappear from the model selector;
     - runs with that value are removed from the effective selected-run set;
     - runs with that value do not appear in any plot, legend, summary count, or hover state.
   - Re-enabling a value makes those runs available in the model selector again, but they re-enter as normal unselected runs rather than as hidden retained selections.

8. Dashboard presentation
   - The model selector will display benchmark runs, not only canonical model names.
   - Each visible entry will show the canonical model label plus its active component count, for example by appending a short `num_enabled` suffix.
   - Tooltips and chart labels will also expose `benchmark_version` and `num_enabled` so benchmark conditions are always visible.
   - Color grouping stays model-centric and uses the existing taxonomy fields; the new `num_enabled` control is a filter, not a new color mode.

9. Historical artifact handling
   - We will not mix `v1` and `v2` JSON artifacts inside the default dashboard corpus under `results/models/`.
   - Existing checked-in `v1` JSONs must be removed from the working tree; git history is the historical archive.
   - The default `results/models/` directory must contain one coherent benchmark version so the dashboard does not silently compare incompatible contracts.

## Implementation Steps

1. Update the benchmark contract and config.
   - Change `configs/dataset_aiono_basic_components_balanced.yaml` from `benchmark_version: v1` to `benchmark_version: v2`.
   - Replace `aiono.basic_components.num_enabled: 2` with `aiono.basic_components.num_enabled_values: [1, 2, 3]`.
   - Keep the periodic contract unchanged unless a separate benchmark change requires it.

2. Refactor runtime dataset building around an explicit active enabled-component count.
   - Extend `aionoscope_benchmarks/runtime_dataset.py` to validate `num_enabled_values`.
   - Require the runtime builder to resolve one active `num_enabled` for each concrete run.
   - Fail fast on invalid values, duplicates, empty lists, or an override outside the configured count set.
   - Extend `DatasetManifest` so each result records both the active count and the configured count set.

3. Update runner orchestration.
   - Refactor `aionoscope_benchmarks/run_model.py` so one call can execute multiple enabled-component runs and emit one JSON artifact per run.
   - Add explicit CLI plumbing for `--num-enabled`.
   - Update `aionoscope_benchmarks/run_many.py` so progress and failure handling are per `(model, num_enabled)` run.
   - Update `scripts/run_foundational_sequential.py` so skip / force logic keys off `results/models/<slug>__num_enabled_<k>.json` instead of the old unsuffixed path.

4. Update result payload expectations.
   - Keep the existing result shape centered on one benchmark run per JSON file.
   - Do not introduce nested per-`num_enabled` result payloads inside a single model JSON.
   - Rely on dataset-manifest metadata for run identity and dashboard filtering.

5. Update the dashboard.
   - Add the new enabled-components sidebar filter and wire it into the global visible-run set.
   - Move all internal selection, hover, chart, and legend logic to the composite run key.
   - Make the model selector render only runs that pass the enabled-components filter.
   - Make all charts, summaries, and hover synchronization operate only on runs that pass the enabled-components filter and are then explicitly selected.
   - Expose the active `num_enabled` in chip labels, tooltips, and benchmark-condition text.

6. Migrate the checked-in artifact set.
   - Remove `v1` JSONs from the default `results/models/` discovery path.
   - Keep `v1` only in git history rather than in a checked-in archive directory.
   - Ensure the checked-in `results/models/` corpus is coherent for `v2` and the new filename convention.

7. Expand tests.
   - Add runtime-dataset tests for `num_enabled_values`, active-count overrides, manifest metadata, and invalid-config failures.
   - Add runner tests for fan-out behavior, per-run output naming, and `--num-enabled` override validation.
   - Update dashboard contract tests so they assert:
     - the enabled-components filter exists;
     - the dashboard uses a composite run key instead of raw `model.name`;
     - unchecked `num_enabled` values remove runs from both the model selector and the plots.
   - Run the relevant pytest subset and a dashboard smoke check with representative `v2` JSON artifacts.

8. Review and update documentation as part of the same implementation.
   - `README.md`: explain the `v2` contract, the three interference regimes, new artifact naming, dashboard filtering, and that `v1` now lives only in git history.
   - `ARCHITECTURE.md`: document the explicit multi-run benchmark model, composite dashboard identity, and the rule that `results/models/` must not mix incompatible benchmark versions.
   - `DOCUMENTATION.md`: document config shape, runtime override behavior, artifact naming, manifest fields, the git-history-only handling for `v1`, and the enabled-components dashboard filter workflow.

## Test Plan

- `tests/test_sequence_length_contract.py`
  - cover `num_enabled_values`, active-count selection, manifest fields, and invalid-config cases.
- New runner-focused tests
  - verify that one model invocation produces three distinct output paths;
  - verify that `--num-enabled 2` only emits the `num_enabled=2` artifact;
  - verify that invalid `--num-enabled` values fail with a clear error.
- `tests/test_dashboard_discovery_contract.py`
  - assert the enabled-components filter markup and logic are present;
  - assert a composite run key is used instead of `model.name`;
  - assert filtered-out `num_enabled` runs are excluded from the model selector and plot builders.
- Smoke checks
  - run the relevant `pytest` subset;
  - run at least one local benchmark invocation that emits representative `v2` artifacts;
  - verify that `results/dashboard.html` initializes correctly from the generated `v2` JSON files.

## Remaining Execution Risks

The design questions are resolved. The remaining risks are operational rather than architectural:

- Regenerating a coherent checked-in `v2` artifact corpus may be time-consuming because the foundational sweep spans multiple pinned environments.
- Moving `v1` artifacts out of the default dashboard path must be coordinated with docs so historical references remain discoverable.
- The dashboard refactor touches selection, hover, legend, and plot state in one file, so regression tests are important even though the intended behavior is now fully specified.

## Documentation Review Step

Before marking the implementation complete, explicitly review `ARCHITECTURE.md`, `README.md`, and `DOCUMENTATION.md`, update all three where needed, and record why any unchanged section truly requires no edit.

## Documentation Impact

- `ARCHITECTURE.md`: must change to describe the new `v2` contract, the explicit `1/2/3` interference-run model, the composite dashboard run identity, and the rule that `results/models/` must not mix incompatible benchmark versions.
- `README.md`: must change to replace the current "exactly 2 components" explanation with the new `1/2/3` interference story and to explain new artifact names plus the enabled-components dashboard filter.
- `DOCUMENTATION.md`: must change to document the `v2` dataset-config shape, active-count override behavior, manifest/result metadata, artifact naming, the git-history-only handling for `v1`, and dashboard filtering/comparison workflow for `num_enabled=1/2/3`.
