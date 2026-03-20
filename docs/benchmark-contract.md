# Benchmark Contract

The benchmark contract lives primarily in:

- [../configs/dataset_aiono_basic_components_balanced.yaml](../configs/dataset_aiono_basic_components_balanced.yaml)
- [../configs/probe.yaml](../configs/probe.yaml)

Treat changes to those files as benchmark-definition changes, not routine tuning.

## Contract Fields That Must Stay Explicit

- benchmark family/version
- exact sequence-length policy
- training seed
- validation seed values and validation-generator offset
- `num_enabled_values`
- train/validation batch counts
- dense target definitions
- periodic-waveform resolution semantics inherited from `aiono`

## Current Execution Model

1. Load the adapter first.
2. Resolve that adapter's exact benchmark sequence length.
3. Materialize the finite train/validation splits in memory.
4. Preserve reproducibility metadata in the dataset manifest.
5. Run frozen-feature probes.
6. Emit one JSON artifact per `(model, num_enabled)` run.

## What Counts As A Contract Change

Update code, tests, and docs together when changing:

- `benchmark_family` or `benchmark_version`
- validation-seed semantics or ordering
- dense-target definitions
- result-schema fields required by the dashboard
- output naming under `results/models/`

If the change affects run-to-run comparability, treat it as a versioned benchmark change and update the top-level docs in the same task.
