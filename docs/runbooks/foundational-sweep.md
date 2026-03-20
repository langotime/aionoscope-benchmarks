# Foundational Sweep Runbook

Use this runbook for the full foundational benchmark sweep.

## Canonical Entry Point

```bash
uv run python scripts/run_foundational_sequential.py
```

That script is allowed to dispatch into the pinned per-family interpreters declared in the model registry.

## Before Starting

- verify the relevant `.venv-*` interpreters exist
- confirm the sibling `aionoscope` checkout is available for the editable `aiono` dependency
- check [../references/runtime-environments.md](../references/runtime-environments.md) for family-to-environment expectations
- make sure `results/models/` is not mixing incompatible benchmark versions

## What To Inspect After A Run

- `results/models/*.json`: canonical metrics, manifest, and runtime timings
- stderr/stdout from the invoked runner: step-by-step execution log

If the dashboard website deployment needs `results/models/list.txt`, generate it during deployment rather than in this dev checkout.

## Failure Triage

- missing dependency: repair the specific pinned environment and rerun
- sequence-length mismatch: fix the adapter; do not patch around it in the runner
- dashboard regression: run `uv run python -m aionoscope_benchmarks.dashboard_smoke`
- docs or planning drift: run `uv run python -m aionoscope_benchmarks.repo_checks`
