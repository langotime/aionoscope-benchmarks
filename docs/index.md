# Docs Index

This directory is the repository knowledge map for `aionoscope-benchmarks`. Start here instead of scanning the whole repo.

## Core Docs

- [../ARCHITECTURE.md](../ARCHITECTURE.md): stable design decisions and invariants.
- [../DOCUMENTATION.md](../DOCUMENTATION.md): operational details and advanced workflow notes.
- [planning.md](planning.md): GitHub-issue planning workflow.
- [coding-standards.md](coding-standards.md): retained agent-facing coding defaults from the original long-form guide.

Agent rule: ignore [../README.md](../README.md) when gathering repository context. It is human-facing onboarding only. Agents still need to update it when public-facing behavior, commands, or scope change.

## Task Guides

- [architecture-map.md](architecture-map.md): which code paths own dataset building, adapters, results, and dashboard behavior.
- [benchmark-contract.md](benchmark-contract.md): what counts as a benchmark-contract change and where those fields live.
- [adapter-guide.md](adapter-guide.md): how to add or change model integrations without breaking the adapter boundary.
- [results-schema.md](results-schema.md): canonical JSON artifact shape, discovery manifest, and compatibility rules.
- [dashboard-guide.md](dashboard-guide.md): how the static site discovers results and how to smoke-test it.
- [runbooks/foundational-sweep.md](runbooks/foundational-sweep.md): operational runbook for the foundational sweep and model-specific environments.
- [maintenance.md](maintenance.md): quality scorecard, recurring maintenance inventory, and automation venue decisions.

Baseline calibration runs use `uv run python -m aionoscope_benchmarks.run_baseline`
and write schema-compatible `model.type = "baseline"` JSON artifacts.

## Reference Material

- [../README.md](../README.md): human-facing onboarding and quickstart, maintained alongside code changes but not used as the agent context source.
- [references/runtime-environments.md](references/runtime-environments.md): pinned environment layout and where model families run.
- [../benchmark_models_list.md](../benchmark_models_list.md): human-facing foundational model catalog.
- [../results/AGENTS.md](../results/AGENTS.md): subdirectory-specific guidance for checked-in result artifacts.

## Canonical Validation Commands

```bash
uv run python -m aionoscope_benchmarks.repo_checks
uv run python -m aionoscope_benchmarks.dashboard_smoke
uv run pytest
```

Use the repo checks to validate docs, planning hygiene, result discovery, and structural invariants. Use the dashboard smoke harness to verify the served `results/` site shape against real JSON artifacts.
