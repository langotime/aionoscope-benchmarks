# Plan: Align Agent Guidance With Benchmark Repo

## Goal

Make the repository instructions match the actual `aionoscope-benchmarks` codebase instead of the upstream `aionoscope` library guide. Add the missing top-level documentation files that the repo should now maintain, and remove guidance that assumes an `examples/` tree or generator-library internals.

## Steps

1. Review the current repo structure, entry points, runtime workflow, and result artifacts so the rewritten guidance reflects the real benchmark contract.
2. Add `ARCHITECTURE.md` describing the benchmark execution model, adapter registry, runtime dataset generation, probe pipeline, multi-environment runner, and result artifact boundaries.
3. Add `DOCUMENTATION.md` with operational details for running single-model and multi-model benchmarks, understanding configs, reading JSON outputs, and using the dashboard.
4. Rewrite `AGENTS.md` so it is benchmark-specific:
   - describe this repo as a benchmark runner, not the upstream generator library;
   - keep Python/pytest/uv guidance, but allow the existing multi-venv runner pattern;
   - remove `examples/` and notebook-sync rules;
   - replace generator-library-specific design rules with benchmark-specific invariants around seeds, validation splits, adapters, JSON results, and dashboard compatibility;
   - keep planning requirements and make them point to the new docs.
5. Review `README.md`, `ARCHITECTURE.md`, and `DOCUMENTATION.md` together to ensure the three documents have distinct roles and do not contradict each other.
6. Run relevant validation for the documentation change set and summarize the outcome.

## Documentation impact

- `ARCHITECTURE.md`: new file; will document the stable benchmark architecture and contracts that were previously undocumented in this repo.
- `README.md`: add brief pointers to the new top-level docs so onboarding users can discover the architecture and operational guides.
- `DOCUMENTATION.md`: new file; will document benchmark operations, result interpretation, and workflow details that should not live in the README.
