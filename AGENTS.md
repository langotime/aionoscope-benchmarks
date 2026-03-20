# Agent Guide

`aionoscope-benchmarks` is the benchmark harness, not the upstream `aionoscope` / `aiono` generator library. This repo owns runtime dataset materialization, adapter-based model integration, offline probes, JSON result artifacts, and the static dashboard that reads those artifacts.

## Start Here

- [docs/index.md](docs/index.md): repo map and the fastest way to find the right doc.
- [docs/planning.md](docs/planning.md): planning workflow. Plans live in GitHub issues managed with `gh`, not in checked-in Markdown.
- [docs/coding-standards.md](docs/coding-standards.md): retained coding defaults from the original repository guide.
- [ARCHITECTURE.md](ARCHITECTURE.md): durable architecture decisions and repository invariants.
- [DOCUMENTATION.md](DOCUMENTATION.md): operational details, run commands, and result-schema expectations.
- [results/AGENTS.md](results/AGENTS.md): extra guidance that only applies inside `results/`.

Ignore `README.md` when gathering agent context. It is human-facing onboarding, not the agent knowledge source. Agents still must update it when user-facing behavior or workflow changes.

## Working Rules

- Default execution mode: treat requests as single-phase feature work unless the user explicitly asks for phased or agile delivery.
- Use `uv` for Python workflows and `pytest` for tests.
- Keep benchmark computation in Python. `results/dashboard.html` is a pure reader of `results/models/*.json`.
- Every model integration must go through a `FrozenTimeSeriesAdapter`; keep model-specific hacks inside the adapter layer or model registry.
- Treat `configs/dataset_aiono_basic_components_balanced.yaml` and `configs/probe.yaml` as the benchmark contract. Do not make silent contract changes.
- Fail fast with human-readable errors. Do not hide missing environments or missing dependencies.
- Do not create or keep checked-in plan Markdown under `plans/`. Historical plans belong in closed GitHub issues.
- After code changes, review `README.md`, `ARCHITECTURE.md`, and `DOCUMENTATION.md` in the same task.

## Canonical Checks

```bash
uv run python -m aionoscope_benchmarks.repo_checks
uv run python -m aionoscope_benchmarks.dashboard_smoke
uv run pytest
```

Run the repo checks plus the relevant test subset before reporting work complete.

## External APIs

Do not guess external-library APIs. Validate them against Context7 or primary upstream docs before coding against them.
