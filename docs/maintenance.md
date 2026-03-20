# Maintenance

This file records the benchmark-harness quality scorecard and the agreed automation venue for recurring checks.

## Quality Scorecard

The repo is healthy when all of these are true:

- `AGENTS.md` stays a concise table of contents and points at current docs.
- planning uses GitHub issues via `gh`, with no checked-in `plans/*.md`.
- the required `docs/` knowledge map exists and local cross-links resolve.
- deployment-only artifacts such as `results/models/list.txt` do not live in the dev tree.
- the checked-in result corpus contains one coherent benchmark family/version.
- every registry model points at an adapter module under `aionoscope_benchmarks/adapters/`.
- the dashboard smoke harness can serve `results/` and load real JSON artifacts.

## Recurring Maintenance Inventory

- Doc structure, cross-link, stale-reference, planning-hygiene, deploy-artifact, and structural checks:
  Run on every push to active branches in GitHub Actions. Also run weekly on a schedule because they are cheap and fully repo-local.
- Dashboard smoke against checked-in result JSON:
  Run on every push to active branches in GitHub Actions.
- Lightweight benchmark smoke in the base environment:
  Keep case by case. Add a manual workflow only if the runtime is stable enough for shared automation.
- Full foundational multi-environment or GPU-heavy sweeps:
  Keep off the default GitHub Actions path. Run case by case on the benchmark machine or another dedicated runner.
- Dependency freshness and security scanning:
  Use GitHub-native tooling case by case.

## Workflow Files

- [../.github/workflows/repo-checks.yml](../.github/workflows/repo-checks.yml)
- [../.github/workflows/weekly-gardening.yml](../.github/workflows/weekly-gardening.yml)

## Local Command

```bash
uv run python -m aionoscope_benchmarks.repo_checks
```

That command is the local equivalent of the automated doc/planning/result-structure gate.
