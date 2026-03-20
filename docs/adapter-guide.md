# Adapter Guide

Every model integration in this repo must go through `FrozenTimeSeriesAdapter`.

## Where Adapter Work Lives

- registry entry: [../aionoscope_benchmarks/model_registry.py](../aionoscope_benchmarks/model_registry.py)
- adapter implementations: [../aionoscope_benchmarks/adapters](../aionoscope_benchmarks/adapters)
- adapter-specific tests: [../tests](../tests)

## Required Adapter Properties

- exact benchmark sequence length
- stable `available_layers`
- deterministic benchmark-facing preprocessing when needed
- truthful `adapter_metadata()`
- fail-fast input-length validation

## Boundary Rules

- Keep external-repo shims and environment-specific logic inside the adapter layer.
- Do not add label-aware preprocessing shortcuts.
- Do not silently crop, pad, or resample benchmark inputs.
- Canonical benchmark model names must stay explicit about version/size or published variant token.

## Validation Path

For adapter changes, run:

```bash
uv run python -m aionoscope_benchmarks.repo_checks
uv run pytest tests/test_extended_foundation_registry.py tests/test_sequence_length_contract.py
```

Then run the adapter-specific test subset and at least one relevant benchmark or encode-only smoke path if the change touches runtime behavior.
