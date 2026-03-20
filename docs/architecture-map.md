# Architecture Map

Use this file when you need to know which part of the repository owns a behavior before changing code.

## Ownership Map

- `aionoscope_benchmarks/runtime_dataset.py`: builds the finite benchmark train/validation splits and the dataset manifest.
- `aionoscope_benchmarks/model_registry.py`: canonical model catalog, environment mapping, and adapter lookup.
- `aionoscope_benchmarks/adapters/`: model-specific integrations behind `FrozenTimeSeriesAdapter`.
- `aionoscope_benchmarks/offline_probe.py`: probe training and frozen-feature evaluation.
- `aionoscope_benchmarks/results.py`: canonical result payload assembly.
- `aionoscope_benchmarks/run_model.py`: one-model execution path that fans out across enabled-component regimes.
- `aionoscope_benchmarks/run_many.py`: current-environment multi-model runner.
- `scripts/run_foundational_sequential.py`: full foundational sweep across pinned interpreters.
- `results/dashboard.html`: static dashboard that reads JSON artifacts only.

## Stable Boundaries

- Dataset generation, adapter integration, offline probing, result serialization, and dashboard presentation stay separate.
- Model-specific logic belongs in `aionoscope_benchmarks/adapters/` or `aionoscope_benchmarks/model_registry.py`, not in the benchmark pipeline.
- `results/models/*.json` is the machine-readable source of truth. `results/dashboard.html` is a consumer.
- Deployment may generate `results/models/list.txt` for static hosting, but it is not part of the dev-tree source of truth.

## Mechanical Enforcement

The repository now checks these boundaries with:

- `uv run python -m aionoscope_benchmarks.repo_checks`
- `uv run pytest tests/test_repo_contracts.py`

Those checks enforce the slim `AGENTS.md` contract, planning hygiene, result-corpus coherence, dev-tree cleanliness, and the adapter-module boundary declared in the model registry.
