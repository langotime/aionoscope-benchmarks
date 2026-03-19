# Coding guide

This file provides guidance to AI agents when working in this repository.

## Repository Overview

This repo contains the `aionoscope-benchmarks` benchmark runner. It is a separate repo from the upstream `aionoscope` / `aiono` library and is responsible for:

- building the benchmark split at runtime from the sibling library;
- wrapping foundational models behind benchmark adapters;
- running frozen-feature offline probes;
- aggregating per-model JSON results;
- serving a static browser dashboard over those JSON artifacts.

Do not treat this repo as the source tree of the generator library itself.

## Work planning

- By default assume that requests are feature requests and should be implemented in a single phase.
- If I explicitly request an agile approach, split the work into phases where each phase delivers a usable research artifact.
- NO API GUESSING. Use Context7 MCP to validate external library APIs. If Context7 does not help, use Perplexity MCP to validate.

## Code Guidelines

- Use Python for benchmark orchestration, adapters, tests, result processing, and CLI entry points.
- Static HTML/JavaScript is allowed for the dashboard and should remain a thin presentation layer over JSON artifacts.
- Keep the code clean with a solid separation between benchmark computation, result serialization, and presentation.
- Make sure the code is idiomatic Python.
  - Follow PEP 585 and PEP 604.
- Make sure the code is DRY.
- Keep the code minimal.
- KISS.
  - Implement one best solution instead of carrying multiple fallback code paths.
  - If the best solution requires an unavailable dependency or external repo checkout, fail with a clear error instead of silently degrading.
- Fail fast and always raise a human-readable error with enough context to understand what happened and how to fix it.
- No silent benchmark-contract changes. CLI convenience defaults are fine when they are explicit and documented.
- Avoid defensive programming that hides broken assumptions.
- Use human-readable meaningful variable and function names. Avoid one-letter naming.
- In PyTorch-heavy code, add brief comments with tensor dimensions where shapes are not obvious from context.

## Python Development Tooling

- Use `uv` for Python package management.
- Use `pytest` for unit tests.
- Prefer `uv run python -m ...` when running Python code inside the current environment.
- The full foundational sweep intentionally uses multiple pinned environments such as `.venv-tivit` and `.venv-tabular`. Do not fight this architecture; `scripts/run_foundational_sequential.py` is allowed to dispatch directly to those interpreters.
- Never hide missing packages. If an environment is incomplete, fail immediately with a clear error.

## Code Maintenance Principles

- Always keep one canonical version of the code.
- When reimplementing existing code, do not keep the old code as fallback. Replace it and use tests to preserve behavior.
- ALWAYS run relevant tests or smoke checks after changing code and before reporting that the job is done.
- Put Python unit tests into the `tests/` subdirectory.
- DO NOT commit plan files to the git, and do NOT delete them, the human will delete them. You MUST update the project documentation with all important information from the plan file during its implementaiton. See the Documentation section for the documentation structure.

## Development Best Practices

- ALWAYS clean up temporary files.
- Use `uv run python -m` instead of ad-hoc import hacks when the current environment supports it.
- Avoid `sys.path.insert` in normal package code.
- Adapter bridge code may use narrowly scoped import shims when integrating vendored or external repos under `external/` and there is no cleaner import path. Keep those shims isolated to the adapter layer.
- This repo does not maintain a top-level `examples/` tree. Do not add process requirements around `examples/` or notebook mirroring unless I explicitly ask for that workflow.

## Engineering Best Practices

### Benchmark contract

- Treat `configs/dataset_aiono_basic_components_balanced.yaml` and `configs/probe.yaml` as the benchmark contract.
- Do not silently change train seed semantics, validation seed ordering, validation seed offsets, target definitions, or batch-count semantics.
- When the benchmark contract changes, make the change explicit in code, tests, and docs.

### Reproducibility

- Keep train-seed, validation-seed-value, and validation-generator-seed semantics explicit.
- Preserve reproducibility metadata in the dataset manifest and result JSON.
- If global seeding is required, centralize it in benchmark runner or probe code. Do not scatter `torch.manual_seed` calls across unrelated modules or adapters.
- Do not silently drop, reorder, or merge validation seeds during aggregation.

### Adapter boundaries

- Every model integration must go through a `FrozenTimeSeriesAdapter`.
- Keep model-specific hacks, external repo shims, and environment-specific logic inside the adapter layer or the model registry, not spread through the benchmark pipeline.
- Canonical benchmark model names must include the exact official version and size whenever a family publishes multiple checkpoints or generations. Prefer names such as `TimesFM-2.5-200M`, `Moirai-1.1-R-Small`, or `Toto-Open-Base-1.0`; do not introduce ambiguous family-only names such as `Moirai`. If an upstream repo-hosted singleton checkpoint is identified by a published variant token instead of a size marker, include that token too, for example `NuTime-Bias9` or `T-Loss-CricketX`.
- Use the official upstream repository and the official Hugging Face checkpoint whenever one exists. If the only official published checkpoint is hosted directly in the upstream repo instead of Hugging Face, treat that as an explicit documented exception and record it in `README.md`, `DOCUMENTATION.md`, and `benchmark_models_list.md`.
- Adapters must not use labels to build representations or introduce label-aware preprocessing shortcuts.
- If an adapter changes the train or validation split for technical reasons, it must do so deterministically and describe the behavior in `adapter_metadata()`.
- Adapter metadata must be honest and useful for interpreting benchmark outputs.
- If adapter metadata exposes parameter counts, keep the semantics explicit: `parameter_count` / `parameter_count_total` mean the full registered model path, while any layer-aware cumulative count must be published separately (for example `parameter_count_prefix_by_layer`) instead of overloading the total-count field.

### Results and dashboard

- `results/models/*.json` is the canonical machine-readable artifact for benchmark runs.
- `results/dashboard.html` is a consumer of that JSON schema, not an independent source of truth.
- If the JSON schema changes, update the dashboard, tests, `README.md`, `ARCHITECTURE.md`, and `DOCUMENTATION.md` in the same task.
- Do not move benchmark computations into browser-only logic.

### Performance

- Avoid recomputing representations when staged collected features can be reused.
- Be deliberate about memory use when materializing full train and validation feature tensors.
- Avoid Python loops over the entire dataset when vectorized or batched code is available and clearer.
- When adding a new model, choose `default_encode_batch_size` and any runtime-specific batch-size overrides for the actual benchmark GPU we run on, currently NVIDIA H200, with minimum end-to-end wall time as the target. Do not ship obviously conservative tiny defaults such as `1` or `2` unless the model, attention implementation, or memory footprint truly requires it. Push batch size up until throughput stops improving or stability/memory becomes unsafe, and document intentionally small settings in code or adapter metadata.

### Testability

- Test shapes and dtypes where tensor contracts matter.
- Test deterministic behavior for components that are expected to be reproducible with fixed seeds.
- Test validation-seed aggregation and ordering when changing result assembly.
- When changing result schema or dashboard-facing summaries, run a smoke check that the dashboard can still consume the produced JSON.

### Documentation

- Keep the three top-level docs distinct and coherent:
  - `ARCHITECTURE.md`: stable design decisions, execution model, adapter boundaries, result contracts, and other durable architectural choices.
  - `README.md`: concise onboarding, install/run basics, current benchmark scope, and the main mental model for users.
  - `DOCUMENTATION.md`: operational details, workflow details, result schema expectations, and advanced usage notes.
- After changing code, ALWAYS review whether `ARCHITECTURE.md`, `README.md`, and `DOCUMENTATION.md` must be updated, and update them in the same task when needed.
- If a code change does not require updating one of these files, explicitly verify that and mention it in the plan or review rather than silently skipping the check.

## Planning

- Write plans to files in Markdown.
- Put plans into the `plans/` subdirectory with a unique increasing numeric prefix, for example `001_short_description.md`.
- Every plan MUST include an explicit documentation review step for `ARCHITECTURE.md`, `README.md`, and `DOCUMENTATION.md`.
- Every plan MUST include a `Documentation impact` section with three explicit entries:
  - `ARCHITECTURE.md`: what stable design decisions will change, or why no change is needed.
  - `README.md`: what onboarding or public-facing text will change, or why no change is needed.
  - `DOCUMENTATION.md`: what usage or operational details will change, or why no change is needed.
- Before submitting a plan for review you MUST:
  - study the existing code base so the resulting benchmark API and internals remain coherent, lean, clean, and logical as a whole;
  - check whether similar logic already exists and propose refactoring instead of duplication;
  - describe how the change affects repository documentation;
  - make updating or explicitly rejecting updates to `ARCHITECTURE.md`, `README.md`, and `DOCUMENTATION.md` a mandatory part of the implementation plan, not an optional follow-up.
