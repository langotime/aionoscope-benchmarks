# Coding Standards

This file preserves the general coding defaults that used to live in the old long-form `AGENTS.md`.

## Execution Defaults

- Treat requests as feature requests implemented in one phase unless the user explicitly asks for phased or agile delivery.
- Do not guess external APIs. Validate them with Context7 first, then primary upstream docs if needed.

## Code Style

- Use Python for benchmark orchestration, adapters, tests, result processing, and CLI entry points.
- Keep code idiomatic Python and follow PEP 585 plus PEP 604.
- Keep code DRY, minimal, and KISS.
- Prefer one best implementation instead of carrying multiple fallback paths.
- Use meaningful names and avoid one-letter identifiers.
- In PyTorch-heavy code, add brief tensor-dimension comments where shapes are not obvious.

## Maintenance Rules

- Keep one canonical version of the code.
- When replacing code, remove the old fallback path and preserve behavior with tests.
- Always run relevant tests or smoke checks before reporting completion.
- Put Python unit tests under `tests/`.
- Always clean up temporary files.

## Import and Repo Hygiene

- Prefer `uv run python -m ...` over ad hoc import hacks when the environment supports it.
- Avoid `sys.path.insert` in normal package code.
- Keep adapter bridge import shims narrow and isolated to the adapter layer when integrating vendored or external repos under `external/`.
- This repo does not maintain a top-level `examples/` tree. Do not add process requirements around `examples/` or notebook mirroring unless explicitly requested.
