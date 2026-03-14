# Plan: Toto 4096 And Tabular TS Audit

## Goal

Update the benchmark contract so `Toto` uses a defensible exact native benchmark length of `4096`, and audit whether the current `TabPFN` and `TabICL` integrations should stay as tabular fallbacks or move to their time-series variants.

## Steps

1. Verify the official `Toto` sources for a defensible exact context length and confirm whether the current `4992` value is a benchmark heuristic rather than an upstream model recommendation.
2. Update the `Toto` adapter to use the chosen exact length and expose an honest `benchmark_sequence_length_source`.
3. Update the public docs and model notes to replace the old `Toto=4992` text with the new exact benchmark length.
4. Add or update tests so the `Toto` exact-length contract is covered alongside the existing sequence-length contract tests.
5. Audit official `TabPFN` and `TabICL` sources to explain the difference between the current tabular fallback adapters and their time-series-oriented variants, including whether a native exact sequence length exists.
6. Run the relevant `pytest` checks and summarize the recommended next step for `TabPFN` and `TabICL`.

## Documentation impact

- `ARCHITECTURE.md`: no stable architectural decision changes beyond the already-adopted exact-length contract, so no update should be needed unless the `Toto` rationale is currently described there.
- `README.md`: update the exact-length list so `Toto` is documented as `4096` instead of `4992`.
- `DOCUMENTATION.md`: update the operational exact-length table for `Toto`, and keep the `TabPFN` / `TabICL` audit as a recommendation unless code changes are made.
