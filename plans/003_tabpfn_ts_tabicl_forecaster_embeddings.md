# Plan: TabPFN-TS And TabICLForecaster Embedding Adapters

## Goal

Add two new benchmark models, `TabPFN-TS` and `TabICLForecaster`, by extracting frozen context-conditioned embeddings that can be used by the existing offline classification probes.

The implementation should follow the same benchmark pattern as `Chronos2` and `Moirai`: use the forecasting stack only up to a pre-head representation boundary, then cache one embedding vector per benchmark sample for the downstream probes.

## Confirmed API Facts

- The current benchmark adapter contract already supports cached split features and single-layer adapters, as shown by the existing `TabPFN` and `TabICL` fallbacks in `src/aionoscope_benchmarks/adapters/tabpfn.py` and `src/aionoscope_benchmarks/adapters/tabicl.py`.
- `TabPFN` has an official `get_embeddings(X, data_source={"train","test"})` API on `TabPFNRegressor` / `TabPFNClassifier`, returning `(n_estimators, n_samples, embedding_dim)` for a fitted model.
- `TabICLForecaster` is only a thin forecasting wrapper around `TabICLRegressor` plus time-series feature engineering.
- `TabICLRegressor` already exposes a representation boundary internally via `kv_cache="repr"` and the raw `TabICL` model methods `prepare_repr_cache(...)` and `forward_with_repr_cache(...)`.
- `TabICLForecaster` and the official `TabPFN-TS` framing are per-time-series forecasting pipelines, so calling the high-level forecaster once per benchmark sample would be too expensive at benchmark scale. We should reuse their official time-series preprocessing, but bypass the high-level per-series prediction loop during embedding extraction.

## Proposed Design

### 1. Shared time-series-to-forecast-table builder

Create a small shared helper module for the two new adapters that:

1. converts each benchmark waveform `[B, 1, L]` into a deterministic single-item time-series table with benchmark timestamps derived from `sampling_frequency`;
2. splits each waveform into:
   - context: the first `L - prediction_length` timesteps with observed target values;
   - forecast horizon: the last `prediction_length` timesteps with target hidden;
3. runs the official time-series feature engineering stack to produce:
   - `train_X`
   - `train_y`
   - `test_X`

Use `prediction_length = 1` for both adapters.

Reasoning:

- it matches the current forecasting adapters, which extract one representation per context rather than averaging over a long synthetic forecast horizon;
- it removes the need for temporal pooling over multiple future rows;
- it maximizes usable context length under the exact-length benchmark contract.

### 2. `TabPFNTSAdapter`

Implement a new adapter with `available_layers = (0,)`.

Extraction path:

1. use the official `TabPFN-TS` preprocessing / feature-transform path to convert each benchmark waveform into a tabular forecasting problem;
2. fit the underlying `TabPFNRegressor` on `train_X, train_y`;
3. call the official `TabPFNRegressor.get_embeddings(test_X, data_source="test")`;
4. average over the estimator axis;
5. because `prediction_length = 1`, take the single returned test-row embedding as the benchmark representation for that sample.

Important constraint:

- do not treat `TabPFN-TS` as a sequence encoder with internal layer access unless the installed package exposes that cleanly;
- the first implementation should use the supported `get_embeddings(...)` API and expose only layer `0`.

### 3. `TabICLForecasterAdapter`

Implement a new adapter with a context-conditioned representation cut inside the official `TabICL` model.

Extraction path:

1. use the official `TabICLForecaster` preprocessing / transform path to obtain `train_X, train_y, test_X`;
2. fit `TabICLRegressor` on `train_X, train_y`, with `kv_cache="repr"` enabled;
3. bypass forecast output decoding and extract representations from the raw `TabICL` model.

Representation boundary:

- `layer 0`: row interaction output for the held-out forecast row, after the official column embedding + row interaction stack;
- `layers 1..N`: hidden states after each ICL transformer block for the held-out forecast row, before the final decoder head.

Implementation detail:

- mirror the existing `Chronos2` / `Moirai` adapter style by manually stepping through the raw model blocks rather than trying to scrape logits or quantiles from the forecasting wrapper;
- use `prepare_repr_cache(...)` for the train portion so the held-out row embedding is conditioned on the observed history in the same way as forecasting.

This gives us a genuinely forecasting-conditioned frozen representation instead of a post-hoc forecast scalar.

### 4. Scale strategy

Do not use the public `TabICLForecaster.predict_df()` or the public `TabPFN-TS` high-level prediction loop directly inside the benchmark path.

Instead:

- reuse the official time-series preprocessing and table construction;
- extract embeddings in adapter `prepare(...)` / `update_probe_val_split(...)`;
- cache per-split features exactly like the current `TabPFN` / `TabICL` fallback adapters.

For the first implementation, keep explicit probe sample caps for both new adapters, similar to the current tabular fallbacks. This is the pragmatic way to keep runtime bounded without pretending the public forecasting wrappers are cheap enough to run over the full benchmark split.

## Steps

1. Add a shared forecasting-table utility module for deterministic timestamp creation, `prediction_length=1` splitting, and official time-series feature transformation.
2. Add `TabPFN-TS` environment wiring and model registry entries:
   - new adapter module, likely `src/aionoscope_benchmarks/adapters/tabpfn_ts.py`;
   - new `ModelSpec`;
   - new environment entry in `scripts/run_foundational_sequential.py`;
   - fail fast if the `tabpfn-time-series` package is not installed.
3. Implement `TabPFNTSAdapter` as a cached single-layer adapter using official `TabPFNRegressor.get_embeddings(...)`.
4. Add `TabICLForecaster` environment wiring and model registry entries:
   - new adapter module, likely `src/aionoscope_benchmarks/adapters/tabicl_forecaster.py`;
   - new `ModelSpec`;
   - environment requirement for `tabicl[forecast]`;
   - fail fast if the forecasting extra is unavailable.
5. Implement `TabICLForecasterAdapter` by reusing official forecasting transforms but extracting layerwise frozen representations from the raw `TabICL` model before the decoder head.
6. Reuse the current cached-feature adapter pattern:
   - `prepare(...)` builds and stores train embeddings;
   - `update_probe_val_split(...)` builds and stores validation embeddings;
   - `make_representation_fn(...)` serves the cached tensors to the probe runner.
7. Add tests:
   - feature-table builder shape and determinism tests;
   - adapter metadata and exact-length contract tests;
   - monkeypatched smoke tests for `TabPFNTSAdapter` and `TabICLForecasterAdapter`;
   - one regression test that the adapters expose cached split features rather than raw `forward_layer_dict(...)`.
8. Run the relevant `pytest` suite and one adapter smoke check in the target environments.
9. Review and update `ARCHITECTURE.md`, `README.md`, and `DOCUMENTATION.md` in the same task.

## Risks And Constraints

- `TabPFN-TS` does not appear to expose a forecasting-specific embedding API of its own, so the adapter should depend on the official underlying `TabPFNRegressor.get_embeddings(...)` interface after time-series featurization.
- `TabICLForecaster` is a forecasting wrapper, not an encoder API. We should treat it as a preprocessing pipeline plus a raw `TabICL` model, not as a black-box feature extractor.
- Exact benchmark sequence length for `TabICLForecaster` is straightforward if we adopt its official `max_context_length=4096`.
- Exact benchmark sequence length for `TabPFN-TS` still needs one explicit implementation-time decision if the package does not expose a native fixed context limit. If no native exact context exists, the implementation must either:
  - document the chosen benchmark context as an explicit policy value; or
  - stop and ask whether `TabPFN-TS` belongs in the model-native exact-length track.

## Documentation impact

- `ARCHITECTURE.md`: update the adapter-boundary section to describe the new forecasting-to-embedding pattern, where a forecasting pipeline may be integrated by cutting at a pre-head frozen representation boundary rather than using forecast scalars.
- `README.md`: add `TabPFN-TS` and `TabICLForecaster` to the foundational model list, document their environment prerequisites, and explain at a high level that they use forecasting-conditioned embeddings rather than the older tabular fallback heads.
- `DOCUMENTATION.md`: add operational details for the new adapters, including exact context contract, `prediction_length=1`, cached split-feature behavior, sample caps if retained, and the model-specific metadata fields exposed for the new forecasting-derived representations.
