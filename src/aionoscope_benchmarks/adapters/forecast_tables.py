from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


def sample_probe_indices(
    size: int,
    *,
    sample_cap: int,
    seed: int,
) -> np.ndarray:
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if sample_cap <= 0:
        raise ValueError(f"sample_cap must be positive, got {sample_cap}")
    if sample_cap >= size:
        return np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices = rng.choice(size, size=sample_cap, replace=False)
    indices.sort()
    return indices.astype(np.int64, copy=False)


def subset_split(
    split: Mapping[str, torch.Tensor],
    indices: np.ndarray,
) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in split.items()}


def build_context_dataframe(
    waveforms: np.ndarray,
    *,
    sampling_frequency_hz: int,
):
    import pandas as pd

    waveforms = np.asarray(waveforms, dtype=np.float32)
    if waveforms.ndim != 2:
        raise ValueError(f"Expected [N, L] waveform array, got shape {waveforms.shape}")
    if waveforms.shape[0] <= 0 or waveforms.shape[1] <= 0:
        raise ValueError(f"Waveforms must be non-empty, got shape {waveforms.shape}")
    if sampling_frequency_hz <= 0:
        raise ValueError(f"sampling_frequency_hz must be positive, got {sampling_frequency_hz}")

    num_series, seq_len = waveforms.shape
    step = pd.to_timedelta(1.0 / float(sampling_frequency_hz), unit="s")
    timestamps = pd.date_range(start="2000-01-01 00:00:00", periods=seq_len, freq=step)

    return pd.DataFrame(
        {
            "item_id": np.repeat(np.arange(num_series, dtype=np.int64), seq_len),
            "timestamp": np.tile(timestamps.to_numpy(copy=False), num_series),
            "target": waveforms.reshape(num_series * seq_len),
        }
    )


def make_cached_representation_fn(
    *,
    model_name: str,
    split_feature_cache: dict[str, dict[int, torch.Tensor]],
    layers: tuple[int, ...],
    split: str,
):
    requested_layers = tuple(int(layer) for layer in layers)
    if not requested_layers:
        raise ValueError(f"{model_name} requires at least one requested layer")

    split_cache = split_feature_cache.get(split)
    if split_cache is None:
        raise RuntimeError(f"{model_name} features are not prepared yet for split={split!r}")

    missing_layers = [layer for layer in requested_layers if layer not in split_cache]
    if missing_layers:
        raise ValueError(
            f"{model_name} does not have cached features for split={split!r} "
            f"layers={missing_layers}"
        )

    total_size = int(split_cache[requested_layers[0]].size(0))
    for layer in requested_layers[1:]:
        layer_size = int(split_cache[layer].size(0))
        if layer_size != total_size:
            raise RuntimeError(
                f"{model_name} cached feature size mismatch for split={split!r}: "
                f"layer {requested_layers[0]} has {total_size} rows but layer {layer} has {layer_size}"
            )

    offset = 0

    def _representation_fn(x: torch.Tensor) -> dict[int, torch.Tensor]:
        nonlocal offset
        batch_size = int(x.size(0))
        start = offset
        stop = start + batch_size
        if stop > total_size:
            raise ValueError(
                f"{model_name} cached features exhausted for split={split}: "
                f"requested stop={stop} total={total_size}"
            )
        offset = stop
        return {layer: split_cache[layer][start:stop] for layer in requested_layers}

    return _representation_fn
