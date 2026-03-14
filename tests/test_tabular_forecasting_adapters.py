from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from aionoscope_benchmarks.adapters.forecast_tables import build_context_dataframe
from aionoscope_benchmarks.adapters.tabicl_forecaster import TabICLForecasterAdapter
from aionoscope_benchmarks.adapters.tabpfn_ts import TabPFNTSAdapter
from aionoscope_benchmarks.constants import FOUNDATIONAL_MODELS
from aionoscope_benchmarks.model_registry import MODEL_SPECS, all_foundational_model_names


def _dummy_split(size: int, seq_len: int) -> dict[str, torch.Tensor]:
    return {
        "x": torch.arange(size * seq_len, dtype=torch.float32).reshape(size, 1, seq_len),
        "y_cls": torch.zeros(size, 14, dtype=torch.float32),
        "y_dense": torch.zeros(size, 34, dtype=torch.float32),
    }


def _install_fake_tabpfn_ts_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    tabpfn_ts_module = types.ModuleType("tabpfn_time_series")

    class _FakeFeatureTransformer:
        def __init__(self, features):
            self.features = list(features)

    class _FakeTimeSeriesDataFrame:
        @classmethod
        def from_data_frame(cls, df):
            return df.copy()

    tabpfn_ts_module.FeatureTransformer = _FakeFeatureTransformer
    tabpfn_ts_module.TimeSeriesDataFrame = _FakeTimeSeriesDataFrame
    tabpfn_ts_module.TABPFN_TS_DEFAULT_FEATURES = [object(), object()]
    tabpfn_ts_module.TABPFN_DEFAULT_CONFIG = {"model_path": "dummy-tabpfn-regressor.ckpt"}

    tabpfn_ts_data_module = types.ModuleType("tabpfn_time_series.data_preparation")

    def _fake_generate_test_x(context_tsdf, prediction_length: int):
        del prediction_length
        return context_tsdf.copy()

    tabpfn_ts_data_module.generate_test_X = _fake_generate_test_x

    monkeypatch.setitem(sys.modules, "tabpfn_time_series", tabpfn_ts_module)
    monkeypatch.setitem(sys.modules, "tabpfn_time_series.data_preparation", tabpfn_ts_data_module)


def _install_fake_tabicl_forecast_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    tabicl_module = types.ModuleType("tabicl")

    class _FakeTabICL:
        def __init__(self) -> None:
            self.icl_predictor = types.SimpleNamespace(
                tf_icl=types.SimpleNamespace(blocks=[object()] * 12)
            )

    class _FakeTabICLForecaster:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    tabicl_module.TabICL = _FakeTabICL
    tabicl_module.TabICLForecaster = _FakeTabICLForecaster
    monkeypatch.setitem(sys.modules, "tabicl", tabicl_module)


def test_registry_contains_new_tabular_forecasting_models() -> None:
    assert "TabPFN-TS" in MODEL_SPECS
    assert "TabICLForecaster" in MODEL_SPECS
    assert "TabPFN-TS" in FOUNDATIONAL_MODELS
    assert "TabICLForecaster" in FOUNDATIONAL_MODELS
    assert "TabPFN-TS" in all_foundational_model_names()
    assert "TabICLForecaster" in all_foundational_model_names()


def test_build_context_dataframe_is_deterministic() -> None:
    pd = pytest.importorskip("pandas")
    waveforms = np.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]], dtype=np.float32)

    frame_a = build_context_dataframe(waveforms, sampling_frequency_hz=500)
    frame_b = build_context_dataframe(waveforms, sampling_frequency_hz=500)

    assert frame_a.equals(frame_b)
    assert list(frame_a.columns) == ["item_id", "timestamp", "target"]
    assert frame_a["item_id"].tolist() == [0, 0, 0, 1, 1, 1]
    assert frame_a["target"].tolist() == [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]
    assert frame_a["timestamp"].iloc[1] - frame_a["timestamp"].iloc[0] == pd.Timedelta(milliseconds=2)


def test_tabpfn_ts_adapter_uses_cached_exact_length_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_tabpfn_ts_modules(monkeypatch)

    def _fake_encode_waveforms(
        self: TabPFNTSAdapter,
        waveforms: np.ndarray,
        *,
        split_name: str,
    ) -> dict[int, torch.Tensor]:
        del split_name
        return {0: torch.from_numpy(waveforms[:, :3].copy())}

    monkeypatch.setattr(TabPFNTSAdapter, "_encode_waveforms", _fake_encode_waveforms)

    adapter = TabPFNTSAdapter()
    train_split = _dummy_split(size=3, seq_len=4096)
    val_split = _dummy_split(size=2, seq_len=4096)
    adapter.prepare(
        manifest={"sampling_frequency": 500},
        train_split=train_split,
        val_split=val_split,
    )

    assert adapter.available_layers == (0,)
    assert adapter.benchmark_sequence_length == 4096
    assert adapter.benchmark_sequence_length_source == "tabpfn_ts_pipeline.max_context_length"

    metadata = adapter.adapter_metadata()
    assert metadata["prediction_length"] == 1
    assert metadata["benchmark_sequence_length"] == 4096
    assert metadata["forecast_query_policy"] == "synthetic_next_step"
    assert metadata["probe_train_sample_count"] == 3
    assert metadata["probe_val_sample_count"] == 2

    representation_fn = adapter.make_representation_fn(layers=(0,), split="train")
    first = representation_fn(torch.zeros(2, 1, 4096))
    second = representation_fn(torch.zeros(1, 1, 4096))
    assert torch.equal(first[0], adapter._split_feature_cache["train"][0][:2])
    assert torch.equal(second[0], adapter._split_feature_cache["train"][0][2:3])

    with pytest.raises(ValueError, match="only exposes layer 0"):
        adapter.make_representation_fn(layers=(1,), split="train")
    with pytest.raises(RuntimeError, match="cached split features"):
        adapter.forward_layer_dict(torch.zeros(1, 1, 4096))


def test_tabicl_forecaster_adapter_exposes_layered_cached_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_tabicl_forecast_modules(monkeypatch)

    def _fake_encode_waveforms(
        self: TabICLForecasterAdapter,
        waveforms: np.ndarray,
        *,
        split_name: str,
    ) -> dict[int, torch.Tensor]:
        del split_name
        base = torch.from_numpy(waveforms[:, :4].copy())
        return {
            layer: base + float(layer)
            for layer in self.available_layers
        }

    monkeypatch.setattr(TabICLForecasterAdapter, "_encode_waveforms", _fake_encode_waveforms)

    adapter = TabICLForecasterAdapter()
    train_split = _dummy_split(size=3, seq_len=4096)
    val_split = _dummy_split(size=2, seq_len=4096)
    adapter.prepare(
        manifest={"sampling_frequency": 500},
        train_split=train_split,
        val_split=val_split,
    )

    assert adapter.available_layers == tuple(range(13))
    assert adapter.benchmark_sequence_length == 4096
    assert adapter.benchmark_sequence_length_source == "tabicl_forecaster.max_context_length"

    metadata = adapter.adapter_metadata()
    assert metadata["prediction_length"] == 1
    assert metadata["benchmark_sequence_length"] == 4096
    assert metadata["forecast_query_policy"] == "synthetic_next_step"
    assert metadata["probe_train_sample_count"] == 3
    assert metadata["probe_val_sample_count"] == 2

    requested_layers = (0, 5, 12)
    representation_fn = adapter.make_representation_fn(layers=requested_layers, split="val")
    batch = representation_fn(torch.zeros(2, 1, 4096))
    assert set(batch) == set(requested_layers)
    for layer in requested_layers:
        assert torch.equal(batch[layer], adapter._split_feature_cache["val"][layer][:2])

    with pytest.raises(ValueError, match="does not expose layers"):
        adapter.make_representation_fn(layers=(99,), split="val")
    with pytest.raises(RuntimeError, match="cached split features"):
        adapter.forward_layer_dict(torch.zeros(1, 1, 4096))
