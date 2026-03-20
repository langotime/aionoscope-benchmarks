from __future__ import annotations

import enum
import sys
import types
from pathlib import Path

import pytest
import torch

import aionoscope_benchmarks.adapters.tempopfn as tempopfn_module
from aionoscope_benchmarks.adapters.tempopfn import TempoPFN38MAdapter


class _FakeBatchTimeSeriesContainer:
    def __init__(
        self,
        *,
        history_values: torch.Tensor,
        future_values: torch.Tensor,
        start: list[object],
        frequency: list[object],
        history_mask: torch.Tensor | None = None,
        future_mask: torch.Tensor | None = None,
    ) -> None:
        self.history_values = history_values
        self.future_values = future_values
        self.start = start
        self.frequency = frequency
        self.history_mask = history_mask
        self.future_mask = future_mask


class _FakeFrequency(enum.Enum):
    D = "D"


class _FakeTimeSeriesModel(torch.nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()


def _install_fake_tempopfn_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    snapshot_root = tmp_path / "tempo-pfn"
    (snapshot_root / "configs").mkdir(parents=True)
    (snapshot_root / "models").mkdir(parents=True)
    (snapshot_root / "src").mkdir(parents=True)

    (snapshot_root / "configs" / "example.yaml").write_text(
        "\n".join(
            [
                "gift_eval:",
                "  max_context_length: 3072",
                "TimeSeriesModel:",
                "  embed_size: 8",
                "  num_encoder_layers: 2",
                "  K_max: 3",
                "  encoder_config:",
                "    num_heads: 2",
                "    weaving: true",
                "    attn_mode: chunk",
                "    conv_size: 32",
                "    num_householder: 4",
                "  loss_type: quantile",
                "  quantiles: [0.1, 0.5, 0.9]",
            ]
        ),
        encoding="utf-8",
    )
    torch.save(
        {"model_state_dict": {}},
        snapshot_root / "models" / "checkpoint_38M.pth",
    )

    containers_module = types.ModuleType("src.data.containers")
    containers_module.BatchTimeSeriesContainer = _FakeBatchTimeSeriesContainer

    frequency_module = types.ModuleType("src.data.frequency")
    frequency_module.Frequency = _FakeFrequency

    time_features_module = types.ModuleType("src.data.time_features")

    def _compute_batch_time_features(
        *,
        start: list[object],
        history_length: int,
        future_length: int,
        batch_size: int,
        frequency: list[object],
        K_max: int,
        time_feature_config: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del start, frequency, time_feature_config
        return (
            torch.zeros(batch_size, history_length, K_max),
            torch.zeros(batch_size, future_length, K_max),
        )

    time_features_module.compute_batch_time_features = _compute_batch_time_features

    model_module = types.ModuleType("src.models.model")
    model_module.TimeSeriesModel = _FakeTimeSeriesModel

    src_package = types.ModuleType("src")
    data_package = types.ModuleType("src.data")
    models_package = types.ModuleType("src.models")
    src_package.data = data_package
    src_package.models = models_package
    data_package.containers = containers_module
    data_package.frequency = frequency_module
    data_package.time_features = time_features_module
    models_package.model = model_module

    monkeypatch.setitem(sys.modules, "src", src_package)
    monkeypatch.setitem(sys.modules, "src.data", data_package)
    monkeypatch.setitem(sys.modules, "src.data.containers", containers_module)
    monkeypatch.setitem(sys.modules, "src.data.frequency", frequency_module)
    monkeypatch.setitem(sys.modules, "src.data.time_features", time_features_module)
    monkeypatch.setitem(sys.modules, "src.models", models_package)
    monkeypatch.setitem(sys.modules, "src.models.model", model_module)
    monkeypatch.setattr(
        tempopfn_module,
        "snapshot_download",
        lambda **_: str(snapshot_root),
    )
    return snapshot_root


def test_tempopfn_adapter_uses_official_gift_eval_context_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot_root = _install_fake_tempopfn_runtime(monkeypatch, tmp_path)

    adapter = TempoPFN38MAdapter()

    assert adapter.snapshot_root == snapshot_root
    assert adapter.checkpoint_path == snapshot_root / "models" / "checkpoint_38M.pth"
    assert adapter.benchmark_sequence_length == 3072
    assert adapter.benchmark_sequence_length_source == "official_tempopfn_gift_eval_max_context_length"
    assert tuple(adapter.available_layers) == (0, 1, 2)

    metadata = adapter.adapter_metadata()
    assert metadata["benchmark_sequence_length"] == 3072
    assert metadata["context_length"] == 3072
    assert metadata["prediction_length"] == 1
    assert metadata["embed_size"] == 8
    assert metadata["num_hidden_layers"] == 2
    assert metadata["k_max"] == 3
    assert metadata["quantiles"] == [0.1, 0.5, 0.9]
    assert metadata["fixed_frequency"] == "D"
    assert metadata["time_grid_start"] == "2000-01-01"
    assert metadata["checkpoint_file"] == "models/checkpoint_38M.pth"
    assert metadata["parameter_count"] is None
    assert metadata["parameter_count_source"] == "unavailable"
    assert metadata["notes"] is not None
    assert ".venv-tempopfn" in str(metadata["notes"])


def test_tempopfn_adapter_requires_cuda_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_tempopfn_runtime(monkeypatch, tmp_path)
    adapter = TempoPFN38MAdapter()

    with pytest.raises(RuntimeError, match="requires CUDA"):
        adapter.prepare_runtime(device=torch.device("cpu"))
