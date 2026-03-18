from __future__ import annotations

import sys
import types

import pytest
import torch

from aionoscope_benchmarks.adapters.base import FrozenTimeSeriesAdapter
from aionoscope_benchmarks.adapters.mantisv2 import MantisV2Adapter
from aionoscope_benchmarks.adapters.toto import TotoAdapter
from aionoscope_benchmarks.constants import DATASET_CONFIG_PATH
from aionoscope_benchmarks.runtime_dataset import build_runtime_splits_by_validation_seed


class _DummyAdapter(FrozenTimeSeriesAdapter):
    model_name = "Dummy"
    model_slug = "Dummy"
    source = "local"
    checkpoint = "none"
    import_path = "none"
    benchmark_sequence_length = 123
    benchmark_sequence_length_source = "unit_test"

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(1, 1, bias=False)

    @property
    def available_layers(self) -> tuple[int, ...]:
        return (0,)

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        del layers
        self.validate_benchmark_input(x, channels=1)
        return {0: x.mean(dim=-1)}


class _ParameterlessAdapter(FrozenTimeSeriesAdapter):
    model_name = "Parameterless"
    model_slug = "Parameterless"
    source = "local"
    checkpoint = "none"
    import_path = "none"
    benchmark_sequence_length = 16
    benchmark_sequence_length_source = "unit_test"

    @property
    def available_layers(self) -> tuple[int, ...]:
        return (0,)

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        del layers
        self.validate_benchmark_input(x, channels=1)
        return {0: x.mean(dim=-1)}


class _PrefixCountAdapter(FrozenTimeSeriesAdapter):
    model_name = "PrefixCount"
    model_slug = "PrefixCount"
    source = "local"
    checkpoint = "none"
    import_path = "none"
    benchmark_sequence_length = 8
    benchmark_sequence_length_source = "unit_test"

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Linear(1, 2, bias=False)
        self.block1 = torch.nn.Linear(2, 2, bias=False)
        self.block2 = torch.nn.Linear(2, 2, bias=False)
        self.shared_norm = torch.nn.LayerNorm(2)

    @property
    def available_layers(self) -> tuple[int, ...]:
        return (0, 1, 2)

    def parameter_count_prefix_sources(self) -> dict[int, tuple[object, ...]]:
        return {
            0: (self.embedding, self.shared_norm),
            1: (self.block1,),
            2: (self.block2, self.shared_norm),
        }

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        del layers
        self.validate_benchmark_input(x, channels=1)
        pooled = x.mean(dim=-1)
        return {0: pooled, 1: pooled, 2: pooled}


def test_base_adapter_metadata_reports_exact_length_contract() -> None:
    adapter = _DummyAdapter()

    metadata = adapter.adapter_metadata()

    assert metadata["benchmark_sequence_length"] == 123
    assert metadata["benchmark_sequence_length_source"] == "unit_test"
    assert metadata["input_length_policy"] == "exact"
    assert metadata["parameter_count"] == 1
    assert metadata["parameter_count_total"] == 1
    assert metadata["trainable_parameter_count"] == 1
    assert metadata["parameter_count_source"] == "torch_registered_parameters"
    assert metadata["parameter_count_prefix_by_layer"] is None
    assert metadata["parameter_count_prefix_source"] == "unavailable"


def test_base_adapter_metadata_reports_unavailable_parameter_count_honestly() -> None:
    adapter = _ParameterlessAdapter()

    metadata = adapter.adapter_metadata()

    assert metadata["parameter_count"] is None
    assert metadata["parameter_count_total"] is None
    assert metadata["trainable_parameter_count"] is None
    assert metadata["parameter_count_source"] == "unavailable"
    assert metadata["parameter_count_prefix_by_layer"] is None
    assert metadata["parameter_count_prefix_source"] == "unavailable"


def test_base_adapter_metadata_reports_cumulative_prefix_parameter_counts() -> None:
    adapter = _PrefixCountAdapter()

    metadata = adapter.adapter_metadata()

    assert metadata["parameter_count"] == 14
    assert metadata["parameter_count_total"] == 14
    assert metadata["parameter_count_prefix_by_layer"] == {"0": 6, "1": 10, "2": 14}
    assert metadata["parameter_count_prefix_source"] == "cumulative_representation_path_unique_parameters"


def test_base_adapter_validation_rejects_wrong_sequence_length() -> None:
    adapter = _DummyAdapter()

    with pytest.raises(ValueError, match="exact sequence length 123"):
        adapter.forward_layer_dict(torch.zeros(2, 1, 122))


def test_runtime_split_builder_respects_exact_channel_size_override() -> None:
    manifest, train, val_splits = build_runtime_splits_by_validation_seed(
        config_path=DATASET_CONFIG_PATH,
        device=torch.device("cpu"),
        batch_size=2,
        channel_size_override=64,
        channel_size_policy_override="test_exact",
        channel_size_source_override="unit_test",
        train_batches=1,
        val_batches=1,
        validation_seed_values=[0],
        show_progress_bar=False,
    )

    assert manifest["default_channel_size"] == 5000
    assert manifest["channel_size"] == 64
    assert manifest["channel_size_policy"] == "test_exact"
    assert manifest["channel_size_source"] == "unit_test"
    assert manifest["benchmark_family"] == "aiono_basic_components"
    assert manifest["benchmark_version"] == "v1"
    assert manifest["baseline_sampling_frequency_hz"] == 500
    assert manifest["sampling_frequency"] == 500
    assert manifest["periodic_frequency_mode"] == "auto"
    assert manifest["sine_frequency_hz_resolved_low"] == pytest.approx(500.0 / 63.0)
    assert manifest["sine_frequency_hz_resolved_high"] == pytest.approx(225.0)
    assert manifest["sawtooth_frequency_hz_resolved_high"] == pytest.approx(100.0)
    assert manifest["square_frequency_hz_resolved_high"] == pytest.approx(25.0)
    assert manifest["square_min_points_in_shorter_plateau"] == 2
    assert manifest["square_duty_cycle_min"] == pytest.approx(0.1)
    assert manifest["square_duty_cycle_max"] == pytest.approx(0.9)
    periodic_specs = manifest["periodic_sampler_specs"]
    assert periodic_specs["square"]["frequency_hz"]["high"] == pytest.approx(25.0)
    assert periodic_specs["square"]["duty_cycle"]["low"] == pytest.approx(0.1)
    assert periodic_specs["square"]["duty_cycle"]["high"] == pytest.approx(0.9)
    assert tuple(train["x"].shape) == (2, 1, 64)
    assert tuple(val_splits[0]["x"].shape) == (2, 1, 64)


def test_mantis_v2_uses_official_recommended_pretrained_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    architecture_module = types.ModuleType("mantis.architecture")

    class _FakeMantisV2:
        def __init__(self, **_: object) -> None:
            self.num_patches = 32
            self.tokgen_unit = torch.nn.Linear(1, 1, bias=False)
            self.transf_unit = types.SimpleNamespace(
                cls_token=torch.nn.Parameter(torch.zeros(1)),
                transformer=types.SimpleNamespace(layers=[torch.nn.Linear(1, 1, bias=False)] * 5),
            )

        def from_pretrained(self, checkpoint: str):
            del checkpoint
            return self

        def eval(self):
            return self

    architecture_module.MantisV2 = _FakeMantisV2
    mantis_module = types.ModuleType("mantis")
    mantis_module.architecture = architecture_module
    monkeypatch.setitem(sys.modules, "mantis", mantis_module)
    monkeypatch.setitem(sys.modules, "mantis.architecture", architecture_module)

    adapter = MantisV2Adapter()

    metadata = adapter.adapter_metadata()
    assert adapter.benchmark_sequence_length == 512
    assert adapter.benchmark_sequence_length_source == "official_mantis_recommended_pretrained_length"
    assert metadata["benchmark_sequence_length"] == 512
    assert metadata["num_patches"] == 32


def test_toto_uses_official_quickstart_context_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toto_model_module = types.ModuleType("toto.model.toto")

    class _FakeBackbone:
        def __init__(self) -> None:
            self.patch_embed = torch.nn.Linear(1, 1, bias=False)
            self.patch_embed.patch_size = 64
            self.patch_embed.stride = 64
            self.embed_dim = 256
            self.num_layers = 12
            self.transformer = types.SimpleNamespace(
                layers=[torch.nn.Linear(1, 1, bias=False)] * self.num_layers
            )

        def eval(self):
            return self

    class _FakeToto:
        @classmethod
        def from_pretrained(cls, checkpoint: str, map_location: str = "cpu"):
            del cls, checkpoint, map_location
            return types.SimpleNamespace(model=_FakeBackbone())

    toto_model_module.Toto = _FakeToto
    toto_model_package = types.ModuleType("toto.model")
    toto_model_package.toto = toto_model_module
    toto_module = types.ModuleType("toto")
    toto_module.model = toto_model_package
    monkeypatch.setitem(sys.modules, "toto", toto_module)
    monkeypatch.setitem(sys.modules, "toto.model", toto_model_package)
    monkeypatch.setitem(sys.modules, "toto.model.toto", toto_model_module)

    adapter = TotoAdapter()

    metadata = adapter.adapter_metadata()
    assert adapter.benchmark_sequence_length == 4096
    assert (
        adapter.benchmark_sequence_length_source
        == "official_toto_open_base_quickstart_context_length"
    )
    assert metadata["benchmark_sequence_length"] == 4096
    assert metadata["patch_size"] == 64
    assert metadata["patch_stride"] == 64
