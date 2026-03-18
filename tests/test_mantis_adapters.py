from __future__ import annotations

import sys
import types

import pytest
import torch

from aionoscope_benchmarks.adapters.mantis import Mantis8MAdapter, MantisPlusAdapter


class _FakePositionalEncoding(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 1.0


class _AddConstant(torch.nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = float(delta)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.full_like(hidden_states, self.delta)


class _FakeTokenGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 1, 512] -> [B, 32, 2]
        patches = x[:, 0, :64].reshape(x.size(0), 32, 2)
        return self.proj(patches)


class _FakeMantisV1(torch.nn.Module):
    init_calls: list[dict[str, object]] = []
    load_calls: list[str] = []

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        type(self).init_calls.append(dict(kwargs))
        self.num_patches = 32
        self.tokgen_unit = _FakeTokenGenerator()
        self.transf_unit = types.SimpleNamespace(
            cls_token=torch.nn.Parameter(torch.tensor([100.0, 200.0], dtype=torch.float32)),
            pos_encoder=_FakePositionalEncoding(),
            transformer=types.SimpleNamespace(
                layers=[
                    torch.nn.ModuleList([_AddConstant(10.0), _AddConstant(1.0)]),
                    torch.nn.ModuleList([_AddConstant(20.0), _AddConstant(2.0)]),
                ]
            ),
        )

    def from_pretrained(self, checkpoint: str):
        type(self).load_calls.append(str(checkpoint))
        return self

    def eval(self):
        return self


def _install_fake_mantis(monkeypatch: pytest.MonkeyPatch) -> type[_FakeMantisV1]:
    _FakeMantisV1.init_calls = []
    _FakeMantisV1.load_calls = []

    architecture_module = types.ModuleType("mantis.architecture")
    architecture_module.MantisV1 = _FakeMantisV1

    mantis_module = types.ModuleType("mantis")
    mantis_module.architecture = architecture_module

    monkeypatch.setitem(sys.modules, "mantis", mantis_module)
    monkeypatch.setitem(sys.modules, "mantis.architecture", architecture_module)
    return _FakeMantisV1


def test_mantis_v1_adapters_load_official_checkpoints_and_use_exact_512_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mantis_v1 = _install_fake_mantis(monkeypatch)

    mantis = Mantis8MAdapter()
    mantis_plus = MantisPlusAdapter()
    x = torch.arange(512, dtype=torch.float32).reshape(1, 1, 512)

    mantis_reps = mantis.forward_layer_dict(x, layers=(0, 1, 2))
    mantis_plus_reps = mantis_plus.forward_layer_dict(x, layers=(0, 2))

    patch_tokens = x[:, 0, :64].reshape(1, 32, 2)
    base_tokens = torch.cat([torch.tensor([[[100.0, 200.0]]], dtype=torch.float32), patch_tokens], dim=1)
    layer0_hidden = base_tokens + 1.0
    layer1_hidden = layer0_hidden + 11.0
    layer2_hidden = layer1_hidden + 22.0

    def _combined(hidden_states: torch.Tensor) -> torch.Tensor:
        cls_token = hidden_states[:, 0, :]
        mean_token = hidden_states[:, 1:, :].mean(dim=1)
        return torch.cat([cls_token, mean_token], dim=1)

    assert fake_mantis_v1.init_calls == [
        {"return_transf_layer": -1, "output_token": "combined", "device": "cpu"},
        {"return_transf_layer": -1, "output_token": "combined", "device": "cpu"},
    ]
    assert fake_mantis_v1.load_calls == ["paris-noah/Mantis-8M", "paris-noah/MantisPlus"]

    assert mantis.benchmark_sequence_length == 512
    assert mantis.benchmark_sequence_length_source == "official_mantis_recommended_pretrained_length"
    assert tuple(mantis.available_layers) == (0, 1, 2)
    assert tuple(mantis_plus.available_layers) == (0, 1, 2)
    assert torch.allclose(mantis_reps[0], _combined(layer0_hidden))
    assert torch.allclose(mantis_reps[1], _combined(layer1_hidden))
    assert torch.allclose(mantis_reps[2], _combined(layer2_hidden))
    assert torch.allclose(mantis_plus_reps[0], _combined(layer0_hidden))
    assert torch.allclose(mantis_plus_reps[2], _combined(layer2_hidden))

    metadata = mantis_plus.adapter_metadata()
    assert metadata["benchmark_sequence_length"] == 512
    assert metadata["architecture_module"] == "MantisV1"
    assert metadata["output_token"] == "combined"
    assert metadata["num_patches"] == 32
    assert metadata["parameter_count_prefix_source"] == "cumulative_representation_path_unique_parameters"
