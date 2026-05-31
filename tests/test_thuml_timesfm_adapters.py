from __future__ import annotations

import sys
import types

import pytest
import torch

from aionoscope_benchmarks.adapters.thuml import (
    SundialBase128MAdapter,
    TimerBase84MAdapter,
)
from aionoscope_benchmarks.adapters.timesfm import TimesFM25Adapter


class _FakeTHUMLDecoder(torch.nn.Module):
    def __init__(self, input_token_len: int) -> None:
        super().__init__()
        self.input_token_len = int(input_token_len)
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        del use_cache, output_hidden_states, return_dict
        tokens = input_ids.reshape(input_ids.size(0), -1, self.input_token_len)
        base = torch.stack([tokens[..., 0], tokens[..., 1]], dim=-1)
        return types.SimpleNamespace(hidden_states=(base, base + 10.0, base + 20.0))


class _FakeTHUMLModel(torch.nn.Module):
    def __init__(self, input_token_len: int) -> None:
        super().__init__()
        self.dummy_parameter = torch.nn.Parameter(torch.zeros(()))
        self.config = types.SimpleNamespace(
            num_hidden_layers=2,
            input_token_len=int(input_token_len),
            hidden_size=2,
        )
        self.decoder = _FakeTHUMLDecoder(input_token_len=int(input_token_len))

    def get_decoder(self) -> torch.nn.Module:
        return self.decoder

    def eval(self):
        super().eval()
        return self


def _install_fake_transformers_for_thuml(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, dict[str, object]]]:
    load_calls: list[tuple[str, dict[str, object]]] = []

    class _FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, checkpoint: str, **kwargs: object):
            del cls, kwargs
            if checkpoint == "thuml/timer-base-84m":
                return types.SimpleNamespace(
                    num_hidden_layers=2,
                    input_token_len=96,
                    hidden_size=2,
                )
            if checkpoint == "thuml/sundial-base-128m":
                return types.SimpleNamespace(
                    num_hidden_layers=2,
                    input_token_len=16,
                    hidden_size=2,
                )
            raise AssertionError(f"Unexpected checkpoint {checkpoint}")

    class _FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, checkpoint: str, **kwargs: object):
            del cls
            load_calls.append((checkpoint, dict(kwargs)))
            if checkpoint == "thuml/timer-base-84m":
                return _FakeTHUMLModel(input_token_len=96)
            if checkpoint == "thuml/sundial-base-128m":
                return _FakeTHUMLModel(input_token_len=16)
            raise AssertionError(f"Unexpected checkpoint {checkpoint}")

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoConfig = _FakeAutoConfig
    transformers_module.AutoModelForCausalLM = _FakeAutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    return load_calls


class _FakeTimesFMTransformerLayer(torch.nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = float(delta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        decode_cache=None,
    ) -> tuple[torch.Tensor, None]:
        del mask, decode_cache
        return hidden_states + self.delta, None


class _FakeTimesFMTokenizer(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :2]


class _FakeTimesFMInnerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = 4
        self.o = 8
        self.os = 16
        self.md = 2
        self.x = 2
        self.config = types.SimpleNamespace(context_limit=16)
        self.tokenizer = _FakeTimesFMTokenizer()
        self.stacked_xf = torch.nn.ModuleList(
            [_FakeTimesFMTransformerLayer(10.0), _FakeTimesFMTransformerLayer(20.0)]
        )


def _install_fake_timesfm(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, object]]]:
    load_calls: list[tuple[str, dict[str, object]]] = []

    class _FakeTimesFMWrapper:
        @classmethod
        def from_pretrained(cls, checkpoint: str, **kwargs: object):
            del cls
            load_calls.append((checkpoint, dict(kwargs)))
            return types.SimpleNamespace(model=_FakeTimesFMInnerModel())

    util_module = types.ModuleType("timesfm.torch.util")

    def _update_running_stats(
        n: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
    ):
        del mu, sigma, masks
        return (n + values.size(-1), values.mean(dim=-1), torch.ones_like(values.mean(dim=-1))), None

    def _revin(
        inputs: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> torch.Tensor:
        del mu, sigma, reverse
        return inputs

    util_module.update_running_stats = _update_running_stats
    util_module.revin = _revin

    timesfm_torch_module = types.ModuleType("timesfm.torch")
    timesfm_torch_module.util = util_module
    timesfm_module = types.ModuleType("timesfm")
    timesfm_module.TimesFM_2p5_200M_torch = _FakeTimesFMWrapper
    timesfm_module.torch = timesfm_torch_module

    monkeypatch.setitem(sys.modules, "timesfm", timesfm_module)
    monkeypatch.setitem(sys.modules, "timesfm.torch", timesfm_torch_module)
    monkeypatch.setitem(sys.modules, "timesfm.torch.util", util_module)
    return load_calls


def test_timer_and_sundial_adapters_use_official_2880_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls = _install_fake_transformers_for_thuml(monkeypatch)

    timer = TimerBase84MAdapter()
    sundial = SundialBase128MAdapter()
    x = torch.arange(2880, dtype=torch.float32).reshape(1, 1, 2880)

    timer_reps = timer.forward_layer_dict(x, layers=(0, 1, 2))
    sundial_reps = sundial.forward_layer_dict(x, layers=(0, 2))

    timer_base = torch.tensor([[1392.0, 1393.0]], dtype=torch.float32)
    sundial_base = torch.tensor([[1432.0, 1433.0]], dtype=torch.float32)

    assert load_calls == [
        (
            "thuml/timer-base-84m",
            {
                "trust_remote_code": True,
                "attn_implementation": "eager",
                "torch_dtype": "auto",
            },
        ),
        (
            "thuml/sundial-base-128m",
            {
                "trust_remote_code": True,
                "attn_implementation": "eager",
                "torch_dtype": "auto",
            },
        ),
    ]

    assert timer.benchmark_sequence_length == 2880
    assert timer.benchmark_sequence_length_source == "official_timer_model_card_context_length"
    assert tuple(timer.available_layers) == (0, 1, 2)
    assert torch.allclose(timer_reps[0], timer_base)
    assert torch.allclose(timer_reps[1], timer_base + 10.0)
    assert torch.allclose(timer_reps[2], timer_base + 20.0)

    assert sundial.benchmark_sequence_length == 2880
    assert (
        sundial.benchmark_sequence_length_source
        == "official_sundial_readme_quickstart_lookback_length"
    )
    assert tuple(sundial.available_layers) == (0, 1, 2)
    assert torch.allclose(sundial_reps[0], sundial_base)
    assert torch.allclose(sundial_reps[2], sundial_base + 20.0)


def test_timesfm_adapter_uses_official_context_limit_and_layerwise_pooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls = _install_fake_timesfm(monkeypatch)

    adapter = TimesFM25Adapter()
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16)
    reps = adapter.forward_layer_dict(x, layers=(0, 1, 2))

    assert load_calls == [
        ("google/timesfm-2.5-200m-pytorch", {"torch_compile": False}),
    ]
    assert adapter.benchmark_sequence_length == 16
    assert adapter.benchmark_sequence_length_source == "timesfm_2p5_config.context_limit"
    assert tuple(adapter.available_layers) == (0, 1, 2)
    assert torch.allclose(reps[0], torch.tensor([[6.0, 7.0]], dtype=torch.float32))
    assert torch.allclose(reps[1], torch.tensor([[16.0, 17.0]], dtype=torch.float32))
    assert torch.allclose(reps[2], torch.tensor([[36.0, 37.0]], dtype=torch.float32))
