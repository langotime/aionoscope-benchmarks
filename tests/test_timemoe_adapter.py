from __future__ import annotations

import sys
import types

import pytest
import torch

from aionoscope_benchmarks.adapters.timemoe import TimeMoeBaseAdapter, TimeMoeLargeAdapter
from aionoscope_benchmarks.constants import FOUNDATIONAL_MODELS
from aionoscope_benchmarks.model_registry import MODEL_SPECS, all_foundational_model_names


class _FakeFinalNorm(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 100.0


class _FakeDecoderLayer(torch.nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = float(delta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        del attention_mask, position_ids, past_key_value, output_attentions, use_cache
        return hidden_states + self.delta, None, None, None


class _FakeDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_layer = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.embed_layer.weight.copy_(torch.tensor([[1.0], [2.0]], dtype=torch.float32))
        self.layers = torch.nn.ModuleList([
            _FakeDecoderLayer(10.0),
            _FakeDecoderLayer(20.0),
        ])
        self.norm = _FakeFinalNorm()


class _FakeTimeMoeModel(torch.nn.Module):
    def __init__(self, config: types.SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.decoder = _FakeDecoder()

    def get_decoder(self) -> torch.nn.Module:
        return self.decoder

    def eval(self):
        super().eval()
        return self


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> tuple[type, type]:
    config_loads: list[tuple[str, dict[str, object]]] = []
    model_loads: list[tuple[str, dict[str, object]]] = []

    def _make_config() -> types.SimpleNamespace:
        return types.SimpleNamespace(
            model_type="time_moe",
            input_size=1,
            hidden_size=2,
            num_hidden_layers=2,
            num_experts=8,
            num_experts_per_tok=2,
            horizon_lengths=[1, 8, 32, 64],
            max_position_embeddings=4096,
            _attn_implementation="eager",
        )

    class _FakeAutoConfig:
        load_calls = config_loads

        @classmethod
        def from_pretrained(cls, checkpoint: str, **kwargs: object):
            del cls
            config_loads.append((checkpoint, dict(kwargs)))
            return _make_config()

    class _FakeAutoModelForCausalLM:
        load_calls = model_loads

        @classmethod
        def from_pretrained(cls, checkpoint: str, **kwargs: object):
            model_loads.append((checkpoint, dict(kwargs)))
            config = _make_config()
            config._attn_implementation = str(kwargs.get("attn_implementation", "eager"))
            return _FakeTimeMoeModel(config)

    modeling_attn_mask_utils = types.ModuleType("transformers.modeling_attn_mask_utils")

    def _prepare_4d_causal_attention_mask(
        attention_mask: torch.Tensor | None,
        input_shape: tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        sliding_window=None,
    ) -> torch.Tensor:
        del attention_mask, inputs_embeds, past_key_values_length, sliding_window
        batch_size, seq_length = input_shape
        return torch.zeros(batch_size, 1, seq_length, seq_length, dtype=torch.float32)

    modeling_attn_mask_utils._prepare_4d_causal_attention_mask = _prepare_4d_causal_attention_mask

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoConfig = _FakeAutoConfig
    transformers_module.AutoModelForCausalLM = _FakeAutoModelForCausalLM
    transformers_module.modeling_attn_mask_utils = modeling_attn_mask_utils

    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "transformers.modeling_attn_mask_utils", modeling_attn_mask_utils)
    return _FakeAutoConfig, _FakeAutoModelForCausalLM


def test_timemoe_registry_contains_base_and_large() -> None:
    assert "Time-MoE-50M" in MODEL_SPECS
    assert "Time-MoE-200M" in MODEL_SPECS
    assert MODEL_SPECS["Time-MoE-50M"].checkpoint == "Maple728/TimeMoE-50M"
    assert MODEL_SPECS["Time-MoE-200M"].checkpoint == "Maple728/TimeMoE-200M"
    assert MODEL_SPECS["Time-MoE-50M"].env == "timemoe"
    assert MODEL_SPECS["Time-MoE-200M"].env == "timemoe"
    assert "Time-MoE-50M" in FOUNDATIONAL_MODELS
    assert "Time-MoE-200M" in FOUNDATIONAL_MODELS
    assert "Time-MoE-50M" in all_foundational_model_names()
    assert "Time-MoE-200M" in all_foundational_model_names()


def test_timemoe_adapter_uses_published_context_length_and_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_auto_config, fake_auto_model = _install_fake_transformers(monkeypatch)

    adapter = TimeMoeBaseAdapter()
    x = torch.arange(4096, dtype=torch.float32).reshape(1, 1, 4096)
    reps = adapter.forward_layer_dict(x, layers=(0, 1, 2))

    normalized = x[:, 0, :]
    normalized = (normalized - normalized.mean(dim=-1, keepdim=True)) / normalized.std(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-6)
    embedded = torch.cat([normalized.unsqueeze(-1), (2.0 * normalized).unsqueeze(-1)], dim=-1)
    layer1 = embedded + 10.0
    layer2 = layer1 + 20.0 + 100.0

    assert fake_auto_config.load_calls == [("Maple728/TimeMoE-50M", {"trust_remote_code": True})]
    assert fake_auto_model.load_calls == [
        (
            "Maple728/TimeMoE-50M",
            {
                "trust_remote_code": True,
                "torch_dtype": "auto",
                "attn_implementation": "eager",
            },
        )
    ]
    assert tuple(adapter.available_layers) == (0, 1, 2)
    assert adapter.benchmark_sequence_length == 4096
    assert adapter.benchmark_sequence_length_source == "time_moe_config.max_position_embeddings"
    assert adapter.default_encode_batch_size == 1
    assert torch.allclose(reps[0], embedded.mean(dim=1))
    assert torch.allclose(reps[1], layer1.mean(dim=1))
    assert torch.allclose(reps[2], layer2.mean(dim=1))

    metadata = adapter.adapter_metadata()
    assert metadata["benchmark_sequence_length"] == 4096
    assert metadata["context_length"] == 4096
    assert metadata["attention_implementation"] == "eager"
    assert metadata["hidden_size"] == 2
    assert metadata["num_hidden_layers"] == 2
    assert metadata["num_experts"] == 8
    assert metadata["num_experts_per_tok"] == 2
    assert metadata["horizon_lengths"] == [1, 8, 32, 64]
    assert metadata["parameter_count"] == 2


def test_timemoe_large_adapter_reuses_same_context_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_auto_config, fake_auto_model = _install_fake_transformers(monkeypatch)

    adapter = TimeMoeLargeAdapter()

    assert adapter.benchmark_sequence_length == 4096
    assert tuple(adapter.available_layers) == (0, 1, 2)
    assert fake_auto_config.load_calls == [("Maple728/TimeMoE-200M", {"trust_remote_code": True})]
    assert fake_auto_model.load_calls == []
