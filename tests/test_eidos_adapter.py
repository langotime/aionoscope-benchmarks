from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

import aionoscope_benchmarks.adapters.eidos as eidos_module
from aionoscope_benchmarks.adapters.eidos import EIDOSAdapter


class _FakeEidosConfig:
    def __init__(self, *, _attn_implementation: str = "eager") -> None:
        self.hidden_size = 2
        self.intermediate_size = 4
        self.horizon_lengths = [64]
        self.num_local_decoder_layers = 2
        self.num_attention_heads = 1
        self.num_key_value_heads = 1
        self.max_position_embeddings = 4096
        self.quantiles = [0.1, 0.5, 0.9]
        self._attn_implementation = str(_attn_implementation)


class _FakeFinalNorm(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 100.0


class _FakeDecoderLayer(torch.nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = torch.nn.Parameter(torch.tensor(float(delta), dtype=torch.float32))

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


class _FakeTransformerBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([
            _FakeDecoderLayer(10.0),
            _FakeDecoderLayer(20.0),
        ])

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        del past_key_values, use_cache, output_attentions
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return types.SimpleNamespace(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class _FakeDecoderModel(torch.nn.Module):
    def __init__(self, config: _FakeEidosConfig) -> None:
        super().__init__()
        del config
        self.embed_layer = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.embed_layer.weight.copy_(torch.tensor([[1.0], [2.0]], dtype=torch.float32))
        self.decoder = _FakeTransformerBlock()
        self.norm = _FakeFinalNorm()


class _FakeEidosForPrediction(torch.nn.Module):
    def __init__(self, config: _FakeEidosConfig) -> None:
        super().__init__()
        self.config = config
        self.model = _FakeDecoderModel(config)

    def get_decoder(self) -> torch.nn.Module:
        return self.model

    def eval(self):
        super().eval()
        return self


def _install_fake_eidos_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    eidos_root = tmp_path / "external" / "EIDOS"
    eidos_root.mkdir(parents=True)

    config_module = types.ModuleType("configuration_eidos")
    config_module.EidosConfig = _FakeEidosConfig

    modeling_module = types.ModuleType("modeling_eidos")
    modeling_module.EidosForPrediction = _FakeEidosForPrediction

    transformers_module = types.ModuleType("transformers")
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
    transformers_module.modeling_attn_mask_utils = modeling_attn_mask_utils

    monkeypatch.setitem(sys.modules, "configuration_eidos", config_module)
    monkeypatch.setitem(sys.modules, "modeling_eidos", modeling_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "transformers.modeling_attn_mask_utils", modeling_attn_mask_utils)
    monkeypatch.setattr(eidos_module, "REPO_ROOT", tmp_path)

    fake_model = _FakeEidosForPrediction(_FakeEidosConfig())
    prefixed_state_dict = {
        f"_orig_mod.model.{key}": value.clone()
        for key, value in fake_model.state_dict().items()
    }
    torch.save(
        {"model_state_dict": prefixed_state_dict},
        eidos_root / "eidos 1.pt",
    )
    return eidos_root


def test_eidos_adapter_uses_local_repo_hosted_checkpoint_and_readme_context_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    eidos_root = _install_fake_eidos_runtime(monkeypatch, tmp_path)

    adapter = EIDOSAdapter()
    x = torch.arange(512, dtype=torch.float32).reshape(1, 1, 512)
    reps = adapter.forward_layer_dict(x, layers=(0, 1, 2))

    normalized = x[:, 0, :]
    normalized = (normalized - normalized.mean(dim=-1, keepdim=True)) / normalized.std(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-5)
    embedded = torch.cat([normalized.unsqueeze(-1), (2.0 * normalized).unsqueeze(-1)], dim=-1)
    layer1 = embedded + 10.0
    layer2 = layer1 + 20.0 + 100.0

    assert adapter.eidos_root == eidos_root
    assert adapter.checkpoint_path == eidos_root / "eidos 1.pt"
    assert adapter.benchmark_sequence_length == 512
    assert adapter.benchmark_sequence_length_source == "official_eidos_readme_basic_usage_history_length"
    assert tuple(adapter.available_layers) == (0, 1, 2)
    assert adapter.attention_implementation == "eager"
    assert torch.allclose(reps[0], embedded.mean(dim=1))
    assert torch.allclose(reps[1], layer1.mean(dim=1))
    assert torch.allclose(reps[2], layer2.mean(dim=1))

    metadata = adapter.adapter_metadata()
    assert metadata["benchmark_sequence_length"] == 512
    assert metadata["context_length"] == 512
    assert metadata["max_position_embeddings"] == 4096
    assert metadata["hidden_size"] == 2
    assert metadata["intermediate_size"] == 4
    assert metadata["num_hidden_layers"] == 2
    assert metadata["num_attention_heads"] == 1
    assert metadata["num_key_value_heads"] == 1
    assert metadata["horizon_lengths"] == [64]
    assert metadata["quantiles"] == [0.1, 0.5, 0.9]
    assert metadata["attention_implementation"] == "eager"
    assert metadata["checkpoint_origin"] == "local_repo_hosted_checkpoint"
    assert metadata["checkpoint_file"] == "external/EIDOS/eidos 1.pt"
    assert metadata["parameter_count"] == 4
    assert metadata["parameter_count_prefix_by_layer"] == {"0": 2, "1": 3, "2": 4}
    assert metadata["notes"] is not None
    assert ".venv-timemoe" in str(metadata["notes"])


def test_eidos_adapter_rejects_unknown_layers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_eidos_runtime(monkeypatch, tmp_path)
    adapter = EIDOSAdapter()
    x = torch.zeros(1, 1, 512)

    with pytest.raises(ValueError, match="invalid layers"):
        adapter.forward_layer_dict(x, layers=(3,))
