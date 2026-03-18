from __future__ import annotations

import sys
from pathlib import Path

import torch

from .base import FrozenTimeSeriesAdapter


class _BaseKairosAdapter(FrozenTimeSeriesAdapter):
    source = "https://github.com/foundation-model-research/Kairos"
    import_path = "Kairos repo + transformers"
    env_name = "core"
    default_encode_batch_size = 16
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        repo_root = Path(__file__).resolve().parents[2]
        self.kairos_root = repo_root / "external" / "Kairos"
        if not self.kairos_root.is_dir():
            raise FileNotFoundError(f"Expected Kairos repo at {self.kairos_root}")
        sys.path.insert(0, str(self.kairos_root))

        from tsfm.model.kairos import KairosModel

        self.model = KairosModel.from_pretrained(self.checkpoint)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype
        self.context_length = int(self.model.config.context_length)
        self.hidden_size = int(self.model.config.d_model)
        self.num_hidden_layers = int(self.model.config.num_layers)
        self.input_patch_size = int(self.model.config.input_patch_size)
        self.input_patch_stride = int(self.model.config.input_patch_stride)
        self.levels = int(self.model.config.levels)
        self.benchmark_sequence_length = int(self.context_length)
        self.benchmark_sequence_length_source = "kairos_config.context_length"

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_hidden_layers + 1))

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["context_length"] = int(self.context_length)
        payload["hidden_size"] = int(self.hidden_size)
        payload["input_patch_size"] = int(self.input_patch_size)
        payload["input_patch_stride"] = int(self.input_patch_stride)
        payload["levels"] = int(self.levels)
        payload["use_reg_token"] = bool(self.model.config.use_reg_token)
        payload["pooling"] = "mean over encoder tokens after attention_mask filtering"
        payload["preprocess"] = (
            "apply the official instance normalization, FFT side features, and adaptive patching pipeline; "
            "mean-pool encoder token states over valid tokens"
        )
        payload["layer_layout"] = (
            "layer 0 is the encoder input embedding stream; "
            "layers 1..N-1 are encoder block outputs; "
            "layer N is the post-final-norm encoder output"
        )
        return payload

    def _build_encoder_inputs(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate_benchmark_input(x, channels=1)
        context = x[:, 0, :].to(dtype=torch.float32)
        mask = torch.ones_like(context, dtype=torch.float32)

        context = torch.where(mask > 0.0, context, torch.nan)
        context, _ = self.model.instance_norm(context)
        s_features = self.model.fft_process(context, mask)
        context = context.to(dtype=self.model_dtype)
        mask = mask.to(dtype=self.model_dtype)

        patched_context, patched_mask, size, expert_weights, expert_indices, x_final = self.model.patch(
            context,
            mask,
        )
        patched_mask = torch.nan_to_num(patched_mask, nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        attention_mask = (patched_mask.sum(dim=-1) > 0).to(dtype=self.model_dtype)
        input_embeds = self.model.input_patch_embedding(
            patched_context,
            size,
            expert_weights,
            expert_indices,
            x_final,
        )
        if self.model.config.use_reg_token:
            reg_input_ids = torch.full(
                (input_embeds.size(0), 1),
                self.model.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.model.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones_like(reg_input_ids, dtype=self.model_dtype),
                ],
                dim=-1,
            )
        return input_embeds, attention_mask, s_features, size

    def _pool_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        weight = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        denom = weight.sum(dim=1).clamp_min(1.0)
        return ((hidden_states * weight).sum(dim=1) / denom).float()

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = set(int(layer) for layer in (layers or self.available_layers))
        input_embeds, attention_mask, s_features, size = self._build_encoder_inputs(x)
        encoder_outputs = self.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=False,
            s_features=s_features,
            size=size,
        )
        hidden_states = encoder_outputs.hidden_states
        if hidden_states is None:
            raise ValueError(f"{self.model_name} encoder did not return hidden states")
        reps: dict[int, torch.Tensor] = {}
        for layer_index in requested_layers:
            reps[int(layer_index)] = self._pool_hidden(
                hidden_states[int(layer_index)],
                attention_mask=attention_mask,
            )
        return reps


class Kairos10MAdapter(_BaseKairosAdapter):
    model_name = "Kairos-10M"
    model_slug = "Kairos-10M"
    checkpoint = "mldi-lab/Kairos_10m"


class Kairos23MAdapter(_BaseKairosAdapter):
    model_name = "Kairos-23M"
    model_slug = "Kairos-23M"
    checkpoint = "mldi-lab/Kairos_23m"


class Kairos50MAdapter(_BaseKairosAdapter):
    model_name = "Kairos-50M"
    model_slug = "Kairos-50M"
    checkpoint = "mldi-lab/Kairos_50m"
    default_encode_batch_size = 8
