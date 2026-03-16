from __future__ import annotations

import torch

from .base import FrozenTimeSeriesAdapter


class Chronos2Adapter(FrozenTimeSeriesAdapter):
    model_name = "Chronos2"
    model_slug = "Chronos2"
    source = "https://github.com/amazon-science/chronos-forecasting"
    checkpoint = "amazon/chronos-2"
    import_path = "chronos-forecasting"
    env_name = "chronos"
    default_encode_batch_size = 64
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        from chronos import Chronos2Pipeline

        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = Chronos2Pipeline.from_pretrained(self.checkpoint, device_map=device_map)
        self.model = pipe.model
        self.model.eval()
        self.num_layers = int(len(self.model.encoder.block)) + 1
        self.benchmark_sequence_length = int(self.model.chronos_config.context_length)
        self.benchmark_sequence_length_source = "chronos_config.context_length"

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_layers))

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["context_length"] = int(self.benchmark_sequence_length)
        payload["input_patch_size"] = int(self.model.chronos_config.input_patch_size)
        payload["uses_reg_token"] = bool(self.model.chronos_config.use_reg_token)
        payload["preprocess"] = (
            "expect exact model context length; squeeze channel dimension and mean-pool non-output encoder tokens"
        )
        payload["layer_layout"] = (
            "layer 0 is the encoder input embedding stream; "
            "layers 1..N are encoder block outputs"
        )
        return payload

    def _build_encoder_inputs(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(context.shape[0])
        patched_context, attention_mask, loc_scale = self.model._prepare_patched_context(context=context)

        input_embeds: torch.Tensor = self.model.input_patch_embedding(patched_context)
        if self.model.chronos_config.use_reg_token:
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.model.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.model.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask.to(self.model.dtype), torch.ones_like(reg_input_ids).to(self.model.dtype)],
                dim=-1,
            )

        patched_future, _ = self.model._prepare_patched_future(
            future_covariates=None,
            future_covariates_mask=None,
            loc_scale=loc_scale,
            num_output_patches=1,
            batch_size=batch_size,
        )
        future_embeds = self.model.input_patch_embedding(patched_future)
        future_attention_mask = torch.ones(
            batch_size,
            1,
            dtype=self.model.dtype,
            device=input_embeds.device,
        )

        input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        attention_mask = torch.cat([attention_mask, future_attention_mask], dim=-1)
        return input_embeds, attention_mask

    def _pool_hidden(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pool_mask = attention_mask.to(dtype=hidden_states.dtype).clone()
        # The last token is the masked output patch token used for forecasting.
        pool_mask[:, -1] = 0.0
        denom = pool_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (hidden_states * pool_mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = set(layers or self.available_layers)
        self.validate_benchmark_input(x, channels=1)
        context = x[:, 0, :].to(dtype=torch.float32)
        input_embeds, attention_mask = self._build_encoder_inputs(context)
        batch_size, seq_length = input_embeds.shape[:2]
        position_ids = torch.arange(seq_length, device=input_embeds.device, dtype=torch.long).unsqueeze(0)
        group_ids = torch.arange(batch_size, dtype=torch.long, device=input_embeds.device)

        extended_attention_mask = self.model.encoder._expand_and_invert_time_attention_mask(
            attention_mask,
            input_embeds.dtype,
        )
        group_time_mask = self.model.encoder._construct_and_invert_group_time_mask(
            group_ids,
            attention_mask,
            input_embeds.dtype,
        )

        hidden_states = self.model.encoder.dropout(input_embeds)
        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            reps[0] = self._pool_hidden(hidden_states, attention_mask).float()
        for layer_index, layer_module in enumerate(self.model.encoder.block, start=1):
            layer_outputs = layer_module(
                hidden_states,
                position_ids=position_ids,
                attention_mask=extended_attention_mask,
                group_time_mask=group_time_mask,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]
            if layer_index in requested_layers:
                reps[int(layer_index)] = self._pool_hidden(hidden_states, attention_mask).float()
        return reps
