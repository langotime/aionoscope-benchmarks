from __future__ import annotations

import torch

from .base import FrozenTimeSeriesAdapter


class MoiraiAdapter(FrozenTimeSeriesAdapter):
    model_name = "Moirai"
    model_slug = "Moirai"
    source = "https://github.com/SalesforceAIResearch/uni2ts"
    checkpoint = "Salesforce/moirai-1.1-R-small"
    import_path = "uni2ts"
    env_name = "moirai"
    default_encode_batch_size = 64
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        from uni2ts.model.moirai import MoiraiModule
        from uni2ts.model.moirai.forecast import MoiraiForecast

        self.model = MoiraiModule.from_pretrained(self.checkpoint)
        self.model.eval()
        self.context_length = int(self.model.max_seq_len)
        self.patch_size = int(min(self.model.patch_sizes))
        self.helper = MoiraiForecast(
            prediction_length=1,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=self.context_length,
            module=self.model,
            patch_size=self.patch_size,
            num_samples=1,
        )

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(int(self.model.num_layers) + 1))

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["context_length"] = int(self.context_length)
        payload["patch_size"] = int(self.patch_size)
        payload["patch_sizes"] = [int(size) for size in self.model.patch_sizes]
        payload["preprocess"] = (
            "transpose to [B,T,C]; crop to last max_seq_len or left-pad with zeros; "
            "pack tokens with MoiraiForecast._convert; mean-pool observed non-prediction tokens"
        )
        payload["layer_layout"] = (
            "layer 0 is the encoder input embedding stream; "
            "layers 1..N are transformer block outputs"
        )
        return payload

    def _prepare_context(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_target = x.transpose(1, 2).to(dtype=torch.float32)
        batch_size, seq_len, channels = past_target.shape
        if channels != 1:
            raise ValueError(f"Moirai adapter currently expects 1 channel, got {channels}")
        if seq_len > self.context_length:
            past_target = past_target[:, -self.context_length :, :]
            past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
            past_is_pad = torch.zeros(batch_size, self.context_length, dtype=torch.bool, device=past_target.device)
        elif seq_len < self.context_length:
            pad_len = self.context_length - seq_len
            pad = torch.zeros(batch_size, pad_len, channels, dtype=past_target.dtype, device=past_target.device)
            pad_mask = torch.zeros(batch_size, pad_len, channels, dtype=torch.bool, device=past_target.device)
            pad_is_pad = torch.ones(batch_size, pad_len, dtype=torch.bool, device=past_target.device)
            observed = torch.ones_like(past_target, dtype=torch.bool)
            past_target = torch.cat([pad, past_target], dim=1)
            past_observed_target = torch.cat([pad_mask, observed], dim=1)
            past_is_pad = torch.cat(
                [
                    pad_is_pad,
                    torch.zeros(batch_size, seq_len, dtype=torch.bool, device=past_target.device),
                ],
                dim=1,
            )
        else:
            past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
            past_is_pad = torch.zeros(batch_size, self.context_length, dtype=torch.bool, device=past_target.device)
        return past_target, past_observed_target, past_is_pad

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        from uni2ts.common.torch_util import mask_fill, packed_attention_mask

        requested_layers = set(layers or self.available_layers)
        past_target, past_observed_target, past_is_pad = self._prepare_context(x)
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self.helper._convert(
            self.patch_size,
            past_target,
            past_observed_target,
            past_is_pad,
        )
        patch_size = torch.full_like(time_id, fill_value=self.patch_size)
        loc, scale = self.model.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale
        hidden_states = self.model.in_proj(scaled_target, patch_size)
        hidden_states = mask_fill(hidden_states, prediction_mask, self.model.mask_encoding.weight)
        attention_mask = packed_attention_mask(sample_id)
        pool_mask = observed_mask.any(dim=-1) & ~prediction_mask

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            weight = pool_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
            denom = weight.sum(dim=1).clamp_min(1.0)
            reps[0] = ((hidden_states * weight).sum(dim=1) / denom).float()
        for layer_index, layer in enumerate(self.model.encoder.layers, start=1):
            hidden_states = layer(
                hidden_states,
                attention_mask,
                var_id=variate_id,
                time_id=time_id,
            )
            if layer_index in requested_layers:
                weight = pool_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
                denom = weight.sum(dim=1).clamp_min(1.0)
                reps[int(layer_index)] = (hidden_states * weight).sum(dim=1) / denom
                reps[int(layer_index)] = reps[int(layer_index)].float()
        return reps
