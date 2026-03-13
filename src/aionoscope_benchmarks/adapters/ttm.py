from __future__ import annotations

import torch

from .base import FrozenTimeSeriesAdapter


class TTMAdapter(FrozenTimeSeriesAdapter):
    model_name = "TTM"
    model_slug = "TTM"
    source = "https://github.com/ibm-granite/granite-tsfm"
    checkpoint = "ibm-granite/granite-timeseries-ttm-r2"
    import_path = "tsfm_public"
    env_name = "ttm"
    default_encode_batch_size = 128
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        from tsfm_public import TinyTimeMixerForPrediction

        self.model = TinyTimeMixerForPrediction.from_pretrained(self.checkpoint)
        self.model.eval()
        self.sequence_length = int(self.model.backbone.patching.sequence_length)
        self.patch_length = int(self.model.config.patch_length)
        self.patch_stride = int(self.model.config.patch_stride)
        self.resolution_prefix_tuning = bool(getattr(self.model.config, "resolution_prefix_tuning", False))

        dummy = torch.zeros(1, self.sequence_length, 1)
        extra = {}
        if self.resolution_prefix_tuning:
            extra["freq_token"] = torch.zeros(1, dtype=torch.long)
        with torch.inference_mode():
            output = self.model.backbone(
                past_values=dummy,
                output_hidden_states=True,
                return_dict=True,
                **extra,
            )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise ValueError("TTM backbone did not expose hidden states")
        self.num_layers = int(len(hidden_states))

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_layers))

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["context_length"] = int(self.sequence_length)
        payload["patch_length"] = int(self.patch_length)
        payload["patch_stride"] = int(self.patch_stride)
        payload["resolution_prefix_tuning"] = bool(self.resolution_prefix_tuning)
        payload["preprocess"] = (
            "transpose to [B,T,C]; crop to last context window or left-pad with zeros and observed-mask"
        )
        return payload

    def _prepare_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        past_values = x.transpose(1, 2).to(dtype=torch.float32)
        batch_size, seq_len, _ = past_values.shape
        if seq_len > self.sequence_length:
            past_values = past_values[:, -self.sequence_length :, :]
            observed_mask = torch.ones_like(past_values)
        elif seq_len < self.sequence_length:
            pad_len = self.sequence_length - seq_len
            pad = torch.zeros(batch_size, pad_len, past_values.shape[-1], device=past_values.device)
            mask_pad = torch.zeros_like(pad)
            observed = torch.ones_like(past_values)
            past_values = torch.cat([pad, past_values], dim=1)
            observed_mask = torch.cat([mask_pad, observed], dim=1)
        else:
            observed_mask = torch.ones_like(past_values)
        return past_values, observed_mask

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = set(layers or self.available_layers)
        past_values, observed_mask = self._prepare_inputs(x)
        extra = {}
        if self.resolution_prefix_tuning:
            extra["freq_token"] = torch.zeros(
                past_values.shape[0],
                dtype=torch.long,
                device=past_values.device,
            )
        outputs = self.model.backbone(
            past_values=past_values,
            past_observed_mask=observed_mask,
            output_hidden_states=True,
            return_dict=True,
            **extra,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("TTM backbone did not return hidden states")

        reps: dict[int, torch.Tensor] = {}
        for layer_index in requested_layers:
            state = hidden_states[int(layer_index)]
            if state.dim() != 4:
                raise ValueError(
                    f"Expected TTM hidden state [B,C,P,D], got {tuple(state.shape)}"
                )
            reps[int(layer_index)] = state.mean(dim=(1, 2)).float()
        return reps
