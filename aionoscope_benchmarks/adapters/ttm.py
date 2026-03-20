from __future__ import annotations

import torch

from .base import FrozenTimeSeriesAdapter


class TTMAdapter(FrozenTimeSeriesAdapter):
    model_name = "TTM-r2"
    model_slug = "TTM-r2"
    source = "https://github.com/ibm-granite/granite-tsfm"
    checkpoint = "ibm-granite/granite-timeseries-ttm-r2"
    import_path = "tsfm_public"
    env_name = "ttm"
    default_encode_batch_size = 512
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        from tsfm_public import TinyTimeMixerForPrediction

        self.model = TinyTimeMixerForPrediction.from_pretrained(self.checkpoint)
        self.model.eval()
        self.sequence_length = int(self.model.backbone.patching.sequence_length)
        self.benchmark_sequence_length = int(self.sequence_length)
        self.benchmark_sequence_length_source = "model.backbone.patching.sequence_length"
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

    def parameter_count_prefix_sources(self) -> dict[int, tuple[object, ...]]:
        encoder = self.model.backbone.encoder
        layer_zero_sources: tuple[object, ...] = (encoder.patcher,)
        if hasattr(encoder, "freq_mod"):
            layer_zero_sources = layer_zero_sources + (encoder.freq_mod,)
        if getattr(encoder, "positional_encoder", None) is not None:
            layer_zero_sources = layer_zero_sources + (encoder.positional_encoder,)

        sources: dict[int, tuple[object, ...]] = {}
        layer_index = 0
        first_mixer = True
        for mixer in encoder.mlp_mixer_encoder.mixers:
            if first_mixer:
                sources[int(layer_index)] = layer_zero_sources
                first_mixer = False
            else:
                sources[int(layer_index)] = ()
            layer_index += 1
            if hasattr(mixer, "mixer_layers"):
                sources[int(layer_index)] = ()
                layer_index += 1
                for mixer_layer in mixer.mixer_layers:
                    sources[int(layer_index)] = (mixer_layer,)
                    layer_index += 1
                sources[int(layer_index)] = ()
                layer_index += 1
                continue
            sources[int(layer_index)] = (mixer,)
            layer_index += 1
        return sources

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["context_length"] = int(self.sequence_length)
        payload["patch_length"] = int(self.patch_length)
        payload["patch_stride"] = int(self.patch_stride)
        payload["resolution_prefix_tuning"] = bool(self.resolution_prefix_tuning)
        payload["preprocess"] = (
            "expect exact model context length; transpose to [B,T,C] and build the observed-mask directly"
        )
        return payload

    def _prepare_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate_benchmark_input(x, channels=1)
        past_values = x.transpose(1, 2).to(dtype=torch.float32)
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
