from __future__ import annotations

import torch

from ..constants import BENCHMARK_DEFAULT_CHANNEL_SIZE
from .base import FrozenTimeSeriesAdapter
from .tivit_common import sqrt_patch_size_for_length, timeseries_to_clip_images


class TiViTHAdapter(FrozenTimeSeriesAdapter):
    model_name = "TiViT-H"
    model_slug = "TiViT-H"
    source = "https://github.com/ExplainableML/TiViT"
    checkpoint = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    import_path = "transformers"
    env_name = "tivit"
    default_encode_batch_size = 16
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        from transformers import CLIPVisionModel

        self.model = CLIPVisionModel.from_pretrained(self.checkpoint)
        self.model.eval()
        self.num_layers = int(len(self.model.vision_model.encoder.layers)) + 1
        self.image_size = int(self.model.config.image_size)
        self.stride_fraction = 0.1
        self.aggregation = "mean"
        self.benchmark_sequence_length = int(BENCHMARK_DEFAULT_CHANNEL_SIZE)
        self.benchmark_sequence_length_source = "benchmark_default_channel_size"

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_layers))

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["aggregation"] = self.aggregation
        payload["patch_size_mode"] = "sqrt"
        payload["stride_fraction"] = float(self.stride_fraction)
        payload["image_size"] = int(self.image_size)
        payload["preprocess"] = (
            "TiViT ts2image transform on the exact benchmark waveform with robust scaling, sqrt patch size, stride 0.1, "
            "and CLIP mean/std normalization"
        )
        payload["layer_layout"] = (
            "layer 0 is the CLIP vision embedding stream; "
            "layers 1..N are transformer block outputs"
        )
        return payload

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = tuple(int(layer) for layer in (layers or self.available_layers))
        self.validate_benchmark_input(x)
        patch_size = sqrt_patch_size_for_length(int(x.size(-1)))
        images, num_channels = timeseries_to_clip_images(
            x,
            patch_size=patch_size,
            stride_fraction=self.stride_fraction,
            image_size=self.image_size,
        )
        outputs = self.model(
            pixel_values=images,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("TiViT-H CLIP model did not return hidden states")

        batch_size = int(x.size(0))
        reps: dict[int, torch.Tensor] = {}
        for layer in requested_layers:
            state = hidden_states[int(layer)]
            if num_channels > 1:
                state = state.reshape(batch_size, num_channels, *state.shape[1:]).mean(dim=1)
            reps[int(layer)] = state.mean(dim=1).float()
        return reps
