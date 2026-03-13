from __future__ import annotations

import torch

from .base import FrozenTimeSeriesAdapter
from .tivit_common import sqrt_patch_size_for_length, timeseries_to_clip_images


class TiConvNextAdapter(FrozenTimeSeriesAdapter):
    model_name = "TiConvNext"
    model_slug = "TiConvNext"
    source = "https://github.com/ExplainableML/TiViT"
    checkpoint = "laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg"
    import_path = "open_clip_torch"
    env_name = "tivit"
    default_encode_batch_size = 8
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        import open_clip

        self.model = open_clip.create_model(
            "convnext_xxlarge",
            pretrained="laion2b_s34b_b82k_augreg",
            device="cpu",
        )
        self.model.eval()
        self.visual = self.model.visual
        if isinstance(self.visual.image_size, tuple):
            self.image_size = int(self.visual.image_size[0])
        else:
            self.image_size = int(self.visual.image_size)
        self.stage_depths = tuple(int(len(stage.blocks)) for stage in self.visual.trunk.stages)
        self.num_layers = int(sum(self.stage_depths))
        self.stride_fraction = 0.1

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_layers))

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["patch_size_mode"] = "sqrt"
        payload["stride_fraction"] = float(self.stride_fraction)
        payload["image_size"] = int(self.image_size)
        payload["stage_depths"] = list(self.stage_depths)
        payload["preprocess"] = (
            "TiViT ts2image transform with robust scaling, sqrt patch size, stride 0.1, "
            "and CLIP mean/std normalization"
        )
        return payload

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = set(int(layer) for layer in (layers or self.available_layers))
        patch_size = sqrt_patch_size_for_length(int(x.size(-1)))
        images, num_channels = timeseries_to_clip_images(
            x,
            patch_size=patch_size,
            stride_fraction=self.stride_fraction,
            image_size=self.image_size,
        )

        batch_size = int(x.size(0))
        hidden = self.visual.trunk.stem(images)
        reps: dict[int, torch.Tensor] = {}
        layer_index = 0
        for stage in self.visual.trunk.stages:
            hidden = stage.downsample(hidden)
            for block in stage.blocks:
                hidden = block(hidden)
                if layer_index in requested_layers:
                    state = hidden
                    if num_channels > 1:
                        state = state.reshape(batch_size, num_channels, *state.shape[1:]).mean(dim=1)
                    reps[int(layer_index)] = state.mean(dim=(2, 3)).float()
                layer_index += 1
        return reps
