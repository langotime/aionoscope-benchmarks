from __future__ import annotations

import torch

from .base import FrozenTimeSeriesAdapter


class TotoAdapter(FrozenTimeSeriesAdapter):
    model_name = "Toto"
    model_slug = "Toto"
    source = "https://github.com/DataDog/toto"
    checkpoint = "Datadog/Toto-Open-Base-1.0"
    import_path = "toto-ts"
    env_name = "toto"
    default_encode_batch_size = 64
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        from toto.model.toto import Toto

        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded = Toto.from_pretrained(self.checkpoint, map_location=device)
        self.model = loaded.model
        self.model.eval()

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(int(self.model.num_layers) + 1))

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["patch_size"] = int(self.model.patch_embed.patch_size)
        payload["patch_stride"] = int(self.model.patch_embed.stride)
        payload["embed_dim"] = int(self.model.embed_dim)
        payload["preprocess"] = (
            "squeeze channel dimension; left-pad to a multiple of patch stride; "
            "use Toto patch embedding and transformer; mean-pool valid patch tokens"
        )
        payload["layer_layout"] = (
            "layer 0 is the patch embedding stream; "
            "layers 1..N are transformer block outputs"
        )
        return payload

    def _prepare_inputs(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from toto.data.util.dataset import pad_array, pad_id_mask, replace_extreme_values

        series = replace_extreme_values(x.to(dtype=torch.float32))
        padding_mask = torch.ones_like(series, dtype=torch.bool)
        id_mask = torch.zeros_like(series, dtype=torch.long)
        patch_stride = int(self.model.patch_embed.stride)
        series = pad_array(series, patch_stride)
        padding_mask = pad_array(padding_mask, patch_stride).to(dtype=torch.bool)
        id_mask = pad_id_mask(id_mask, patch_stride)
        return series, padding_mask, id_mask

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        from toto.model.attention import AttentionAxis

        requested_layers = set(layers or self.available_layers)
        series, padding_mask, id_mask = self._prepare_inputs(x)
        weights = torch.ones_like(series, dtype=series.dtype, device=series.device)
        scaled_inputs, _, _ = self.model.scaler(
            series,
            weights=weights,
            padding_mask=padding_mask,
            prefix_length=None,
        )
        hidden_states, reduced_id_mask = self.model.patch_embed(scaled_inputs, id_mask)

        patch_valid = padding_mask.unfold(
            dimension=-1,
            size=self.model.patch_embed.patch_size,
            step=self.model.patch_embed.stride,
        ).any(dim=-1)
        num_heads = int(self.model.transformer.layers[0].num_heads)
        spacewise_attention_mask = self.model.transformer._get_mask(
            num_heads=num_heads,
            dtype=hidden_states.dtype,
            id_mask=reduced_id_mask,
        )

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            weight = patch_valid.unsqueeze(-1).to(dtype=hidden_states.dtype)
            denom = weight.sum(dim=(1, 2)).clamp_min(1.0)
            reps[0] = ((hidden_states * weight).sum(dim=(1, 2)) / denom).float()
        for block_index, layer in enumerate(self.model.transformer.layers):
            hidden_states = layer(
                block_index,
                hidden_states,
                None if layer.attention_axis == AttentionAxis.TIME else spacewise_attention_mask,
                None,
            )
            layer_index = block_index + 1
            if layer_index in requested_layers:
                weight = patch_valid.unsqueeze(-1).to(dtype=hidden_states.dtype)
                denom = weight.sum(dim=(1, 2)).clamp_min(1.0)
                reps[int(layer_index)] = ((hidden_states * weight).sum(dim=(1, 2)) / denom).float()
        return reps
