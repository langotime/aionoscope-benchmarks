from __future__ import annotations

import torch

from .base import FrozenTimeSeriesAdapter


class MantisV2Adapter(FrozenTimeSeriesAdapter):
    model_name = "MantisV2"
    model_slug = "MantisV2"
    source = "https://github.com/vfeofanov/mantis"
    checkpoint = "paris-noah/MantisV2"
    import_path = "mantis-tsfm"
    env_name = "mantis"
    default_encode_batch_size = 128
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        from mantis.architecture import MantisV2

        self.model = MantisV2(
            return_transf_layer=-1,
            output_token="combined",
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).from_pretrained(self.checkpoint)
        self.model.eval()
        self.num_layers = int(len(self.model.transf_unit.transformer.layers)) + 1

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_layers))

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["output_token"] = "combined"
        payload["preprocess"] = "right-pad sequence length to nearest multiple of num_patches"
        payload["layer_layout"] = (
            "layer 0 is the tokenizer-plus-CLS embedding stream; "
            "layers 1..N are transformer block outputs"
        )
        return payload

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = set(layers or self.available_layers)
        seq_len = x.shape[2]
        remainder = seq_len % self.model.num_patches
        if remainder != 0:
            pad = self.model.num_patches - remainder
            x = torch.nn.functional.pad(x, (0, pad))

        x_embeddings = self.model.tokgen_unit(x)
        batch_size = int(x_embeddings.shape[0])
        cls_tokens = self.model.transf_unit.cls_token.unsqueeze(0).unsqueeze(0).expand(
            batch_size, 1, -1
        )
        hidden = torch.cat([x_embeddings, cls_tokens], dim=1)
        hidden = hidden[:, None, :, :]

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            layer_tokens = hidden[:, 0, :, :]
            cls_token = layer_tokens[:, -1, :]
            mean_token = layer_tokens[:, :-1, :].mean(dim=1)
            reps[0] = torch.cat([cls_token, mean_token], dim=1).float()
        for layer_index, layer in enumerate(self.model.transf_unit.transformer.layers, start=1):
            hidden = layer(hidden)
            if layer_index in requested_layers:
                layer_tokens = hidden[:, 0, :, :]
                cls_token = layer_tokens[:, -1, :]
                mean_token = layer_tokens[:, :-1, :].mean(dim=1)
                reps[int(layer_index)] = torch.cat([cls_token, mean_token], dim=1).float()
        return reps
