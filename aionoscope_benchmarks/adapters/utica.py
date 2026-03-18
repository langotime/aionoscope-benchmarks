from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download

from .base import FrozenTimeSeriesAdapter


class MantisUTICA8MAdapter(FrozenTimeSeriesAdapter):
    model_name = "Mantis-UTICA-8M"
    model_slug = "Mantis-UTICA-8M"
    source = "https://github.com/fegounna/Utica"
    checkpoint = "fegounna/Utica"
    import_path = "mantis-tsfm + huggingface_hub"
    env_name = "mantis"
    default_encode_batch_size = 128
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        from mantis.architecture import Mantis8M

        init_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Mantis8M(device=init_device)
        checkpoint_path = hf_hub_download(repo_id=self.checkpoint, filename="pytorch_model.bin")
        state_dict = torch.load(checkpoint_path, map_location=init_device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.num_layers = int(len(self.model.vit_unit.transformer.layers)) + 1
        self.benchmark_sequence_length = 512
        self.benchmark_sequence_length_source = "official_utica_readme_resize_example"

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_layers))

    def parameter_count_prefix_sources(self) -> dict[int, tuple[object, ...]]:
        sources: dict[int, tuple[object, ...]] = {
            0: (
                self.model.tokgen_unit,
                self.model.vit_unit.cls_token,
                self.model.vit_unit.pos_encoder,
            )
        }
        for layer_index, layer in enumerate(self.model.vit_unit.transformer.layers, start=1):
            sources[int(layer_index)] = tuple(layer)
        return sources

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["num_patches"] = int(self.model.num_patches)
        payload["representation_token"] = "cls_token"
        payload["preprocess"] = (
            "expect the official UTICA resize target of 512 samples; "
            "read the CLS token from the reused Mantis-8M backbone"
        )
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
        requested_layers = set(int(layer) for layer in (layers or self.available_layers))
        self.validate_benchmark_input(x, channels=1)

        hidden_states = self.model.tokgen_unit(x)
        batch_size = int(hidden_states.shape[0])
        cls_tokens = self.model.vit_unit.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)
        hidden_states = self.model.vit_unit.pos_encoder(hidden_states.transpose(0, 1)).transpose(0, 1)

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            reps[0] = hidden_states[:, 0, :].float()
        for layer_index, (attn, ff) in enumerate(self.model.vit_unit.transformer.layers, start=1):
            hidden_states = attn(hidden_states) + hidden_states
            hidden_states = ff(hidden_states) + hidden_states
            if layer_index in requested_layers:
                reps[int(layer_index)] = hidden_states[:, 0, :].float()
        return reps
