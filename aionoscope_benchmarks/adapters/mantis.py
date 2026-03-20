from __future__ import annotations

import torch

from .base import FrozenTimeSeriesAdapter


class _BaseMantisV1Adapter(FrozenTimeSeriesAdapter):
    source = "https://github.com/vfeofanov/mantis"
    import_path = "mantis-tsfm"
    env_name = "mantis"
    default_encode_batch_size = 512
    use_bfloat16_amp = True
    backbone_class_name = "MantisV1"

    def __init__(self) -> None:
        super().__init__()
        from mantis.architecture import MantisV1

        # Load on CPU first so adapter construction stays device-agnostic; the
        # benchmark runner moves the full adapter onto the requested runtime
        # device immediately after instantiation.
        self.model = MantisV1(
            return_transf_layer=-1,
            output_token="combined",
            device="cpu",
        ).from_pretrained(self.checkpoint)
        self.model.eval()
        self.num_layers = int(len(self.model.transf_unit.transformer.layers)) + 1
        self.benchmark_sequence_length = 512
        self.benchmark_sequence_length_source = "official_mantis_recommended_pretrained_length"

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_layers))

    def parameter_count_prefix_sources(self) -> dict[int, tuple[object, ...]]:
        return {
            0: (
                self.model.tokgen_unit,
                self.model.transf_unit.cls_token,
                self.model.transf_unit.pos_encoder,
            ),
            **{
                int(layer_index): (layer,)
                for layer_index, layer in enumerate(self.model.transf_unit.transformer.layers, start=1)
            },
        }

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["architecture_module"] = self.backbone_class_name
        payload["output_token"] = "combined"
        payload["num_patches"] = int(self.model.num_patches)
        payload["preprocess"] = (
            "expect exact benchmark length 512, matching official pretrained guidance; "
            "no benchmark-side padding"
        )
        payload["layer_layout"] = (
            "layer 0 is the tokenizer-plus-CLS embedding stream after positional encoding; "
            "layers 1..N are transformer block outputs"
        )
        return payload

    def _pooled_representation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls_token = hidden_states[:, 0, :]
        mean_token = hidden_states[:, 1:, :].mean(dim=1)
        return torch.cat([cls_token, mean_token], dim=1).float()

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
        cls_tokens = self.model.transf_unit.cls_token.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)
        hidden_states = self.model.transf_unit.pos_encoder(hidden_states.transpose(0, 1)).transpose(0, 1)

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            reps[0] = self._pooled_representation(hidden_states)

        for layer_index, (attn, ff) in enumerate(self.model.transf_unit.transformer.layers, start=1):
            hidden_states = attn(hidden_states) + hidden_states
            hidden_states = ff(hidden_states) + hidden_states
            if layer_index in requested_layers:
                reps[int(layer_index)] = self._pooled_representation(hidden_states)
        return reps


class Mantis8MAdapter(_BaseMantisV1Adapter):
    model_name = "Mantis-8M"
    model_slug = "Mantis-8M"
    checkpoint = "paris-noah/Mantis-8M"


class MantisPlusAdapter(_BaseMantisV1Adapter):
    model_name = "MantisPlus"
    model_slug = "MantisPlus"
    checkpoint = "paris-noah/MantisPlus"
