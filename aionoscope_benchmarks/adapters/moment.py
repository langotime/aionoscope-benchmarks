from __future__ import annotations

import torch

from .base import FrozenTimeSeriesAdapter


class MomentAdapter(FrozenTimeSeriesAdapter):
    model_name = "MOMENT"
    model_slug = "MOMENT"
    source = "https://github.com/moment-timeseries-foundation-model/moment"
    checkpoint = "AutonLab/MOMENT-1-large"
    import_path = "momentfm"
    env_name = "moment"
    default_encode_batch_size = 16
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        from momentfm import MOMENTPipeline

        self.model = MOMENTPipeline.from_pretrained(self.checkpoint)
        self.model.eval()
        self.num_layers = int(len(self.model.encoder.block)) + 1
        self.benchmark_sequence_length = int(self.model.config.seq_len)
        self.benchmark_sequence_length_source = "moment_config.seq_len"

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_layers))

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = tuple(int(layer) for layer in (layers or self.available_layers))
        self.validate_benchmark_input(x, channels=1)
        batch_size, n_channels, seq_len = x.shape
        input_mask = torch.ones((batch_size, seq_len), device=x.device, dtype=torch.long)

        x_enc = self.model.normalizer(x=x, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0.0, posinf=0.0, neginf=0.0)
        x_enc = self.model.tokenizer(x=x_enc)
        enc_in = self.model.patch_embedding(x_enc, mask=input_mask)
        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape((batch_size * n_channels, n_patches, self.model.config.d_model))

        patch_view_mask = input_mask[:, :: self.model.patch_len]
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.model.encoder(
            inputs_embeds=enc_in,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("MOMENT encoder did not return hidden states")
        reps: dict[int, torch.Tensor] = {}
        for layer in requested_layers:
            state = hidden_states[int(layer)]
            state = state.reshape((batch_size, n_channels, n_patches, self.model.config.d_model))
            state = state.mean(dim=1)
            mask = patch_view_mask.unsqueeze(-1).to(dtype=state.dtype)
            pooled = (mask * state).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            reps[int(layer)] = pooled.float()
        return reps

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["seq_len"] = int(self.benchmark_sequence_length)
        payload["patch_len"] = int(self.model.patch_len)
        payload["preprocess"] = "expect exact MOMENT seq_len; tokenize and mean-pool valid patch tokens"
        payload["layer_layout"] = (
            "layer 0 is the encoder input embedding stream; "
            "layers 1..N are transformer block outputs"
        )
        return payload
