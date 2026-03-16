from __future__ import annotations

import torch

from .base import FrozenTimeSeriesAdapter


class TiRexAdapter(FrozenTimeSeriesAdapter):
    model_name = "TiRex"
    model_slug = "TiRex"
    source = "https://github.com/NX-AI/tirex"
    checkpoint = "NX-AI/TiRex"
    import_path = "tirex-ts"
    env_name = "tirex"
    default_encode_batch_size = 128
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        from tirex import load_model

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = load_model(self.checkpoint, device=device, backend="torch", compile=False)
        self.model.eval()
        self.num_layers = int(len(self.model.blocks))
        self.benchmark_sequence_length = int(self.model.config.train_ctx_len)
        self.benchmark_sequence_length_source = "model.config.train_ctx_len"

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_layers))

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["patch_size"] = int(self.model.config.input_patch_size)
        payload["train_context_length"] = int(self.model.config.train_ctx_len)
        payload["preprocess"] = (
            "expect exact training context length; squeeze channel dimension; "
            "token-mean pool per sLSTM block"
        )
        return payload

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = set(layers or self.available_layers)
        self.validate_benchmark_input(x, channels=1)
        context = x[:, 0, :].to(dtype=torch.float32)
        hidden_states = self.model._embed_context(context, max_context=int(context.shape[-1]))
        if hidden_states.dim() != 4:
            raise ValueError(
                "Expected TiRex hidden states to be [B,T,L,D], "
                f"got {tuple(hidden_states.shape)}"
            )

        reps: dict[int, torch.Tensor] = {}
        for layer_index in requested_layers:
            reps[int(layer_index)] = hidden_states[:, :, int(layer_index), :].mean(dim=1).float()
        return reps
