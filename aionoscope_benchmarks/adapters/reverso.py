from __future__ import annotations

import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from .base import FrozenTimeSeriesAdapter


class ReversoSmall550KAdapter(FrozenTimeSeriesAdapter):
    model_name = "Reverso-Small-550K"
    model_slug = "Reverso-Small-550K"
    source = "https://github.com/shinfxh/reverso"
    checkpoint = "shinfxh/reverso"
    import_path = "reverso_torch + huggingface_hub"
    env_name = "core"
    default_encode_batch_size = 512
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        repo_root = Path(__file__).resolve().parents[2]
        self.reverso_root = repo_root / "external" / "Reverso"
        if not self.reverso_root.is_dir():
            raise FileNotFoundError(f"Expected Reverso repo at {self.reverso_root}")
        sys.path.insert(0, str(self.reverso_root))

        from reverso_torch.forecast import load_model

        checkpoint_path = hf_hub_download(
            repo_id=self.checkpoint,
            filename="checkpoints/reverso_small/checkpoint.pth",
        )
        args_path = hf_hub_download(
            repo_id=self.checkpoint,
            filename="checkpoints/reverso_small/args.json",
        )
        self.model, self.config = load_model(
            checkpoint_path=checkpoint_path,
            args_json=args_path,
            device="cpu",
        )
        self.model.eval()
        self.num_hidden_layers = int(len(self.model.layers))
        self.benchmark_sequence_length = int(self.config.seq_len)
        self.benchmark_sequence_length_source = "official_reverso_small_args.seq_len"

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_hidden_layers + 1))

    def parameter_count_prefix_sources(self) -> dict[int, tuple[object, ...]]:
        return {
            0: (self.model.embedding,),
            **{
                int(layer_index): (layer,)
                for layer_index, layer in enumerate(self.model.layers, start=1)
            },
        }

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["seq_len"] = int(self.config.seq_len)
        payload["input_token_len"] = int(self.config.input_token_len)
        payload["output_token_len"] = int(self.config.output_token_len)
        payload["hidden_size"] = int(self.config.d_model)
        payload["main_module"] = str(self.config.main_module)
        payload["pooling"] = "mean over time after each published hybrid block"
        payload["preprocess"] = (
            "transpose to [B,L,1]; apply the official min-max normalization path when enabled; "
            "mean-pool sequence states across time"
        )
        payload["layer_layout"] = (
            "layer 0 is the input embedding stream; "
            "layers 1..N are the published sequential hybrid blocks from reverso_torch"
        )
        return payload

    def _normalize_input(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(self.model.use_norm):
            return x
        x_min = x.min(dim=1, keepdim=True).values.detach()
        x_max = x.max(dim=1, keepdim=True).values.detach()
        x_range = torch.clamp(x_max - x_min, min=1e-5).detach()
        return (x - x_min) / x_range

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = set(int(layer) for layer in (layers or self.available_layers))
        self.validate_benchmark_input(x, channels=1)
        hidden_states = x.transpose(1, 2).to(dtype=torch.float32)
        hidden_states = self._normalize_input(hidden_states)
        hidden_states = self.model.embedding(hidden_states).transpose(1, 2)

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            reps[0] = hidden_states.transpose(1, 2).mean(dim=1).float()
        for layer_index, layer in enumerate(self.model.layers, start=1):
            hidden_states = layer(hidden_states)
            if layer_index in requested_layers:
                reps[int(layer_index)] = hidden_states.transpose(1, 2).mean(dim=1).float()
        return reps
