from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from ..constants import BENCHMARK_DEFAULT_CHANNEL_SIZE
from .base import FrozenTimeSeriesAdapter


class TLossAdapter(FrozenTimeSeriesAdapter):
    model_name = "T-Loss-CricketX"
    model_slug = "T-Loss-CricketX"
    source = "https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries"
    checkpoint = "CricketX_CausalCNN_encoder.pth"
    import_path = "TLossRepo"
    env_name = "core"
    default_encode_batch_size = 1024
    use_bfloat16_amp = False

    def __init__(self) -> None:
        super().__init__()
        repo_root = Path(__file__).resolve().parents[3]
        self.tloss_root = repo_root / "external" / "TLossRepo"
        if not self.tloss_root.is_dir():
            raise FileNotFoundError(f"Expected T-Loss repo at {self.tloss_root}")
        sys.path.insert(0, str(self.tloss_root))

        module_path = self.tloss_root / "networks" / "causal_cnn.py"
        spec = importlib.util.spec_from_file_location("tloss_causal_cnn", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load T-Loss causal_cnn module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        CausalCNNEncoder = module.CausalCNNEncoder

        hyper_path = self.tloss_root / "models" / "CricketX_hyperparameters.json"
        hyper = json.loads(hyper_path.read_text(encoding="utf-8"))
        self.hyper = hyper
        self.encoder = CausalCNNEncoder(
            in_channels=int(hyper["in_channels"]),
            channels=int(hyper["channels"]),
            depth=int(hyper["depth"]),
            reduced_size=int(hyper["reduced_size"]),
            out_channels=int(hyper["out_channels"]),
            kernel_size=int(hyper["kernel_size"]),
        )
        checkpoint_path = self.tloss_root / "models" / self.checkpoint
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval()
        self.num_causal_blocks = int(len(self.encoder.network[0].network))
        self.benchmark_sequence_length = int(BENCHMARK_DEFAULT_CHANNEL_SIZE)
        self.benchmark_sequence_length_source = "benchmark_default_channel_size"

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_causal_blocks + 1))

    def parameter_count_prefix_sources(self) -> dict[int, tuple[object, ...]]:
        blocks = self.encoder.network[0].network
        sources = {
            int(layer_index): (block,)
            for layer_index, block in enumerate(blocks)
        }
        final_layer = int(self.num_causal_blocks)
        sources[final_layer] = tuple(self.encoder.network[index] for index in range(1, 4))
        return sources

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["channels"] = int(self.hyper["channels"])
        payload["depth"] = int(self.hyper["depth"])
        payload["reduced_size"] = int(self.hyper["reduced_size"])
        payload["out_channels"] = int(self.hyper["out_channels"])
        payload["kernel_size"] = int(self.hyper["kernel_size"])
        payload["preprocess"] = "expect exact benchmark waveform; run the causal CNN encoder without temporal padding"
        payload["layer_layout"] = (
            "layers 0..N-1 are adaptive-max-pooled outputs after each causal block; "
            "layer N is the final encoder linear output"
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
        reps: dict[int, torch.Tensor] = {}
        hidden = x.to(dtype=torch.float32)
        blocks = self.encoder.network[0].network
        for layer_index, block in enumerate(blocks):
            hidden = block(hidden)
            if layer_index in requested_layers:
                reps[int(layer_index)] = F.adaptive_max_pool1d(hidden, 1).squeeze(-1).float()

        final_layer = self.num_causal_blocks
        if final_layer in requested_layers:
            pooled = self.encoder.network[1](hidden)
            squeezed = self.encoder.network[2](pooled)
            reps[int(final_layer)] = self.encoder.network[3](squeezed).float()
        return reps
