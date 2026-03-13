from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .base import FrozenTimeSeriesAdapter


class NuTimeAdapter(FrozenTimeSeriesAdapter):
    model_name = "NuTime"
    model_slug = "NuTime"
    source = "https://github.com/chenguolin/NuTime"
    checkpoint = "checkpoint_bias9.pth"
    import_path = "NuTime repo"
    env_name = "tivit"
    default_encode_batch_size = 128
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        repo_root = Path(__file__).resolve().parents[3]
        self.nutime_root = repo_root / "external" / "NuTime"
        if not self.nutime_root.is_dir():
            raise FileNotFoundError(f"Expected NuTime repo at {self.nutime_root}")
        sys.path.insert(0, str(self.nutime_root))

        from src.config import Config
        from src.models.build import get_model
        from src.models.encoders.build import get_encoder

        config_path = self.nutime_root / "configs" / "demo_ft_epilepsy.json"
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        config = Config()
        config.update_by_dict(raw)
        config.num_channels = 1
        config.model_series_size = int(config.transform_size)
        config.out_dim = int(config.out_dim)
        config.num_classes = int(config.out_dim)
        self.config = config
        dummy_dataset = SimpleNamespace(samples=None, targets=None)
        encoder = get_encoder(self.config, dummy_dataset)
        backbone = get_model(self.config)
        self.model = torch.nn.Sequential(encoder, backbone)

        checkpoint_path = self.nutime_root / "ckpt" / self.checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("backbone.") and not key.startswith("backbone.fc."):
                remapped[key[len("backbone."):]] = value
        self.model.load_state_dict(remapped, strict=False)
        self.model.eval()
        self.transformer_depth = int(self.config.transformer_depth)
        self.transform_size = int(self.config.transform_size)

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.transformer_depth + 1))

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["transform_size"] = int(self.transform_size)
        payload["window_size"] = int(self.config.window_size)
        payload["stride"] = int(self.config.stride)
        payload["encoder"] = str(self.config.encoder)
        payload["pool_mode"] = str(self.config.pool_mode)
        payload["preprocess"] = (
            "resize input to demo transform_size; run WindowNormEncoder; use WinT cls-token representation "
            "after each transformer block"
        )
        payload["layer_layout"] = (
            "layer 0 is the token embedding stream after WindowNormEncoder and positional encoding; "
            "layers 1..N are transformer block outputs"
        )
        return payload

    def _build_tokens(self, x: torch.Tensor) -> torch.Tensor:
        encoder = self.model[0]
        backbone = self.model[1]
        resized = F.interpolate(
            x.to(dtype=torch.float32),
            size=self.transform_size,
            mode="linear",
            align_corners=False,
        )
        encoded = encoder(resized)
        if backbone.window_slide:
            raise ValueError("NuTime adapter expects wne encoder with window_slide=False")
        x_embed = encoded.transpose(1, 2)
        if backbone.cls_token is not None:
            cls_tokens = backbone.cls_token.expand(x_embed.size(0), -1, -1)
            x_embed = torch.cat((cls_tokens, x_embed), dim=1)
        x_embed = x_embed + backbone.pos_embed[:, : x_embed.shape[1], :]
        return backbone.pos_drop(x_embed)

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = set(layers or self.available_layers)
        backbone = self.model[1]
        hidden_states = self._build_tokens(x)

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            state = backbone.norm(hidden_states)
            if backbone.pool_mode == "cls":
                reps[0] = state[:, 0, :].float()
            else:
                reps[0] = state.mean(dim=1).float()
        for layer_index in range(1, self.transformer_depth + 1):
            attn, ls1, dp1, ff, ls2, dp2 = backbone.transformer.layers[
                (layer_index - 1) * 6 : (layer_index - 1) * 6 + 6
            ]
            hidden_states = dp1(ls1(attn(hidden_states))) + hidden_states
            hidden_states = dp2(ls2(ff(hidden_states))) + hidden_states
            if layer_index in requested_layers:
                state = backbone.norm(hidden_states)
                if backbone.pool_mode == "cls":
                    reps[int(layer_index)] = state[:, 0, :].float()
                else:
                    reps[int(layer_index)] = state.mean(dim=1).float()
        return reps
