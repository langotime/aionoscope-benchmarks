from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from .base import FrozenTimeSeriesAdapter


class _LeNEPABaseAdapter(FrozenTimeSeriesAdapter):
    env_name = "core"
    import_path = "published inference.py via huggingface_hub"
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        inference_path = self._download_bundle_file("inference.py")
        weights_path = self._download_bundle_file("lenepa_encoder.safetensors")
        config_path = self._download_bundle_file("lenepa_encoder_config.json")

        self.config = self._load_config(config_path)
        self.module = self._load_inference_module(inference_path)
        self.model = self._load_model(weights_path)
        self.num_blocks = int(len(self.model.blocks))
        config_depth = int(self.config["depth"])
        if self.num_blocks != config_depth:
            raise ValueError(
                f"{self.model_name} depth mismatch: config depth={config_depth} model blocks={self.num_blocks}"
            )

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_blocks + 1))

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["published_sampling_frequency"] = self.config["sampling_frequency"]
        payload["published_channels"] = list(self.config["channels"])
        payload["published_channel_size"] = int(self.config["channel_size"])
        payload["patch_size"] = int(self.config["patch_size"])
        payload["num_patches"] = int(self.config["num_patches"])
        payload["dim"] = int(self.config["dim"])
        payload["depth"] = int(self.config["depth"])
        payload["n_benchmark_layers"] = int(self.num_blocks + 1)
        payload["tokenizer"] = str(self.config.get("nepa_static_tokenizer", "conv_patch_embed"))
        if "nepa_patch_embed_scalar_stats_mode" in self.config:
            payload["patch_stats_mode"] = str(self.config["nepa_patch_embed_scalar_stats_mode"])
        payload["preprocess"] = "pass through benchmark waveform; benchmark already supplies [B,1,5000]"
        payload["layer_layout"] = (
            "layer 0 is the mean-pooled tokenizer output; "
            "layers 1..N-1 are mean-pooled patch tokens after transformer blocks 1..N-1; "
            "layer N is mean-pooled post-final-norm tokens after the last transformer block"
        )
        return payload

    def _download_bundle_file(self, filename: str) -> Path:
        try:
            path = hf_hub_download(repo_id=self.checkpoint, filename=filename)
        except Exception as exc:  # pragma: no cover - network and cache failures
            raise RuntimeError(
                f"Failed to download {filename!r} for {self.model_name} from repo {self.checkpoint!r}: {exc}"
            ) from exc
        return Path(path)

    def _load_config(self, config_path: Path) -> dict[str, Any]:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(
                f"{self.model_name} config must be a JSON object, got {type(raw).__name__}"
            )
        if raw.get("format") != "lenepa_encoder":
            raise ValueError(
                f"{self.model_name} expected format='lenepa_encoder', got {raw.get('format')!r}"
            )
        if int(raw["channel_size"]) != 5000:
            raise ValueError(
                f"{self.model_name} expects channel_size=5000, got {raw['channel_size']!r}"
            )
        if list(raw["channels"]) != [raw["channels"][0]] or len(raw["channels"]) != 1:
            raise ValueError(
                f"{self.model_name} expects a single-channel encoder export, got channels={raw['channels']!r}"
            )
        return raw

    def _load_inference_module(self, inference_path: Path) -> ModuleType:
        module_hash = hashlib.sha1(f"{self.checkpoint}:{inference_path}".encode("utf-8")).hexdigest()[:12]
        module_name = f"_aionoscope_lenepa_{module_hash}"
        existing = sys.modules.get(module_name)
        if existing is not None:
            return existing

        spec = importlib.util.spec_from_file_location(module_name, inference_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load LeNEPA inference module for {self.model_name} from {inference_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _load_model(self, weights_path: Path) -> torch.nn.Module:
        load_lenepa_encoder = getattr(self.module, "load_lenepa_encoder", None)
        if load_lenepa_encoder is None:
            raise ImportError(
                f"LeNEPA inference module for {self.model_name} does not define load_lenepa_encoder"
            )
        model = load_lenepa_encoder(weights_path=weights_path, device=torch.device("cpu"))
        if not hasattr(model, "blocks"):
            raise ValueError(f"{self.model_name} encoder is missing the expected 'blocks' attribute")
        return model

    def _validate_inputs(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(
                f"{self.model_name} expects input [B, C, L], got {tuple(x.shape)}"
            )
        expected_channels = len(self.config["channels"])
        expected_length = int(self.config["channel_size"])
        if int(x.shape[1]) != expected_channels or int(x.shape[2]) != expected_length:
            raise ValueError(
                f"{self.model_name} expects [B, {expected_channels}, {expected_length}], "
                f"got {tuple(x.shape)}"
            )

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        tokenize = getattr(self.model, "_tokenize", None)
        if tokenize is not None:
            return tokenize(x)
        patch_embed = getattr(self.model, "patch_embed", None)
        if patch_embed is None:
            raise ValueError(
                f"{self.model_name} encoder does not expose _tokenize() or patch_embed"
            )
        return patch_embed(x)

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = set(int(layer) for layer in (layers or self.available_layers))
        self._validate_inputs(x)
        hidden = self._tokenize(x.to(dtype=torch.float32))
        final_layer_index = self.num_blocks

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            reps[0] = hidden.mean(dim=1).float()
        for layer_index, block in enumerate(self.model.blocks, start=1):
            hidden = block(hidden)  # [B, T, D]
            if layer_index in requested_layers:
                pooled_tokens = hidden
                if layer_index == final_layer_index:
                    pooled_tokens = self.model.norm(pooled_tokens)
                reps[int(layer_index)] = pooled_tokens.mean(dim=1).float()
        return reps


class LeNEPAAionoAdapter(_LeNEPABaseAdapter):
    model_name = "LeNEPA-Aiono"
    model_slug = "LeNEPA-Aiono"
    source = "https://huggingface.co/Natively-TS-Understanding/lenepa-encoder-aiono"
    checkpoint = "Natively-TS-Understanding/lenepa-encoder-aiono"
    default_encode_batch_size = 128


class LeNEPACauKer2MAdapter(_LeNEPABaseAdapter):
    model_name = "LeNEPA-CauKer2M"
    model_slug = "LeNEPA-CauKer2M"
    source = "https://huggingface.co/Natively-TS-Understanding/lenepa-cauker2m-5000-patchnorm-d256-steps200k"
    checkpoint = "Natively-TS-Understanding/lenepa-cauker2m-5000-patchnorm-d256-steps200k"
    default_encode_batch_size = 64
