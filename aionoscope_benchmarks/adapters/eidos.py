from __future__ import annotations

import sys

import torch

from ..constants import REPO_ROOT
from .base import FrozenTimeSeriesAdapter


class EIDOSAdapter(FrozenTimeSeriesAdapter):
    model_name = "EIDOS"
    model_slug = "EIDOS"
    source = "https://arxiv.org/abs/2602.14024"
    checkpoint = "external/EIDOS/eidos 1.pt"
    import_path = "local external/EIDOS runtime"
    env_name = "timemoe"
    default_encode_batch_size = 1024
    use_bfloat16_amp = True
    benchmark_sequence_length = 512
    benchmark_sequence_length_source = "official_eidos_readme_basic_usage_history_length"

    def __init__(self) -> None:
        super().__init__()
        self.eidos_root = REPO_ROOT / "external" / "EIDOS"
        self.checkpoint_path = self.eidos_root / "eidos 1.pt"
        if not self.eidos_root.is_dir():
            raise FileNotFoundError(f"Expected EIDOS code at {self.eidos_root}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"{self.model_name} expected checkpoint at {self.checkpoint_path}"
            )
        eidos_root_str = str(self.eidos_root)
        if eidos_root_str not in sys.path:
            sys.path.insert(0, eidos_root_str)

        try:
            from configuration_eidos import EidosConfig
            from modeling_eidos import EidosForPrediction
        except Exception as error:  # pragma: no cover - import error path
            raise ImportError(
                f"{self.model_name} could not import the local EIDOS runtime from {self.eidos_root}. "
                "Install the required 'transformers' and 'einops' packages in the '.venv-timemoe' "
                "environment before running this adapter."
            ) from error

        self.config_cls = EidosConfig
        self.model_cls = EidosForPrediction
        self.config = self.config_cls(_attn_implementation="eager")
        self.hidden_size = int(self.config.hidden_size)
        self.intermediate_size = int(self.config.intermediate_size)
        self.num_hidden_layers = int(self.config.num_local_decoder_layers)
        self.num_attention_heads = int(self.config.num_attention_heads)
        self.num_key_value_heads = int(self.config.num_key_value_heads)
        self.max_position_embeddings = int(self.config.max_position_embeddings)
        self.horizon_lengths = [int(value) for value in self.config.horizon_lengths]
        self.quantiles = [float(value) for value in self.config.quantiles]

        self.model: torch.nn.Module | None = None
        self.decoder_model: torch.nn.Module | None = None
        self.model_input_dtype = torch.float32
        self.attention_implementation = "uninitialized"

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_hidden_layers + 1))

    def prepare(
        self,
        *,
        manifest: dict[str, object],
        train_split: dict[str, torch.Tensor],
        val_split: dict[str, torch.Tensor],
    ) -> None:
        del manifest, train_split, val_split

    def prepare_runtime(self, *, device: torch.device) -> None:
        self._ensure_model_loaded(device=device)

    def parameter_count_prefix_sources(self) -> dict[int, tuple[object, ...]] | None:
        if self.decoder_model is None:
            return None
        decoder_layers = self.decoder_model.decoder.layers
        sources: dict[int, tuple[object, ...]] = {0: (self.decoder_model.embed_layer,)}
        for layer_index, layer in enumerate(decoder_layers, start=1):
            layer_sources: tuple[object, ...] = (layer,)
            if layer_index == self.num_hidden_layers:
                layer_sources = layer_sources + (self.decoder_model.norm,)
            sources[int(layer_index)] = layer_sources
        return sources

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["context_length"] = int(self.benchmark_sequence_length)
        payload["max_position_embeddings"] = int(self.max_position_embeddings)
        payload["hidden_size"] = int(self.hidden_size)
        payload["intermediate_size"] = int(self.intermediate_size)
        payload["num_hidden_layers"] = int(self.num_hidden_layers)
        payload["num_attention_heads"] = int(self.num_attention_heads)
        payload["num_key_value_heads"] = int(self.num_key_value_heads)
        payload["horizon_lengths"] = list(self.horizon_lengths)
        payload["quantiles"] = list(self.quantiles)
        payload["attention_implementation"] = str(self.attention_implementation)
        payload["model_dtype"] = str(self.model_input_dtype).replace("torch.", "")
        payload["checkpoint_origin"] = "local_repo_hosted_checkpoint"
        payload["checkpoint_file"] = str(self.checkpoint)
        payload["preprocess"] = (
            "per-series z-score normalization over the exact 512-sample benchmark context with std clamped at 1e-5, "
            "then mean-pool causal token states across time"
        )
        payload["layer_layout"] = (
            "layer 0 is the mean-pooled SIREN embedding stream; "
            "layers 1..N-1 are mean-pooled decoder block outputs; "
            "layer N is the mean-pooled post-final-norm decoder output"
        )
        payload["notes"] = (
            "Public upstream code is not yet available; the benchmark therefore uses the local official code drop "
            "under 'external/EIDOS' and the repo-hosted checkpoint file 'eidos 1.pt' from the '.venv-timemoe' "
            "environment."
        )
        return payload

    def _preferred_attention_implementation(self, device: torch.device) -> str:
        if device.type != "cuda":
            return "eager"
        try:
            import flash_attn  # noqa: F401
        except Exception:
            return "eager"
        return "flash_attention_2"

    @staticmethod
    def _clean_state_dict(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            cleaned_key = str(key)
            if cleaned_key.startswith("_orig_mod.model.model."):
                cleaned_key = cleaned_key.replace("_orig_mod.model.model.", "model.", 1)
            elif cleaned_key.startswith("_orig_mod.model."):
                cleaned_key = cleaned_key.replace("_orig_mod.model.", "", 1)
            elif cleaned_key.startswith("model.model."):
                cleaned_key = cleaned_key.replace("model.model.", "model.", 1)
            cleaned[cleaned_key] = value
        return cleaned

    def _ensure_model_loaded(self, *, device: torch.device) -> None:
        requested_attention_impl = self._preferred_attention_implementation(device)
        requested_dtype = (
            torch.bfloat16
            if device.type == "cuda" and requested_attention_impl == "flash_attention_2"
            else torch.float32
        )
        if isinstance(self.model, torch.nn.Module):
            model_device = next(self.model.parameters()).device
            current_dtype = next(self.model.parameters()).dtype
            if self.attention_implementation == requested_attention_impl:
                if model_device != device or current_dtype != requested_dtype:
                    self.model = self.model.to(device=device, dtype=requested_dtype)
                    self.model.eval()
                    self.decoder_model = self.model.get_decoder()
                    self.model_input_dtype = next(self.model.parameters()).dtype
                return

            old_model = self.model
            self.model = None
            self.decoder_model = None
            self.attention_implementation = "uninitialized"
            del old_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        loaded_model = self.model_cls(self.config_cls(_attn_implementation=requested_attention_impl))
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"{self.model_name} expected a checkpoint dict at {self.checkpoint_path}, "
                f"got {type(checkpoint).__name__}"
            )
        raw_state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(raw_state_dict, dict):
            raise ValueError(
                f"{self.model_name} checkpoint at {self.checkpoint_path} does not expose a state dict"
            )
        loaded_model.load_state_dict(self._clean_state_dict(raw_state_dict), strict=True)
        loaded_model = loaded_model.to(device=device, dtype=requested_dtype)
        loaded_model.eval()

        decoder_model = loaded_model.get_decoder()
        for attr_name in ("embed_layer", "decoder", "norm"):
            if not hasattr(decoder_model, attr_name):
                raise ValueError(f"{self.model_name} decoder is missing {attr_name!r}")
        decoder_layers = getattr(decoder_model.decoder, "layers", None)
        if decoder_layers is None:
            raise ValueError(f"{self.model_name} decoder is missing 'decoder.layers'")
        if int(len(decoder_layers)) != self.num_hidden_layers:
            raise ValueError(
                f"{self.model_name} layer mismatch: config={self.num_hidden_layers} "
                f"decoder={len(decoder_layers)}"
            )

        self.model = loaded_model
        self.decoder_model = decoder_model
        self.model_input_dtype = next(self.model.parameters()).dtype
        self.attention_implementation = requested_attention_impl

    def _normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        mean = context.mean(dim=-1, keepdim=True)
        std = context.std(dim=-1, keepdim=True)
        return (context - mean) / std.clamp_min(1e-5)

    def _decoder_inputs(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        if self.decoder_model is None:
            raise RuntimeError(f"{self.model_name} decoder is not loaded")

        batch_size, seq_length = context.shape
        input_series = context.unsqueeze(-1).to(dtype=self.model_input_dtype)
        hidden_states = self.decoder_model.embed_layer(input_series)
        position_ids = torch.arange(
            seq_length,
            device=context.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)
        if self.attention_implementation == "flash_attention_2":
            return hidden_states, None, position_ids

        attention_mask = torch.ones(
            (batch_size, seq_length),
            dtype=torch.long,
            device=context.device,
        )
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            0,
            sliding_window=None,
        )
        return hidden_states, causal_attention_mask, position_ids

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = set(int(layer) for layer in (layers or self.available_layers))
        invalid_layers = sorted(layer for layer in requested_layers if layer not in self.available_layers)
        if invalid_layers:
            raise ValueError(f"{self.model_name} requested invalid layers {invalid_layers}")

        self.validate_benchmark_input(x, channels=1)
        self._ensure_model_loaded(device=x.device)
        if self.decoder_model is None:
            raise RuntimeError(f"{self.model_name} decoder is not loaded")

        context = self._normalize_context(x[:, 0, :].to(dtype=torch.float32))
        hidden_states, attention_mask, position_ids = self._decoder_inputs(context)
        outputs = self.decoder_model.decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        decoder_hidden_states = outputs.hidden_states
        if decoder_hidden_states is None:
            raise ValueError(f"{self.model_name} decoder did not return hidden states")
        expected_hidden_state_count = self.num_hidden_layers + 1
        if int(len(decoder_hidden_states)) != expected_hidden_state_count:
            raise ValueError(
                f"{self.model_name} decoder returned {len(decoder_hidden_states)} hidden-state tensors, "
                f"expected {expected_hidden_state_count}"
            )

        reps: dict[int, torch.Tensor] = {}
        for layer_index in sorted(requested_layers):
            if layer_index == self.num_hidden_layers:
                state = self.decoder_model.norm(outputs.last_hidden_state)
            else:
                state = decoder_hidden_states[layer_index]
            reps[int(layer_index)] = state.mean(dim=1).float()
        return reps
