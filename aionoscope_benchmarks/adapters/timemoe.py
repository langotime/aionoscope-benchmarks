from __future__ import annotations

import torch

from .base import FrozenTimeSeriesAdapter


class _TimeMoeAdapter(FrozenTimeSeriesAdapter):
    source = "https://github.com/Time-MoE/Time-MoE"
    import_path = "transformers"
    env_name = "timemoe"
    use_bfloat16_amp = True
    default_encode_batch_size = 1
    eager_encode_batch_size = 1
    flash_encode_batch_size = 1

    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(self.checkpoint, trust_remote_code=True)
        model_type = getattr(self.config, "model_type", None)
        if model_type != "time_moe":
            raise ValueError(f"{self.model_name} expected model_type='time_moe', got {model_type!r}")

        self.num_hidden_layers = int(self.config.num_hidden_layers)
        self.horizon_lengths = [int(length) for length in self.config.horizon_lengths]
        self.benchmark_sequence_length = int(self.config.max_position_embeddings)
        self.benchmark_sequence_length_source = "time_moe_config.max_position_embeddings"

        self.model: torch.nn.Module | None = None
        self.decoder: torch.nn.Module | None = None
        self.model_input_dtype = torch.float32
        self.attention_implementation = "uninitialized"

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_hidden_layers + 1))

    def parameter_count_prefix_sources(self) -> dict[int, tuple[object, ...]] | None:
        if self.decoder is None:
            return None
        sources: dict[int, tuple[object, ...]] = {0: (self.decoder.embed_layer,)}
        for layer_index, layer in enumerate(self.decoder.layers, start=1):
            layer_sources: tuple[object, ...] = (layer,)
            if layer_index == self.num_hidden_layers:
                layer_sources = layer_sources + (self.decoder.norm,)
            sources[int(layer_index)] = layer_sources
        return sources

    def prepare(
        self,
        *,
        manifest: dict[str, object],
        train_split: dict[str, torch.Tensor],
        val_split: dict[str, torch.Tensor],
    ) -> None:
        del manifest, val_split
        self._ensure_model_loaded(device=train_split["x"].device)

    def prepare_runtime(self, *, device: torch.device) -> None:
        self._ensure_model_loaded(device=device)

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["context_length"] = int(self.benchmark_sequence_length)
        payload["input_size"] = int(self.config.input_size)
        payload["hidden_size"] = int(self.config.hidden_size)
        payload["num_hidden_layers"] = int(self.num_hidden_layers)
        payload["num_experts"] = int(self.config.num_experts)
        payload["num_experts_per_tok"] = int(self.config.num_experts_per_tok)
        payload["horizon_lengths"] = list(self.horizon_lengths)
        payload["attention_implementation"] = str(self.attention_implementation)
        payload["preprocess"] = (
            "per-series z-score normalization over the exact benchmark context with std clamped at 1e-6; "
            "mean-pool causal token states across time"
        )
        payload["layer_layout"] = (
            "layer 0 is the mean-pooled input embedding stream; "
            "layers 1..N-1 are mean-pooled decoder block outputs; "
            "layer N is the mean-pooled post-final-norm decoder output"
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

    def _set_runtime_batch_size(self) -> None:
        if self.attention_implementation == "flash_attention_2":
            self.default_encode_batch_size = int(self.flash_encode_batch_size)
            return
        self.default_encode_batch_size = int(self.eager_encode_batch_size)

    def _ensure_model_loaded(self, *, device: torch.device) -> None:
        requested_attention_impl = self._preferred_attention_implementation(device)
        if isinstance(self.model, torch.nn.Module):
            loaded_attention_impl = str(
                getattr(self.model.config, "_attn_implementation", self.attention_implementation)
            )
            needs_attention_reload = (
                requested_attention_impl == "flash_attention_2"
                and loaded_attention_impl != requested_attention_impl
            )
            if not needs_attention_reload:
                model_device = next(self.model.parameters()).device
                if model_device != device:
                    self.model = self.model.to(device)
                    self.model.eval()
                    self.decoder = self.model.get_decoder()
                    self.model_input_dtype = next(self.model.parameters()).dtype
                self.attention_implementation = loaded_attention_impl
                self._set_runtime_batch_size()
                return

            # prepare() can warm the model on CPU before the benchmark switches to CUDA.
            # Reload so the decoder layers are rebuilt with the flash-attention classes.
            old_model = self.model
            self.model = None
            self.decoder = None
            self.attention_implementation = "uninitialized"
            del old_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        from transformers import AutoModelForCausalLM

        loaded_model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation=requested_attention_impl,
        )
        loaded_model = loaded_model.to(device)
        loaded_model.eval()

        get_decoder = getattr(loaded_model, "get_decoder", None)
        if get_decoder is None:
            raise ValueError(f"{self.model_name} model does not expose get_decoder()")
        decoder = get_decoder()
        for attr_name in ("embed_layer", "layers", "norm"):
            if not hasattr(decoder, attr_name):
                raise ValueError(f"{self.model_name} decoder is missing {attr_name!r}")
        if int(len(decoder.layers)) != self.num_hidden_layers:
            raise ValueError(
                f"{self.model_name} layer mismatch: config={self.num_hidden_layers} decoder={len(decoder.layers)}"
            )

        self.model = loaded_model
        self.decoder = decoder
        self.model_input_dtype = next(self.model.parameters()).dtype
        self.attention_implementation = str(
            getattr(self.model.config, "_attn_implementation", requested_attention_impl)
        )
        self._set_runtime_batch_size()

    def _normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        mean = context.mean(dim=-1, keepdim=True)
        std = context.std(dim=-1, keepdim=True)
        return (context - mean) / std.clamp_min(1e-6)

    def _build_decoder_inputs(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        if self.decoder is None:
            raise RuntimeError(f"{self.model_name} decoder is not loaded")

        batch_size, seq_length = context.shape
        input_series = context.unsqueeze(-1).to(dtype=self.model_input_dtype)
        hidden_states = self.decoder.embed_layer(input_series)
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
        position_ids = torch.arange(
            seq_length,
            device=context.device,
            dtype=torch.long,
        ).unsqueeze(0)
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
        if self.decoder is None:
            raise RuntimeError(f"{self.model_name} decoder is not loaded")

        context = self._normalize_context(x[:, 0, :].to(dtype=torch.float32))
        hidden_states, causal_attention_mask, position_ids = self._build_decoder_inputs(context)
        max_requested_layer = max(requested_layers, default=0)

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            reps[0] = hidden_states.mean(dim=1).float()

        for layer_index, layer in enumerate(self.decoder.layers, start=1):
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
            if layer_index in requested_layers:
                pooled_states = hidden_states
                if layer_index == self.num_hidden_layers:
                    pooled_states = self.decoder.norm(pooled_states)
                reps[int(layer_index)] = pooled_states.mean(dim=1).float()
            if layer_index >= max_requested_layer:
                break
        return reps


class TimeMoeBaseAdapter(_TimeMoeAdapter):
    model_name = "Time-MoE-50M"
    model_slug = "Time-MoE-50M"
    checkpoint = "Maple728/TimeMoE-50M"
    flash_encode_batch_size = 256


class TimeMoeLargeAdapter(_TimeMoeAdapter):
    model_name = "Time-MoE-200M"
    model_slug = "Time-MoE-200M"
    checkpoint = "Maple728/TimeMoE-200M"
    flash_encode_batch_size = 256
