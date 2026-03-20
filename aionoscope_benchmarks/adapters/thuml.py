from __future__ import annotations

import torch

from .base import FrozenTimeSeriesAdapter


class _THUMLCausalAdapter(FrozenTimeSeriesAdapter):
    import_path = "transformers"
    env_name = "timemoe"
    default_encode_batch_size = 512
    use_bfloat16_amp = True
    benchmark_sequence_length = 2880

    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
        )
        self.model: torch.nn.Module | None = None
        self.decoder: torch.nn.Module | None = None
        self.attention_implementation = "uninitialized"
        self.model_input_dtype = torch.float32
        self.num_hidden_layers = int(self.config.num_hidden_layers)
        self.input_token_len = int(self.config.input_token_len)
        self.hidden_size = int(self.config.hidden_size)

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

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["hidden_size"] = int(self.hidden_size)
        payload["input_token_len"] = int(self.input_token_len)
        payload["num_hidden_layers"] = int(self.num_hidden_layers)
        payload["attention_implementation"] = str(self.attention_implementation)
        payload["model_dtype"] = str(self.model_input_dtype).replace("torch.", "")
        payload["preprocess"] = (
            "feed the exact official quickstart lookback length directly into the published decoder-only model; "
            "mean-pool token states across time"
        )
        payload["layer_layout"] = (
            "layer 0 is the patch embedding stream; "
            "layers 1..N-1 are decoder block outputs; "
            "layer N is the post-final-norm decoder output"
        )
        return payload

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

    def _preferred_attention_implementation(self, device: torch.device) -> str:
        if device.type != "cuda":
            return "eager"
        try:
            import flash_attn  # noqa: F401
        except Exception:
            return "eager"
        return "flash_attention_2"

    def _preferred_model_dtype(
        self,
        *,
        device: torch.device,
        attention_implementation: str,
    ) -> torch.dtype | None:
        if device.type == "cuda" and attention_implementation == "flash_attention_2":
            return torch.bfloat16
        return None

    def _ensure_model_loaded(self, *, device: torch.device) -> None:
        requested_attention_impl = self._preferred_attention_implementation(device)
        requested_dtype = self._preferred_model_dtype(
            device=device,
            attention_implementation=requested_attention_impl,
        )
        if isinstance(self.model, torch.nn.Module):
            loaded_attention_impl = str(
                getattr(self.model.config, "_attn_implementation", self.attention_implementation)
            )
            model_device = next(self.model.parameters()).device
            current_dtype = next(self.model.parameters()).dtype
            if loaded_attention_impl == requested_attention_impl:
                if model_device != device or (
                    requested_dtype is not None and current_dtype != requested_dtype
                ):
                    to_kwargs: dict[str, object] = {"device": device}
                    if requested_dtype is not None:
                        to_kwargs["dtype"] = requested_dtype
                    self.model = self.model.to(**to_kwargs)
                    self.model.eval()
                    self.decoder = self.model.get_decoder()
                self.attention_implementation = loaded_attention_impl
                self.model_input_dtype = next(self.model.parameters()).dtype
                return

            old_model = self.model
            self.model = None
            self.decoder = None
            self.attention_implementation = "uninitialized"
            del old_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        from transformers import AutoModelForCausalLM

        load_kwargs: dict[str, object] = {
            "trust_remote_code": True,
            "attn_implementation": requested_attention_impl,
            "torch_dtype": "auto" if requested_dtype is None else requested_dtype,
        }
        loaded_model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            **load_kwargs,
        )
        loaded_model = loaded_model.to(device)
        loaded_model.eval()
        get_decoder = getattr(loaded_model, "get_decoder", None)
        if get_decoder is None:
            raise ValueError(f"{self.model_name} model does not expose get_decoder()")
        decoder = get_decoder()
        if int(len(decoder.layers)) != self.num_hidden_layers:
            raise ValueError(
                f"{self.model_name} layer mismatch: config={self.num_hidden_layers} decoder={len(decoder.layers)}"
            )
        self.model = loaded_model
        self.decoder = decoder
        self.attention_implementation = str(
            getattr(self.model.config, "_attn_implementation", requested_attention_impl)
        )
        self.model_input_dtype = next(self.model.parameters()).dtype

    def _context_tensor(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_benchmark_input(x, channels=1)
        context = x[:, 0, :].to(dtype=torch.float32)
        if context.shape[-1] % self.input_token_len != 0:
            raise ValueError(
                f"{self.model_name} expects context length divisible by input_token_len={self.input_token_len}, "
                f"got {tuple(context.shape)}"
            )
        return context

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = set(int(layer) for layer in (layers or self.available_layers))
        self._ensure_model_loaded(device=x.device)
        if self.decoder is None:
            raise RuntimeError(f"{self.model_name} decoder is not loaded")
        context = self._context_tensor(x).to(dtype=self.model_input_dtype)
        outputs = self.decoder(
            input_ids=context,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError(f"{self.model_name} decoder did not return hidden states")
        reps: dict[int, torch.Tensor] = {}
        for layer_index in requested_layers:
            reps[int(layer_index)] = hidden_states[int(layer_index)].mean(dim=1).float()
        return reps


class TimerBase84MAdapter(_THUMLCausalAdapter):
    model_name = "Timer-Base-84M"
    model_slug = "Timer-Base-84M"
    source = "https://github.com/thuml/Timer"
    checkpoint = "thuml/timer-base-84m"
    benchmark_sequence_length_source = "official_timer_model_card_context_length"
    default_encode_batch_size = 1024


class SundialBase128MAdapter(_THUMLCausalAdapter):
    model_name = "Sundial-Base-128M"
    model_slug = "Sundial-Base-128M"
    source = "https://github.com/thuml/Sundial"
    checkpoint = "thuml/sundial-base-128m"
    benchmark_sequence_length_source = "official_sundial_readme_quickstart_lookback_length"
