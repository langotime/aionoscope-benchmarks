from __future__ import annotations

import torch

from .base import FrozenTimeSeriesAdapter


class _THUMLCausalAdapter(FrozenTimeSeriesAdapter):
    import_path = "transformers"
    env_name = "timemoe"
    default_encode_batch_size = 64
    use_bfloat16_amp = True
    benchmark_sequence_length = 2880

    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self.model.eval()
        get_decoder = getattr(self.model, "get_decoder", None)
        if get_decoder is None:
            raise ValueError(f"{self.model_name} model does not expose get_decoder()")
        self.decoder = get_decoder()
        self.num_hidden_layers = int(self.model.config.num_hidden_layers)
        if int(len(self.decoder.layers)) != self.num_hidden_layers:
            raise ValueError(
                f"{self.model_name} layer mismatch: config={self.num_hidden_layers} decoder={len(self.decoder.layers)}"
            )
        self.input_token_len = int(self.model.config.input_token_len)
        self.hidden_size = int(self.model.config.hidden_size)

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_hidden_layers + 1))

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["hidden_size"] = int(self.hidden_size)
        payload["input_token_len"] = int(self.input_token_len)
        payload["num_hidden_layers"] = int(self.num_hidden_layers)
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
        context = self._context_tensor(x)
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


class SundialBase128MAdapter(_THUMLCausalAdapter):
    model_name = "Sundial-Base-128M"
    model_slug = "Sundial-Base-128M"
    source = "https://github.com/thuml/Sundial"
    checkpoint = "thuml/sundial-base-128m"
    benchmark_sequence_length_source = "official_sundial_readme_quickstart_lookback_length"
