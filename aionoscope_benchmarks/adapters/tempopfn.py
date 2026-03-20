from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download

from .base import FrozenTimeSeriesAdapter


class TempoPFN38MAdapter(FrozenTimeSeriesAdapter):
    model_name = "TempoPFN-38M"
    model_slug = "TempoPFN-38M"
    source = "https://github.com/automl/TempoPFN"
    checkpoint = "AutoML-org/TempoPFN"
    import_path = "published Hugging Face repo snapshot via huggingface_hub"
    env_name = "tempopfn"
    default_encode_batch_size = 32
    use_bfloat16_amp = True
    prediction_length = 1
    _fixed_start = np.datetime64("2000-01-01")

    def __init__(self) -> None:
        super().__init__()
        self.snapshot_root = Path(
            snapshot_download(
                repo_id=self.checkpoint,
                allow_patterns=[
                    "configs/example.yaml",
                    "models/checkpoint_38M.pth",
                    "src/**",
                ],
            )
        )
        if not self.snapshot_root.is_dir():
            raise FileNotFoundError(
                f"{self.model_name} snapshot root does not exist: {self.snapshot_root}"
            )
        snapshot_root_str = str(self.snapshot_root)
        if snapshot_root_str not in sys.path:
            sys.path.insert(0, snapshot_root_str)

        self.config_path = self.snapshot_root / "configs" / "example.yaml"
        self.checkpoint_path = self.snapshot_root / "models" / "checkpoint_38M.pth"
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"{self.model_name} expected config at {self.config_path}"
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"{self.model_name} expected checkpoint at {self.checkpoint_path}"
            )

        try:
            from src.data.containers import BatchTimeSeriesContainer
            from src.data.frequency import Frequency
            from src.data.time_features import compute_batch_time_features
            from src.models.model import TimeSeriesModel
        except Exception as error:  # pragma: no cover - import error path
            raise ImportError(
                f"{self.model_name} could not import the official TempoPFN runtime from "
                f"{self.snapshot_root}. Install the published TempoPFN dependencies in "
                f"the '.venv-tempopfn' environment."
            ) from error

        with self.config_path.open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle)
        if not isinstance(raw_config, dict):
            raise ValueError(
                f"{self.model_name} expected a mapping in {self.config_path}, got {type(raw_config).__name__}"
            )
        raw_model_config = raw_config.get("TimeSeriesModel")
        if not isinstance(raw_model_config, dict):
            raise ValueError(
                f"{self.model_name} expected 'TimeSeriesModel' mapping in {self.config_path}"
            )
        gift_eval_config = raw_config.get("gift_eval")
        if not isinstance(gift_eval_config, dict):
            raise ValueError(
                f"{self.model_name} expected 'gift_eval' mapping in {self.config_path}"
            )
        max_context_length = gift_eval_config.get("max_context_length")
        if not isinstance(max_context_length, int) or isinstance(max_context_length, bool):
            raise ValueError(
                f"{self.model_name} expected integer gift_eval.max_context_length in {self.config_path}, "
                f"got {max_context_length!r}"
            )

        encoder_config = raw_model_config.get("encoder_config")
        if not isinstance(encoder_config, dict):
            raise ValueError(
                f"{self.model_name} expected 'TimeSeriesModel.encoder_config' mapping in {self.config_path}"
            )

        self.batch_container_cls = BatchTimeSeriesContainer
        self.fixed_frequency = Frequency.D
        self.compute_batch_time_features = compute_batch_time_features
        self.model_cls = TimeSeriesModel
        self.model_config: dict[str, Any] = raw_model_config
        self.encoder_config: dict[str, Any] = encoder_config
        self.num_hidden_layers = int(raw_model_config["num_encoder_layers"])
        self.embed_size = int(raw_model_config["embed_size"])
        self.k_max = int(raw_model_config["K_max"])
        self.loss_type = str(raw_model_config.get("loss_type", "unknown"))
        self.quantiles = [float(value) for value in raw_model_config.get("quantiles", [])]
        self.benchmark_sequence_length = int(max_context_length)
        self.benchmark_sequence_length_source = "official_tempopfn_gift_eval_max_context_length"

        self.model: torch.nn.Module | None = None
        self.model_input_dtype = torch.float32

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
        # Runtime dataset tensors may stay CPU-backed even when the benchmark device is CUDA.
        # Delay model materialization until prepare_runtime(), which receives the actual execution device.
        del manifest, train_split, val_split

    def prepare_runtime(self, *, device: torch.device) -> None:
        self._ensure_model_loaded(device=device)

    def parameter_count_prefix_sources(self) -> dict[int, tuple[object, ...]] | None:
        if self.model is None:
            return None
        sources: dict[int, tuple[object, ...]] = {
            0: (
                self.model.expand_values,
                self.model.nan_embedding,
                self.model.time_feature_projection,
            )
        }
        for layer_index, encoder_layer in enumerate(self.model.encoder_layers, start=1):
            sources[int(layer_index)] = (
                self.model.initial_hidden_state[layer_index - 1],
                encoder_layer,
            )
        return sources

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["context_length"] = int(self.benchmark_sequence_length)
        payload["prediction_length"] = int(self.prediction_length)
        payload["embed_size"] = int(self.embed_size)
        payload["num_hidden_layers"] = int(self.num_hidden_layers)
        payload["k_max"] = int(self.k_max)
        payload["loss_type"] = str(self.loss_type)
        payload["quantiles"] = list(self.quantiles)
        payload["num_heads"] = int(self.encoder_config["num_heads"])
        payload["attn_mode"] = str(self.encoder_config.get("attn_mode", "unknown"))
        payload["conv_size"] = int(self.encoder_config.get("conv_size", 0))
        payload["num_householder"] = int(self.encoder_config.get("num_householder", 0))
        payload["weaving"] = bool(self.encoder_config.get("weaving", False))
        payload["fixed_frequency"] = str(self.fixed_frequency.value)
        payload["time_grid_start"] = str(self._fixed_start)
        payload["checkpoint_file"] = "models/checkpoint_38M.pth"
        payload["preprocess"] = (
            "use the full exact benchmark waveform as TempoPFN history context, append a deterministic "
            "one-step zero-valued future query internally, run the official robust-scaling and time-feature "
            "pipeline on a fixed daily calendar grid, and mean-pool history token states only"
        )
        payload["layer_layout"] = (
            "layer 0 is the history embedding stream after value and time-feature projection; "
            "layers 1..N are encoder block outputs pooled over history positions"
        )
        payload["notes"] = (
            "The official TempoPFN model card documents CUDA inference. Benchmark results should therefore be "
            "produced from the dedicated '.venv-tempopfn' CUDA environment."
        )
        return payload

    def _ensure_model_loaded(self, *, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError(
                f"{self.model_name} requires CUDA for the published TempoPFN inference path; "
                f"got device={device}."
            )
        if isinstance(self.model, torch.nn.Module):
            model_device = next(self.model.parameters()).device
            if model_device != device:
                self.model = self.model.to(device)
                self.model.eval()
                self.model_input_dtype = next(self.model.parameters()).dtype
            return

        loaded_model = self.model_cls(**self.model_config).to(device)
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"{self.model_name} expected a checkpoint dict at {self.checkpoint_path}, "
                f"got {type(checkpoint).__name__}"
            )
        model_state_dict = checkpoint.get("model_state_dict")
        if not isinstance(model_state_dict, dict):
            raise ValueError(
                f"{self.model_name} checkpoint at {self.checkpoint_path} is missing 'model_state_dict'"
            )
        loaded_model.load_state_dict(model_state_dict)
        loaded_model.eval()
        self.model = loaded_model
        self.model_input_dtype = next(self.model.parameters()).dtype

    def _build_batch_container(
        self,
        context: torch.Tensor,
    ):
        batch_size, history_length = context.shape
        future_values = torch.zeros(
            batch_size,
            self.prediction_length,
            1,
            dtype=context.dtype,
            device=context.device,
        )
        return self.batch_container_cls(
            history_values=context.unsqueeze(-1),
            future_values=future_values,
            start=[self._fixed_start] * batch_size,
            frequency=[self.fixed_frequency] * batch_size,
        )

    def _pool_history_tokens(
        self,
        hidden_states: torch.Tensor,
        *,
        history_length: int,
    ) -> torch.Tensor:
        return hidden_states[:, :history_length, :].mean(dim=1).float()

    def _build_initial_hidden_states(
        self,
        *,
        batch_size: int,
        num_channels: int,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError(f"{self.model_name} model is not loaded")
        return torch.zeros_like(
            self.model.initial_hidden_state[0].repeat(batch_size * num_channels, 1, 1, 1)
        )

    def _tempo_input_sequence(
        self,
        context: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        if self.model is None:
            raise RuntimeError(f"{self.model_name} model is not loaded")
        batch_container = self._build_batch_container(context)
        preprocessed = self.model._preprocess_data(batch_container)

        history_time_features, target_time_features = self.compute_batch_time_features(
            start=batch_container.start,
            history_length=preprocessed["history_length"],
            future_length=preprocessed["future_length"],
            batch_size=preprocessed["batch_size"],
            frequency=batch_container.frequency,
            K_max=self.k_max,
            time_feature_config=self.model.time_feature_config,
        )
        history_time_features = history_time_features.to(device=context.device)
        target_time_features = target_time_features.to(device=context.device)

        scale_statistics = self.model._compute_scaling(
            preprocessed["history_values"],
            preprocessed["history_mask"],
        )
        history_scaled = self.model._apply_scaling_and_masking(
            preprocessed["history_values"],
            scale_statistics,
            preprocessed["history_mask"],
        )

        history_pos_embed = self.model._get_positional_embeddings(
            history_time_features,
            preprocessed["num_channels"],
            preprocessed["batch_size"],
            False,
        )
        target_pos_embed = self.model._get_positional_embeddings(
            target_time_features,
            preprocessed["num_channels"],
            preprocessed["batch_size"],
            False,
        )
        history_embed = self.model._compute_embeddings(
            history_scaled,
            history_pos_embed,
            preprocessed["history_mask"],
        )

        batch_size = int(preprocessed["batch_size"])
        history_length = int(preprocessed["history_length"])
        num_channels = int(preprocessed["num_channels"])
        history_embed = history_embed.view(batch_size, history_length, num_channels, self.embed_size)
        history_tokens = (
            history_embed.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * num_channels, history_length, self.embed_size)
        )
        future_tokens = (
            target_pos_embed.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * num_channels, self.prediction_length, self.embed_size)
        )
        return preprocessed, torch.cat([history_tokens, future_tokens], dim=1)

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
        if self.model is None:
            raise RuntimeError(f"{self.model_name} model is not loaded")

        context = x[:, 0, :].to(dtype=torch.float32)
        preprocessed, hidden_states = self._tempo_input_sequence(context)
        batch_size = int(preprocessed["batch_size"])
        history_length = int(preprocessed["history_length"])
        num_channels = int(preprocessed["num_channels"])

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            reps[0] = self._pool_history_tokens(hidden_states, history_length=history_length)

        if bool(self.model.encoder_config.get("weaving", True)):
            recurrent_hidden = self._build_initial_hidden_states(
                batch_size=batch_size,
                num_channels=num_channels,
            )
            for layer_index, encoder_layer in enumerate(self.model.encoder_layers, start=1):
                recurrent_hidden = recurrent_hidden + self.model.initial_hidden_state[layer_index - 1].repeat(
                    batch_size * num_channels,
                    1,
                    1,
                    1,
                )
                hidden_states, recurrent_hidden = encoder_layer(
                    hidden_states,
                    recurrent_hidden,
                )
                if layer_index in requested_layers:
                    reps[int(layer_index)] = self._pool_history_tokens(
                        hidden_states,
                        history_length=history_length,
                    )
            return reps

        for layer_index, encoder_layer in enumerate(self.model.encoder_layers, start=1):
            initial_hidden = self.model.initial_hidden_state[layer_index - 1].repeat(
                batch_size * num_channels,
                1,
                1,
                1,
            )
            hidden_states, _ = encoder_layer(hidden_states, initial_hidden)
            if layer_index in requested_layers:
                reps[int(layer_index)] = self._pool_history_tokens(
                    hidden_states,
                    history_length=history_length,
                )
        return reps
