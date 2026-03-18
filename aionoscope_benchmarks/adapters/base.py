from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class AdapterMetadata:
    env: str
    encode_batch_size: int
    notes: str | None = None


class FrozenTimeSeriesAdapter(nn.Module, ABC):
    model_name: str
    model_slug: str
    source: str
    checkpoint: str
    import_path: str
    model_type: str = "foundational"
    env_name: str = "core"
    default_encode_batch_size: int = 32
    use_bfloat16_amp: bool = True
    cpu_feature_cache_dtype: torch.dtype = torch.float32
    benchmark_sequence_length: int | None = None
    benchmark_sequence_length_source: str = "adapter"

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def available_layers(self) -> tuple[int, ...]:
        raise NotImplementedError

    def autocast_context(self, device: torch.device):
        if device.type == "cuda" and self.use_bfloat16_amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def parameter_count(self) -> int | None:
        total = sum(int(parameter.numel()) for parameter in self.parameters())
        return int(total) if total > 0 else None

    def trainable_parameter_count(self) -> int | None:
        total = self.parameter_count()
        if total is None:
            return None
        trainable = sum(
            int(parameter.numel()) for parameter in self.parameters() if bool(parameter.requires_grad)
        )
        return int(trainable)

    def parameter_count_prefix_sources(self) -> dict[int, tuple[object, ...]] | None:
        return None

    def _iter_parameter_source_parameters(self, source: object) -> Iterable[nn.Parameter]:
        if source is None:
            return
        if isinstance(source, nn.Parameter):
            yield source
            return
        if isinstance(source, nn.Module):
            yield from source.parameters()
            return
        if isinstance(source, (list, tuple, set)):
            for item in source:
                yield from self._iter_parameter_source_parameters(item)
            return
        raise TypeError(
            f"{self.model_name} parameter-count source must be an nn.Module, nn.Parameter, "
            f"or a nested iterable of those, got {type(source).__name__}"
        )

    def parameter_count_prefix_by_layer(self) -> dict[int, int] | None:
        sources_by_layer = self.parameter_count_prefix_sources()
        if sources_by_layer is None:
            return None
        seen_parameter_ids: set[int] = set()
        cumulative_total = 0
        counts_by_layer: dict[int, int] = {}
        for layer in sorted(int(layer) for layer in sources_by_layer):
            for source in sources_by_layer[int(layer)]:
                for parameter in self._iter_parameter_source_parameters(source):
                    parameter_id = id(parameter)
                    if parameter_id in seen_parameter_ids:
                        continue
                    seen_parameter_ids.add(parameter_id)
                    cumulative_total += int(parameter.numel())
            counts_by_layer[int(layer)] = int(cumulative_total)
        return counts_by_layer

    def adapter_metadata(self) -> dict[str, Any]:
        parameter_count = self.parameter_count()
        parameter_count_prefix_by_layer = self.parameter_count_prefix_by_layer()
        return {
            "env": self.env_name,
            "encode_batch_size": int(self.default_encode_batch_size),
            "cpu_feature_cache_dtype": str(self.cpu_feature_cache_dtype).replace("torch.", ""),
            "benchmark_sequence_length": int(self.exact_benchmark_sequence_length()),
            "benchmark_sequence_length_source": str(self.benchmark_sequence_length_source),
            "input_length_policy": "exact",
            "parameter_count": None if parameter_count is None else int(parameter_count),
            "parameter_count_total": None if parameter_count is None else int(parameter_count),
            "trainable_parameter_count": self.trainable_parameter_count(),
            "parameter_count_source": (
                "torch_registered_parameters" if parameter_count is not None else "unavailable"
            ),
            "parameter_count_prefix_by_layer": (
                None
                if parameter_count_prefix_by_layer is None
                else {
                    str(int(layer)): int(count)
                    for layer, count in sorted(parameter_count_prefix_by_layer.items())
                }
            ),
            "parameter_count_prefix_source": (
                "cumulative_representation_path_unique_parameters"
                if parameter_count_prefix_by_layer is not None
                else "unavailable"
            ),
        }

    def exact_benchmark_sequence_length(self) -> int:
        value = getattr(self, "benchmark_sequence_length", None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"{self.model_name} must define a positive integer benchmark_sequence_length, got {value!r}"
            )
        return int(value)

    def validate_benchmark_input(
        self,
        x: torch.Tensor,
        *,
        channels: int | None = None,
    ) -> None:
        if x.dim() != 3:
            raise ValueError(f"{self.model_name} expects [B, C, L] input, got {tuple(x.shape)}")
        if channels is not None and int(x.shape[1]) != int(channels):
            raise ValueError(
                f"{self.model_name} expects {int(channels)} input channels, got {tuple(x.shape)}"
            )
        expected_length = self.exact_benchmark_sequence_length()
        if int(x.shape[2]) != expected_length:
            raise ValueError(
                f"{self.model_name} expects exact sequence length {expected_length}, got {tuple(x.shape)}"
            )

    def prepare(
        self,
        *,
        manifest: dict[str, Any],
        train_split: dict[str, torch.Tensor],
        val_split: dict[str, torch.Tensor],
    ) -> None:
        del manifest, train_split, val_split

    def prepare_runtime(self, *, device: torch.device) -> None:
        del device

    def update_probe_val_split(
        self,
        *,
        val_split: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return val_split

    @abstractmethod
    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        raise NotImplementedError

    def make_representation_fn(
        self,
        *,
        layers: tuple[int, ...],
        split: str = "val",
    ):
        def _representation_fn(x: torch.Tensor) -> dict[int, torch.Tensor]:
            return self.forward_layer_dict(x, layers=layers)

        return _representation_fn
