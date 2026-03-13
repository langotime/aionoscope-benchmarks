from __future__ import annotations

from abc import ABC, abstractmethod
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

    def adapter_metadata(self) -> dict[str, Any]:
        return {
            "env": self.env_name,
            "encode_batch_size": int(self.default_encode_batch_size),
        }

    def prepare(
        self,
        *,
        manifest: dict[str, Any],
        train_split: dict[str, torch.Tensor],
        val_split: dict[str, torch.Tensor],
    ) -> None:
        del manifest, train_split, val_split

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
