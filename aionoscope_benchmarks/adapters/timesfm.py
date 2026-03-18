from __future__ import annotations

import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from .base import FrozenTimeSeriesAdapter


class TimesFM25Adapter(FrozenTimeSeriesAdapter):
    model_name = "TimesFM-2.5-200M"
    model_slug = "TimesFM-2.5-200M"
    source = "https://github.com/google-research/timesfm"
    checkpoint = "google/timesfm-2.5-200m-pytorch"
    import_path = "timesfm repo"
    env_name = "core"
    default_encode_batch_size = 8
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        repo_root = Path(__file__).resolve().parents[2]
        self.timesfm_src_root = repo_root / "external" / "timesfm" / "src"
        if not self.timesfm_src_root.is_dir():
            raise FileNotFoundError(f"Expected TimesFM repo sources at {self.timesfm_src_root}")
        sys.path.insert(0, str(self.timesfm_src_root))

        from timesfm import TimesFM_2p5_200M_torch
        from timesfm.torch import util as timesfm_util

        self.timesfm_util = timesfm_util
        weights_filename = getattr(TimesFM_2p5_200M_torch, "WEIGHTS_FILENAME", None)
        if isinstance(weights_filename, str):
            checkpoint_path = hf_hub_download(
                repo_id=self.checkpoint,
                filename=weights_filename,
            )
            wrapper = TimesFM_2p5_200M_torch(torch_compile=False)
            wrapper.model.load_checkpoint(checkpoint_path, torch_compile=False)
            self.model = wrapper.model
        else:
            self.model = TimesFM_2p5_200M_torch.from_pretrained(
                self.checkpoint,
                torch_compile=False,
            ).model
        self.model.eval()
        self.patch_size = int(self.model.p)
        self.output_patch_size = int(self.model.o)
        self.output_quantile_len = int(self.model.os)
        self.hidden_size = int(self.model.md)
        self.num_hidden_layers = int(self.model.x)
        self.benchmark_sequence_length = int(self.model.config.context_limit)
        self.benchmark_sequence_length_source = "timesfm_2p5_config.context_limit"

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_hidden_layers + 1))

    def parameter_count_prefix_sources(self) -> dict[int, tuple[object, ...]]:
        return {
            0: (self.model.tokenizer,),
            **{
                int(layer_index): (layer,)
                for layer_index, layer in enumerate(self.model.stacked_xf, start=1)
            },
        }

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["patch_size"] = int(self.patch_size)
        payload["output_patch_size"] = int(self.output_patch_size)
        payload["output_quantile_len"] = int(self.output_quantile_len)
        payload["hidden_size"] = int(self.hidden_size)
        payload["num_hidden_layers"] = int(self.num_hidden_layers)
        payload["preprocess"] = (
            "reshape the exact context into non-overlapping patches; apply the official running-stat ReVIN prefill "
            "normalization used by TimesFM decode; mean-pool patch tokens"
        )
        payload["layer_layout"] = (
            "layer 0 is the tokenizer output embedding stream; "
            "layers 1..N are the stacked transformer outputs"
        )
        return payload

    def _prepare_patched_context(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate_benchmark_input(x, channels=1)
        context = x[:, 0, :].to(dtype=torch.float32)
        if context.shape[-1] % self.patch_size != 0:
            raise ValueError(
                f"{self.model_name} expects context length divisible by patch size {self.patch_size}, "
                f"got {tuple(context.shape)}"
            )
        patched_inputs = context.reshape(context.size(0), -1, self.patch_size)
        patched_masks = torch.zeros_like(patched_inputs, dtype=torch.bool)
        return patched_inputs, patched_masks

    def _running_patch_stats(
        self,
        patched_inputs: torch.Tensor,
        patched_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(patched_masks.any().item()):
            flat_inputs = patched_inputs.reshape(patched_inputs.size(0), -1)
            cumulative_sum = flat_inputs.cumsum(dim=-1)
            cumulative_sum_sq = flat_inputs.square().cumsum(dim=-1)
            patch_end_indices = torch.arange(
                self.patch_size - 1,
                flat_inputs.size(-1),
                self.patch_size,
                device=flat_inputs.device,
            )
            counts = (patch_end_indices + 1).to(dtype=flat_inputs.dtype)
            patch_mu = cumulative_sum.index_select(dim=1, index=patch_end_indices) / counts.unsqueeze(0)
            patch_var = (
                cumulative_sum_sq.index_select(dim=1, index=patch_end_indices) / counts.unsqueeze(0)
                - patch_mu.square()
            )
            patch_sigma = torch.sqrt(torch.clamp(patch_var, min=0.0))
            return patch_mu, patch_sigma

        batch_size, num_patches, _ = patched_inputs.shape
        n = torch.zeros(batch_size, device=patched_inputs.device)
        mu = torch.zeros(batch_size, device=patched_inputs.device)
        sigma = torch.zeros(batch_size, device=patched_inputs.device)
        patch_mu: list[torch.Tensor] = []
        patch_sigma: list[torch.Tensor] = []
        for patch_index in range(num_patches):
            (n, mu, sigma), _ = self.timesfm_util.update_running_stats(
                n,
                mu,
                sigma,
                patched_inputs[:, patch_index],
                patched_masks[:, patch_index],
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        return torch.stack(patch_mu, dim=1), torch.stack(patch_sigma, dim=1)

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = set(int(layer) for layer in (layers or self.available_layers))
        patched_inputs, patched_masks = self._prepare_patched_context(x)
        patch_mu, patch_sigma = self._running_patch_stats(patched_inputs, patched_masks)
        normalized_inputs = self.timesfm_util.revin(
            patched_inputs,
            patch_mu,
            patch_sigma,
            reverse=False,
        )
        normalized_inputs = torch.where(patched_masks, 0.0, normalized_inputs)

        tokenizer_inputs = torch.cat([normalized_inputs, patched_masks.to(normalized_inputs.dtype)], dim=-1)
        hidden_states = self.model.tokenizer(tokenizer_inputs)

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            reps[0] = hidden_states.mean(dim=1).float()
        for layer_index, layer in enumerate(self.model.stacked_xf, start=1):
            hidden_states, _ = layer(hidden_states, patched_masks[..., -1], None)
            if layer_index in requested_layers:
                reps[int(layer_index)] = hidden_states.mean(dim=1).float()
        return reps
