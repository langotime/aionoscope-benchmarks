from __future__ import annotations

import math

import torch
import torch.nn.functional as F


_OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def sqrt_patch_size_for_length(seq_len: int) -> int:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    return max(1, int(math.sqrt(seq_len)))


def timeseries_to_clip_images(
    x: torch.Tensor,
    *,
    patch_size: int,
    stride_fraction: float,
    image_size: int,
) -> tuple[torch.Tensor, int]:
    if x.dim() != 3:
        raise ValueError(f"Expected [B, C, T] input, got {tuple(x.shape)}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    if stride_fraction <= 0.0 or stride_fraction > 1.0:
        raise ValueError(
            f"stride_fraction must lie in (0, 1], got {stride_fraction}"
        )

    batch_size, num_channels, seq_len = x.shape
    x_t = x.to(dtype=torch.float32).transpose(1, 2)  # [B, T, C]

    median = x_t.median(dim=1, keepdim=True).values
    q75 = torch.quantile(x_t, 0.75, dim=1, keepdim=True)
    q25 = torch.quantile(x_t, 0.25, dim=1, keepdim=True)
    x_t = (x_t - median) / (q75 - q25 + 1e-5)
    x_t = x_t.transpose(1, 2)  # [B, C, T]

    if stride_fraction == 1.0:
        pad_left = 0
        remainder = seq_len % patch_size
        if remainder != 0:
            pad_left = patch_size - remainder
        x_pad = F.pad(x_t, (pad_left, 0), mode="replicate")
        num_patches = x_pad.size(-1) // patch_size
        x_2d = x_pad.reshape(batch_size, num_channels, num_patches, patch_size)
        x_2d = x_2d.transpose(-1, -2)
    else:
        stride_len = max(1, int(patch_size * stride_fraction))
        if seq_len < patch_size:
            pad_left = patch_size - seq_len
        else:
            remainder = (seq_len - patch_size) % stride_len
            pad_left = 0 if remainder == 0 else stride_len - remainder
        x_pad = F.pad(x_t, (pad_left, 0), mode="replicate")
        x_2d = x_pad.unfold(dimension=2, size=patch_size, step=stride_len)
        x_2d = x_2d.transpose(-1, -2)

    x_2d = x_2d.reshape(batch_size * num_channels, 1, x_2d.size(-2), x_2d.size(-1))
    min_vals = x_2d.amin(dim=(-2, -1), keepdim=True)
    max_vals = x_2d.amax(dim=(-2, -1), keepdim=True)
    x_2d = (x_2d - min_vals) / (max_vals - min_vals + 1e-5)
    x_2d = x_2d.pow(0.8)
    x_resized = F.interpolate(
        x_2d,
        size=(image_size, image_size),
        mode="nearest",
    )
    images = x_resized.repeat(1, 3, 1, 1)
    mean = images.new_tensor(_OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(_OPENAI_CLIP_STD).view(1, 3, 1, 1)
    images = (images - mean) / std
    return images, num_channels
