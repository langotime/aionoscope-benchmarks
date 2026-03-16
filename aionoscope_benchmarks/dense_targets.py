from __future__ import annotations

from typing import Any

import torch


def aiono_dense_targets_validate_config(*, targets_cfg: list[dict[str, object]]) -> list[str]:
    if not targets_cfg:
        raise ValueError("targets_cfg must be a non-empty list")
    target_names: list[str] = []
    for index, entry in enumerate(targets_cfg):
        if not isinstance(entry, dict):
            raise ValueError(
                "targets_cfg entries must be dicts. "
                f"Got {type(entry).__name__}: {entry!r}"
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                'targets_cfg entries must include a non-empty "name". '
                f"Entry[{index}]={entry!r}"
            )
        source = entry.get("source")
        if source not in ("process", "view"):
            raise ValueError(
                'targets_cfg entries must include "source" set to "process" or "view". '
                f"Entry[{index}]={entry!r}"
            )
        if source == "process":
            if not isinstance(entry.get("node"), str) or not str(entry.get("node")).strip():
                raise ValueError(f'Process dense targets require "node". Entry[{index}]={entry!r}')
        if source == "view":
            if not isinstance(entry.get("view"), str) or not str(entry.get("view")).strip():
                raise ValueError(f'View dense targets require "view". Entry[{index}]={entry!r}')
        if not isinstance(entry.get("param"), str) or not str(entry.get("param")).strip():
            raise ValueError(f'Dense targets require "param". Entry[{index}]={entry!r}')
        enabled_key = entry.get("enabled_key")
        if enabled_key is not None and (
            not isinstance(enabled_key, str) or not enabled_key.strip()
        ):
            raise ValueError(
                'Dense targets "enabled_key" must be a non-empty string when provided. '
                f"Entry[{index}]={entry!r}"
            )
        normalize = entry.get("normalize")
        if normalize is not None and normalize not in ("unit_interval",):
            raise ValueError(
                'Dense targets "normalize" must be one of ["unit_interval"] when provided. '
                f"Entry[{index}]={entry!r}"
            )
        target_names.append(str(name))
    if len(set(target_names)) != len(target_names):
        raise ValueError(f"targets_cfg names must be unique. Got {target_names}")
    return target_names


def aiono_dense_targets_extract(
    *,
    obs: Any,
    targets_cfg: list[dict[str, object]],
    target_names: list[str] | None = None,
) -> tuple[torch.Tensor, list[str]]:
    if target_names is None:
        target_names = aiono_dense_targets_validate_config(targets_cfg=targets_cfg)
    else:
        if len(target_names) != len(targets_cfg):
            raise ValueError(
                "target_names length must match targets_cfg length. "
                f"Got {len(target_names)} vs {len(targets_cfg)}"
            )

    x = getattr(obs, "x", None)
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"obs.x must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim != 3:
        raise ValueError(f"obs.x must be 3D [B, C, L], got {tuple(x.shape)}")
    batch_size = int(x.shape[0])
    seq_len = int(x.shape[-1])
    device = x.device

    if not isinstance(getattr(obs, "meta", None), dict):
        raise ValueError("obs.meta must be a dict")
    process_meta = obs.meta.get("process")
    if not isinstance(process_meta, dict):
        raise ValueError('obs.meta["process"] must be a dict')
    process_samples = process_meta.get("samples")
    if not isinstance(process_samples, dict):
        raise ValueError('obs.meta["process"]["samples"] must be a dict')

    target_tensors: list[torch.Tensor] = []
    for entry in targets_cfg:
        source = entry["source"]
        param = str(entry["param"])
        enabled_key = entry.get("enabled_key")
        enabled_mask = None
        if enabled_key is not None:
            enabled = process_meta.get("enabled")
            if not isinstance(enabled, dict):
                raise ValueError(
                    'Dense targets with enabled_key require obs.meta["process"]["enabled"] to be a dict'
                )
            mask = enabled.get(str(enabled_key))
            if not isinstance(mask, torch.Tensor):
                raise ValueError(
                    f"Dense target enabled mask {enabled_key!r} must be a tensor, got {type(mask).__name__}"
                )
            if mask.dtype != torch.bool:
                raise ValueError(
                    f"Dense target enabled mask {enabled_key!r} must be bool, got {mask.dtype}"
                )
            if mask.ndim != 1 or int(mask.shape[0]) != batch_size:
                raise ValueError(
                    f"Dense target enabled mask {enabled_key!r} must be [B], got {tuple(mask.shape)}"
                )
            enabled_mask = mask.to(device=device)

        allow_missing_value = enabled_mask is not None and not bool(torch.any(enabled_mask).item())

        if source == "process":
            node = str(entry["node"])
            node_samples = process_samples.get(node)
            if not isinstance(node_samples, dict):
                if allow_missing_value:
                    target_tensors.append(
                        torch.full((batch_size,), float("nan"), dtype=torch.float32, device=device)
                    )
                    continue
                raise ValueError(f"Missing process samples for node {node!r}")
            value = node_samples.get(param)
        else:
            view_name = str(entry["view"])
            try:
                view_meta = obs.view_meta(view_name)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Failed to resolve view {view_name!r}") from exc
            view_samples = view_meta.get("samples")
            if not isinstance(view_samples, dict):
                value = None
            else:
                value = view_samples.get(param)

        if value is None:
            if not allow_missing_value:
                raise ValueError(f"Missing dense target value for {entry['name']!r}")
            target_tensors.append(
                torch.full((batch_size,), float("nan"), dtype=torch.float32, device=device)
            )
            continue

        if value.ndim == 1:
            target = value
        elif value.ndim == 2 and int(value.shape[1]) == 1:
            target = value.squeeze(1)
        else:
            raise ValueError(
                f"Dense target {param!r} must be [B] or [B, 1], got {tuple(value.shape)}"
            )
        target = target.to(dtype=torch.float32, device=device)
        if entry.get("normalize") == "unit_interval":
            if seq_len <= 1:
                raise ValueError(f"seq_len must be > 1 for unit_interval normalization, got {seq_len}")
            target = target / float(seq_len - 1)
        if enabled_mask is not None:
            target = target.masked_fill(~enabled_mask, float("nan"))
        target_tensors.append(target)

    if not target_tensors:
        raise ValueError("Dense target extraction produced no targets")
    return torch.stack(target_tensors, dim=1), target_names


def aiono_basic_components_target_metric_from_param(*, target_name: str, param: str) -> str:
    if param == "time_idx" and target_name.endswith("_time_frac"):
        return "time_frac"
    return param


def aiono_basic_components_massage_target_metric(
    *, target_signal: str, target_metric: str
) -> str:
    if target_metric == "amplitude" and target_signal in ("spike", "gaussian", "level_change"):
        return "magnitude"
    if target_signal in ("uniform_noise", "gaussian_noise", "random_walk_noise"):
        return "std"
    if target_signal == "quadratic_trend" and target_metric == "b":
        return "slope"
    if target_signal == "quadratic_trend" and target_metric == "c":
        return "intercept"
    return target_metric

