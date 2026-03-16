from __future__ import annotations

"""Runtime ToyTS/Aiono split builder.

Builds the finite split online from `aiono.datasets.SynthBatchIterableDataset`
and materializes it only in process memory for the current run.
"""

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from aiono import (
    EnableComponentsNode,
    EventRenderView,
    EventSchema,
    GateEventsByEnabledNode,
    LinearTrendView,
    LogTrendView,
    ProcessGraph,
    QuadraticTrendView,
    ResolvedToyTSBasicComponentsPeriodicContract,
    SawtoothWaveView,
    SigmoidTrendView,
    SineWaveView,
    SingleEventNode,
    SquareWaveView,
    SynthPipeline,
    UnionEventsNode,
    UniformSampler,
    ViewChain,
    ToyTSBasicComponentsPeriodicConfig,
    resolve_toyts_basic_components_periodic_contract,
)
from aiono.datasets import SynthBatchIterableDataset
from aiono.processes.constant import ConstantLatentNode
from aiono.views.noise import GaussianNoiseView, RandomWalkNoiseView, UniformNoiseView

from .constants import DATASET_CONFIG_PATH
from .dense_targets import (
    toyts_basic_components_massage_target_metric,
    toyts_basic_components_target_metric_from_param,
    toyts_dense_targets_extract,
    toyts_dense_targets_validate_config,
)


@dataclass(frozen=True)
class DenseTargetSpec:
    name: str
    signal: str
    metric: str


@dataclass(frozen=True)
class DatasetManifest:
    dataset_name: str
    benchmark_family: str
    benchmark_version: str
    view_name: str
    baseline_sampling_frequency_hz: int
    sampling_frequency: int
    channels: list[str]
    default_channel_size: int
    channel_size: int
    channel_size_policy: str
    channel_size_source: str
    train_seed: int
    validation_seed_values: list[int]
    validation_seed_offset: int
    validation_generator_seeds: list[int]
    validation_seed_to_generator_seed: dict[str, int]
    validation_seed_count: int
    batch_size: int
    train_batches: int
    val_batches: int
    num_enabled: int
    component_keys: list[str]
    class_names: list[str]
    group_to_classes: dict[str, list[str]]
    dense_target_names: list[str]
    dense_targets: list[dict[str, str]]
    duration_sec: float
    periodic_frequency_mode: str
    periodic_frequency_resolution_source: str
    periodic_frequency_min_full_periods: float
    periodic_frequency_nyquist_fraction: float
    sine_recoverability_policy: str
    sine_frequency_hz_resolved_low: float
    sine_frequency_hz_resolved_high: float
    sawtooth_recoverability_policy: str
    sawtooth_frequency_hz_resolved_low: float
    sawtooth_frequency_hz_resolved_high: float
    square_recoverability_policy: str
    square_frequency_hz_resolved_low: float
    square_frequency_hz_resolved_high: float
    sawtooth_min_points_per_period: int
    square_min_points_in_shorter_plateau: int
    square_duty_cycle_min: float
    square_duty_cycle_max: float
    square_shorter_plateau_fraction_min: float
    square_frequency_hz_recoverability_upper_bound: float
    periodic_sampler_specs: dict[str, dict[str, dict[str, object]]]
    config_sha256: str
    created_at_unix: float
    generator_device: str
    generation_mode: str


def _load_dataset_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected top-level YAML dict, got {type(cfg).__name__}")
    return cfg


def _require_dict(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a dict, got {type(value).__name__}")
    return value


def _require_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an int, got {type(value).__name__}")
    return int(value)


def _require_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string, got {value!r}")
    return value


def _require_int_list(mapping: dict[str, Any], key: str) -> list[int]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list of ints, got {value!r}")
    out: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise ValueError(f"{key} must contain only ints, got {item!r}")
        out.append(int(item))
    if len(set(out)) != len(out):
        raise ValueError(f"{key} must not contain duplicates, got {out}")
    return out


def _normalize_int_values(values: list[int] | tuple[int, ...], *, name: str) -> list[int]:
    out = [int(value) for value in values]
    if not out:
        raise ValueError(f"{name} must be non-empty")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} must not contain duplicates, got {out}")
    return out


def _resolve_validation_seed_config(
    *,
    toyts: dict[str, Any],
    train_seed: int,
) -> tuple[list[int], int, list[int], dict[str, int]]:
    if "validation_seed_values" in toyts:
        validation_seed_values = _require_int_list(toyts, "validation_seed_values")
        validation_seed_offset = int(toyts.get("validation_seed_offset", 0))
    elif "val_seed" in toyts:
        validation_seed_values = [_require_int(toyts, "val_seed")]
        validation_seed_offset = int(toyts.get("validation_seed_offset", 0))
    else:
        raise ValueError("toyts must define either validation_seed_values or val_seed")

    validation_generator_seeds = [
        int(validation_seed_offset) + int(seed_value) for seed_value in validation_seed_values
    ]
    if int(train_seed) in validation_generator_seeds:
        raise ValueError(
            "Train seed overlaps with at least one validation generator seed. "
            "Use validation_seed_offset (for example 100) to separate the ranges. "
            f"train_seed={int(train_seed)} validation_generator_seeds={validation_generator_seeds}"
        )
    validation_seed_to_generator_seed = {
        str(seed_value): int(generator_seed)
        for seed_value, generator_seed in zip(
            validation_seed_values,
            validation_generator_seeds,
            strict=True,
        )
    }
    return (
        validation_seed_values,
        int(validation_seed_offset),
        validation_generator_seeds,
        validation_seed_to_generator_seed,
    )


def _parse_manifest_components(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    toyts = _require_dict(cfg, "toyts")
    dense_probe = _require_dict(toyts, "dense_probe")
    basic = _require_dict(toyts, "basic_components")
    component_keys = basic.get("component_keys")
    if not isinstance(component_keys, list) or not component_keys:
        raise ValueError("toyts.basic_components.component_keys must be a non-empty list")
    if not all(isinstance(value, str) and value.strip() for value in component_keys):
        raise ValueError("component_keys must contain non-empty strings")
    dense_targets_cfg = dense_probe.get("targets")
    if not isinstance(dense_targets_cfg, list) or not dense_targets_cfg:
        raise ValueError("toyts.dense_probe.targets must be a non-empty list")
    return toyts, [str(value) for value in component_keys], dense_targets_cfg


def _build_group_to_classes(component_keys: list[str]) -> dict[str, list[str]]:
    group_specs = {
        "noise": {"gaussian_noise", "uniform_noise", "random_walk_noise"},
        "trend": {
            "linear_trend",
            "quadratic_trend",
            "log_trend",
            "sigmoid_trend",
            "piecewise_linear_trend",
        },
        "periodic": {"sine", "sawtooth", "square"},
        "events": {"spike", "level_change", "gaussian"},
    }
    group_to_classes: dict[str, list[str]] = {}
    for group_name, group_keys in group_specs.items():
        group_classes = [key for key in component_keys if key in group_keys]
        if group_classes:
            group_to_classes[group_name] = group_classes
    group_to_classes["all"] = list(component_keys)
    return group_to_classes


def _build_dense_target_specs(targets_cfg: list[dict[str, Any]]) -> list[DenseTargetSpec]:
    out: list[DenseTargetSpec] = []
    for item in targets_cfg:
        enabled_key = str(item["enabled_key"])
        name = str(item["name"])
        param = str(item["param"])
        metric_raw = toyts_basic_components_target_metric_from_param(
            target_name=name,
            param=param,
        )
        metric = toyts_basic_components_massage_target_metric(
            target_signal=enabled_key,
            target_metric=metric_raw,
        )
        out.append(DenseTargetSpec(name=name, signal=enabled_key, metric=metric))
    return out


def _build_basic_components_pipeline(
    *,
    seq_len: int,
    sample_rate_hz: float,
    view_name: str,
    component_keys: list[str],
    num_enabled: int,
    periodic_contract: ResolvedToyTSBasicComponentsPeriodicContract,
    device: torch.device,
) -> SynthPipeline:
    schema = EventSchema(
        type_names=["spike", "level_change", "gaussian"],
        param_names=["amplitude", "sigma_sec"],
        time_unit="samples",
    )
    event_time_min = int(seq_len * 0.15)
    event_time_max = int(seq_len * 0.85)

    process_nodes: list[torch.nn.Module] = [
        EnableComponentsNode(component_keys=component_keys, num_enabled=int(num_enabled)),
        ConstantLatentNode(
            seq_len=seq_len,
            channels=1,
            value=UniformSampler(-1.0, 1.0),
            enabled_key="constant",
            out_key="latent",
        ),
    ]
    event_in_keys: list[str] = []
    if "spike" in component_keys:
        process_nodes.extend(
            [
                SingleEventNode(
                    seq_len=seq_len,
                    schema=schema,
                    type_name="spike",
                    time_min=event_time_min,
                    time_max=event_time_max,
                    amplitude=UniformSampler(0.8, 1.2),
                    amplitude_param="amplitude",
                    out_key="spike",
                ),
                GateEventsByEnabledNode(in_key="spike", enabled_key="spike", out_key="spike.gated"),
            ]
        )
        event_in_keys.append("spike.gated")
    if "level_change" in component_keys:
        process_nodes.extend(
            [
                SingleEventNode(
                    seq_len=seq_len,
                    schema=schema,
                    type_name="level_change",
                    time_min=event_time_min,
                    time_max=event_time_max,
                    amplitude=UniformSampler(-1.0, 1.0),
                    amplitude_param="amplitude",
                    out_key="level_change",
                ),
                GateEventsByEnabledNode(
                    in_key="level_change",
                    enabled_key="level_change",
                    out_key="level_change.gated",
                ),
            ]
        )
        event_in_keys.append("level_change.gated")
    if "gaussian" in component_keys:
        process_nodes.extend(
            [
                SingleEventNode(
                    seq_len=seq_len,
                    schema=schema,
                    type_name="gaussian",
                    time_min=event_time_min,
                    time_max=event_time_max,
                    amplitude=UniformSampler(-1.0, 1.0),
                    amplitude_param="amplitude",
                    extra_params={"sigma_sec": UniformSampler(0.01, 0.06)},
                    out_key="gaussian",
                ),
                GateEventsByEnabledNode(
                    in_key="gaussian",
                    enabled_key="gaussian",
                    out_key="gaussian.gated",
                ),
            ]
        )
        event_in_keys.append("gaussian.gated")

    outputs = {"latent"}
    if event_in_keys:
        process_nodes.append(UnionEventsNode(in_keys=event_in_keys, out_key="events"))
        outputs.add("events")
    process = ProcessGraph(
        name="BasicComponentsProcess",
        outputs=outputs,
        base_meta={
            "seq_len": seq_len,
            "sample_rate_hz": sample_rate_hz,
            "component_keys": list(component_keys),
            "num_enabled": int(num_enabled),
        },
        graph=process_nodes,
    )

    views: list[torch.nn.Module] = []
    if event_in_keys:
        views.append(
            EventRenderView(
                seq_len=seq_len,
                amplitude_param="amplitude",
                rounding="nearest",
                sigma_sec_param="sigma_sec",
            )
        )
    if "gaussian_noise" in component_keys:
        views.append(GaussianNoiseView(noise_std=UniformSampler(0.02, 0.15), enabled_key="gaussian_noise"))
    if "uniform_noise" in component_keys:
        views.append(UniformNoiseView(amplitude=UniformSampler(0.05, 0.25), enabled_key="uniform_noise"))
    if "random_walk_noise" in component_keys:
        views.append(RandomWalkNoiseView(step_std=UniformSampler(0.01, 0.08), enabled_key="random_walk_noise"))
    if "linear_trend" in component_keys:
        views.append(
            LinearTrendView(
                seq_len=seq_len,
                slope=UniformSampler(-2.0, 2.0),
                intercept=UniformSampler(-0.5, 0.5),
                enabled_key="linear_trend",
            )
        )
    if "quadratic_trend" in component_keys:
        views.append(
            QuadraticTrendView(
                seq_len=seq_len,
                a=UniformSampler(-4.0, 4.0),
                b=UniformSampler(-2.0, 2.0),
                c=UniformSampler(-0.5, 0.5),
                enabled_key="quadratic_trend",
            )
        )
    if "log_trend" in component_keys:
        views.append(
            LogTrendView(
                seq_len=seq_len,
                amplitude=UniformSampler(-2.0, 2.0),
                offset=UniformSampler(-0.5, 0.5),
                epsilon=1e-3,
                enabled_key="log_trend",
            )
        )
    if "sigmoid_trend" in component_keys:
        views.append(
            SigmoidTrendView(
                seq_len=seq_len,
                amplitude=UniformSampler(-2.0, 2.0),
                center=UniformSampler(0.2, 0.8),
                sharpness=UniformSampler(5.0, 20.0),
                offset=UniformSampler(-0.5, 0.5),
                enabled_key="sigmoid_trend",
            )
        )
    if "sine" in component_keys:
        views.append(
            SineWaveView(
                seq_len=seq_len,
                **periodic_contract.signal("sine").view_kwargs(),
                enabled_key="sine",
            )
        )
    if "sawtooth" in component_keys:
        views.append(
            SawtoothWaveView(
                seq_len=seq_len,
                **periodic_contract.signal("sawtooth").view_kwargs(),
                enabled_key="sawtooth",
            )
        )
    if "square" in component_keys:
        views.append(
            SquareWaveView(
                seq_len=seq_len,
                **periodic_contract.signal("square").view_kwargs(),
                enabled_key="square",
            )
        )
    if not views:
        raise ValueError("At least one non-constant component is required to produce a view")
    view = ViewChain(*views)
    return SynthPipeline(process=process, views={view_name: view}).to(device)


def _extract_class_targets(
    *, obs: Any, class_names: list[str], device: torch.device
) -> torch.Tensor:
    process_meta = obs.meta.get("process")
    if not isinstance(process_meta, dict):
        raise ValueError('obs.meta["process"] must be a dict')
    enabled = process_meta.get("enabled")
    if not isinstance(enabled, dict):
        raise ValueError('obs.meta["process"]["enabled"] must be a dict')
    enabled_batches: list[torch.Tensor] = []
    for class_name in class_names:
        mask = enabled.get(class_name)
        if not isinstance(mask, torch.Tensor):
            raise ValueError(f"Missing enabled mask for {class_name!r}")
        if mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError(f"Enabled mask {class_name!r} must be bool [B], got {mask.dtype} {tuple(mask.shape)}")
        enabled_batches.append(mask.to(dtype=torch.float32, device=device))
    return torch.stack(enabled_batches, dim=1)


def _generate_split(
    *,
    pipeline: SynthPipeline,
    split_name: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    num_batches: int,
    view_name: str,
    class_names: list[str],
    dense_targets_cfg: list[dict[str, Any]],
    dense_target_names: list[str],
    seq_len: int,
    show_progress_bar: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, torch.Tensor]:
    split_start = time.perf_counter()
    num_samples = batch_size * num_batches
    x_all = torch.empty((num_samples, 1, seq_len), dtype=torch.float32)
    y_cls_all = torch.empty((num_samples, len(class_names)), dtype=torch.float32)
    y_dense_all = torch.empty((num_samples, len(dense_target_names)), dtype=torch.float32)

    dataset = SynthBatchIterableDataset(
        pipeline=pipeline,
        batch_size=batch_size,
        device=device,
        seed=seed,
        max_batches=num_batches,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=None,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )
    iterator = (
        tqdm(loader, total=num_batches, desc=f"{split_name} online", leave=False)
        if show_progress_bar
        else loader
    )
    for batch_index, views in enumerate(iterator):
        if not isinstance(views, dict):
            raise ValueError(
                "Aiono online dataset must yield dict[str, Observation], "
                f"got {type(views).__name__}"
            )
        obs = views.get(view_name)
        if obs is None:
            raise ValueError(f"Missing view {view_name!r} in pipeline output")
        start = batch_index * batch_size
        end = start + batch_size
        x_all[start:end] = obs.x.detach().to("cpu")
        y_cls = _extract_class_targets(obs=obs, class_names=class_names, device=device)
        y_dense, _ = toyts_dense_targets_extract(
            obs=obs,
            targets_cfg=dense_targets_cfg,
            target_names=dense_target_names,
        )
        y_cls_all[start:end] = y_cls.detach().to("cpu")
        y_dense_all[start:end] = y_dense.detach().to("cpu")
    if progress_callback is not None:
        progress_callback(
            f"{split_name}: generated {num_samples} samples in {time.perf_counter() - split_start:.1f}s"
        )
    return {"x": x_all, "y_cls": y_cls_all, "y_dense": y_dense_all}


def build_runtime_splits_by_validation_seed(
    *,
    config_path: Path = DATASET_CONFIG_PATH,
    device: torch.device | None = None,
    batch_size: int = 256,
    channel_size_override: int | None = None,
    channel_size_policy_override: str | None = None,
    channel_size_source_override: str | None = None,
    train_batches: int | None = None,
    val_batches: int | None = None,
    validation_seed_values: list[int] | None = None,
    validation_seed_offset: int | None = None,
    show_progress_bar: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    cfg_text = config_path.read_text(encoding="utf-8")
    cfg = _load_dataset_config(config_path)
    toyts, component_keys, dense_targets_cfg = _parse_manifest_components(cfg)
    dense_target_names = toyts_dense_targets_validate_config(targets_cfg=dense_targets_cfg)
    dense_target_specs = _build_dense_target_specs(dense_targets_cfg)

    sampling_frequency = _require_int(cfg, "sampling_frequency")
    benchmark_family = _require_str(cfg, "benchmark_family")
    benchmark_version = _require_str(cfg, "benchmark_version")
    default_channel_size = _require_int(cfg, "default_channel_size")
    channel_size_policy = str(cfg.get("channel_size_policy", "fixed_config")).strip()
    if not channel_size_policy:
        raise ValueError("channel_size_policy must be a non-empty string")
    resolved_channel_size = int(channel_size_override) if channel_size_override is not None else int(default_channel_size)
    if resolved_channel_size <= 0:
        raise ValueError(
            f"resolved channel_size must be > 0, got {resolved_channel_size}"
        )
    resolved_channel_size_policy = (
        str(channel_size_policy_override).strip()
        if channel_size_policy_override is not None
        else channel_size_policy
    )
    if not resolved_channel_size_policy:
        raise ValueError("resolved channel_size_policy must be a non-empty string")
    resolved_channel_size_source = (
        str(channel_size_source_override).strip()
        if channel_size_source_override is not None
        else ("runtime.channel_size_override" if channel_size_override is not None else "config.default_channel_size")
    )
    if not resolved_channel_size_source:
        raise ValueError("resolved channel_size_source must be a non-empty string")
    channels = cfg.get("channels")
    if not isinstance(channels, list) or not channels:
        raise ValueError("channels must be a non-empty list")
    train_seed = _require_int(toyts, "train_seed")
    (
        resolved_validation_seed_values,
        resolved_validation_seed_offset,
        resolved_validation_generator_seeds,
        resolved_validation_seed_to_generator_seed,
    ) = _resolve_validation_seed_config(
        toyts=toyts,
        train_seed=int(train_seed),
    )
    if validation_seed_values is not None:
        resolved_validation_seed_values = _normalize_int_values(
            validation_seed_values,
            name="validation_seed_values",
        )
    if validation_seed_offset is not None:
        resolved_validation_seed_offset = int(validation_seed_offset)
    resolved_validation_generator_seeds = [
        int(resolved_validation_seed_offset) + int(seed_value)
        for seed_value in resolved_validation_seed_values
    ]
    if int(train_seed) in resolved_validation_generator_seeds:
        raise ValueError(
            "Train seed overlaps with at least one validation generator seed after overrides. "
            "Use validation_seed_offset (for example 100) to separate the ranges. "
            f"train_seed={int(train_seed)} validation_generator_seeds={resolved_validation_generator_seeds}"
        )
    resolved_validation_seed_to_generator_seed = {
        str(seed_value): int(generator_seed)
        for seed_value, generator_seed in zip(
            resolved_validation_seed_values,
            resolved_validation_generator_seeds,
            strict=True,
        )
    }
    resolved_train_batches = (
        int(train_batches)
        if train_batches is not None
        else _require_int(toyts, "offline_probe_train_batches")
    )
    resolved_val_batches = (
        int(val_batches)
        if val_batches is not None
        else _require_int(toyts, "offline_probe_val_batches")
    )
    if resolved_train_batches <= 0 or resolved_val_batches <= 0:
        raise ValueError(
            "train_batches and val_batches must be > 0, "
            f"got train_batches={resolved_train_batches} val_batches={resolved_val_batches}"
        )
    view_name = _require_str(toyts, "view_name")
    basic = _require_dict(toyts, "basic_components")
    num_enabled = _require_int(basic, "num_enabled")
    periodic_cfg = _require_dict(basic, "periodic")
    periodic_contract = resolve_toyts_basic_components_periodic_contract(
        seq_len=int(resolved_channel_size),
        sampling_frequency_hz=int(sampling_frequency),
        config=ToyTSBasicComponentsPeriodicConfig.from_mapping(periodic_cfg),
        benchmark_family=benchmark_family,
        benchmark_version=benchmark_version,
    )
    config_sha256 = hashlib.sha256(cfg_text.encode("utf-8")).hexdigest()
    group_to_classes = _build_group_to_classes(component_keys)
    actual_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = DatasetManifest(
        dataset_name="toyts_basic_components_balanced",
        **periodic_contract.manifest_fields(),
        view_name=view_name,
        sampling_frequency=int(sampling_frequency),
        channels=[str(channel) for channel in channels],
        default_channel_size=int(default_channel_size),
        channel_size=int(resolved_channel_size),
        channel_size_policy=resolved_channel_size_policy,
        channel_size_source=resolved_channel_size_source,
        train_seed=int(train_seed),
        validation_seed_values=list(resolved_validation_seed_values),
        validation_seed_offset=int(resolved_validation_seed_offset),
        validation_generator_seeds=list(resolved_validation_generator_seeds),
        validation_seed_to_generator_seed=dict(resolved_validation_seed_to_generator_seed),
        validation_seed_count=int(len(resolved_validation_seed_values)),
        batch_size=int(batch_size),
        train_batches=int(resolved_train_batches),
        val_batches=int(resolved_val_batches),
        num_enabled=int(num_enabled),
        component_keys=list(component_keys),
        class_names=list(component_keys),
        group_to_classes=group_to_classes,
        dense_target_names=list(dense_target_names),
        dense_targets=[asdict(spec) for spec in dense_target_specs],
        config_sha256=config_sha256,
        created_at_unix=time.time(),
        generator_device=str(actual_device),
        generation_mode="online_iterable_materialized",
    )

    pipeline = _build_basic_components_pipeline(
        seq_len=int(resolved_channel_size),
        sample_rate_hz=float(sampling_frequency),
        view_name=view_name,
        component_keys=component_keys,
        num_enabled=int(num_enabled),
        periodic_contract=periodic_contract,
        device=actual_device,
    )
    train = _generate_split(
        pipeline=pipeline,
        split_name="train",
        seed=int(train_seed),
        device=actual_device,
        batch_size=int(batch_size),
        num_batches=int(resolved_train_batches),
        view_name=view_name,
        class_names=component_keys,
        dense_targets_cfg=dense_targets_cfg,
        dense_target_names=dense_target_names,
        seq_len=int(resolved_channel_size),
        show_progress_bar=show_progress_bar,
        progress_callback=progress_callback,
    )
    val_splits: dict[int, dict[str, torch.Tensor]] = {}
    for seed_value, generator_seed in zip(
        resolved_validation_seed_values,
        resolved_validation_generator_seeds,
        strict=True,
    ):
        val_splits[int(seed_value)] = _generate_split(
            pipeline=pipeline,
            split_name=f"val[{int(seed_value)}->g{int(generator_seed)}]",
            seed=int(generator_seed),
            device=actual_device,
            batch_size=int(batch_size),
            num_batches=int(resolved_val_batches),
            view_name=view_name,
            class_names=component_keys,
            dense_targets_cfg=dense_targets_cfg,
            dense_target_names=dense_target_names,
            seq_len=int(resolved_channel_size),
            show_progress_bar=show_progress_bar,
            progress_callback=progress_callback,
        )
    return asdict(manifest), train, val_splits


def build_runtime_splits(
    *,
    config_path: Path = DATASET_CONFIG_PATH,
    device: torch.device | None = None,
    batch_size: int = 256,
    channel_size_override: int | None = None,
    channel_size_policy_override: str | None = None,
    channel_size_source_override: str | None = None,
    train_batches: int | None = None,
    val_batches: int | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    manifest, train, val_splits = build_runtime_splits_by_validation_seed(
        config_path=config_path,
        device=device,
        batch_size=batch_size,
        channel_size_override=channel_size_override,
        channel_size_policy_override=channel_size_policy_override,
        channel_size_source_override=channel_size_source_override,
        train_batches=train_batches,
        val_batches=val_batches,
        show_progress_bar=False,
    )
    first_seed_value = int(manifest["validation_seed_values"][0])
    return manifest, train, val_splits[first_seed_value]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DATASET_CONFIG_PATH,
        help="Dataset config YAML",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Generation device",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Probe batch size used to materialize the finite runtime split",
    )
    parser.add_argument(
        "--channel-size",
        type=int,
        default=None,
        help="Optional exact sequence length override for online dataset generation",
    )
    parser.add_argument(
        "--train-batches",
        type=int,
        default=None,
        help="Optional override for train batch count",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=None,
        help="Optional override for val batch count",
    )
    parser.add_argument(
        "--validation-seed-value",
        action="append",
        dest="validation_seed_values",
        type=int,
        default=None,
        help="Optional validation seed value override; can be repeated",
    )
    parser.add_argument(
        "--validation-seed-offset",
        type=int,
        default=None,
        help="Optional validation generator-seed offset override",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest, train, val_splits = build_runtime_splits_by_validation_seed(
        config_path=args.config,
        device=torch.device(str(args.device)),
        batch_size=int(args.batch_size),
        channel_size_override=args.channel_size,
        channel_size_policy_override=("cli_exact_override" if args.channel_size is not None else None),
        channel_size_source_override=("runtime_dataset_cli" if args.channel_size is not None else None),
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        validation_seed_values=args.validation_seed_values,
        validation_seed_offset=args.validation_seed_offset,
        show_progress_bar=True,
    )
    summary = {
        "manifest": manifest,
        "train": {key: list(value.shape) for key, value in train.items()},
        "val_by_seed": {
            str(seed_value): {key: list(value.shape) for key, value in split.items()}
            for seed_value, split in sorted(val_splits.items())
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
