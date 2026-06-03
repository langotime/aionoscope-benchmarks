from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from aiono import (
    AionoBasicComponentsPeriodicConfig,
    EnableComponentsNode,
    EventRenderView,
    EventSchema,
    GateEventsByEnabledNode,
    GaussianNoiseView,
    LinearTrendView,
    LogTrendView,
    ProcessGraph,
    QuadraticTrendView,
    RandomWalkNoiseView,
    ResolvedAionoBasicComponentsPeriodicContract,
    SawtoothWaveView,
    SigmoidTrendView,
    SineWaveView,
    SquareWaveView,
    SynthPipeline,
    UniformNoiseView,
    ViewChain,
    resolve_aiono_basic_components_periodic_contract,
)
from aiono.core.events import EventBatch
from aiono.core.samplers import ConstantSampler, Sampler, sampler_from_value, sampler_sample
from aiono.core.utils import SAMPLES_PREFIX
from aiono.datasets import SynthBatchIterableDataset
from aiono.processes.constant import ConstantLatentNode
from aiono.processes.graph import ProcessNode, ProcessState

from .constants import DATASET_CONFIG_PATH
from .dense_targets import aiono_dense_targets_extract, aiono_dense_targets_validate_config
from .manifold_config import ManifoldTargetGeometry
from .runtime_dataset import (
    _build_dense_target_specs,
    _build_group_to_classes,
    _extract_class_targets,
    _load_dataset_config,
    _parse_manifest_components,
    _require_dict,
    _require_int,
    _require_str,
)


_EVENT_COMPONENTS = {"spike", "level_change", "gaussian"}


@dataclass(frozen=True)
class ControlledSlice:
    split: dict[str, torch.Tensor]
    manifest: dict[str, Any]
    dense_target_names: list[str]
    target_index: int
    grid_index: torch.Tensor
    latent_coordinates: torch.Tensor
    physical_values: torch.Tensor


@dataclass(frozen=True)
class _ResolvedTarget:
    geometry: ManifoldTargetGeometry
    target_cfg: dict[str, Any]
    physical_low: float
    physical_high: float
    physical_grid_dtype: str
    grid_mode: str
    range_policy: str
    log_min_abs: float


class SequenceSampler(Sampler):
    """Return a fixed per-sample vector for one controlled generation batch."""

    def __init__(self, values: torch.Tensor | list[float] | list[int]) -> None:
        values_tensor = torch.as_tensor(values)
        if values_tensor.ndim != 1:
            raise ValueError(f"SequenceSampler values must be 1D, got {tuple(values_tensor.shape)}")
        if int(values_tensor.numel()) < 1:
            raise ValueError("SequenceSampler values must be non-empty")
        self.values = values_tensor.detach().cpu()

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del rng
        if len(shape) != 1:
            raise ValueError(f"SequenceSampler only supports 1D shape, got {shape}")
        if int(shape[0]) != int(self.values.numel()):
            raise ValueError(
                "SequenceSampler shape must match values length. "
                f"shape={shape} values={int(self.values.numel())}"
            )
        return self.values.to(device=device, dtype=dtype)

    def spec(self) -> dict[str, Any]:
        finite = self.values.to(dtype=torch.float64)
        return {
            "kind": "sequence",
            "n": int(self.values.numel()),
            "first": float(finite[0].item()),
            "last": float(finite[-1].item()),
        }


class ControlledSingleEventNode(ProcessNode):
    """Generate one event per sample with controlled time and parameters."""

    def __init__(
        self,
        *,
        seq_len: int,
        schema: EventSchema,
        type_name: str,
        time_idx: Sampler | float | int,
        amplitude: Sampler | float | int,
        amplitude_param: str,
        extra_params: dict[str, Sampler | float | int] | None,
        out_key: str,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        self.seq_len = int(seq_len)
        self.schema = schema
        self.type_name = str(type_name)
        self.type_id = int(schema.type_id(type_name))
        self.time_idx_sampler = sampler_from_value(time_idx, name="time_idx")
        self.amplitude_sampler = sampler_from_value(amplitude, name="amplitude")
        self.amplitude_index = int(schema.param_id(amplitude_param))
        self.extra_param_samplers = {
            str(name): sampler_from_value(value, name=str(name))
            for name, value in (extra_params or {}).items()
        }
        for name in self.extra_param_samplers:
            schema.param_id(name)
        self.out_key = str(out_key)

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        self._record_seed(state, rng)
        batch_size = int(state.batch_size)
        times_idx = sampler_sample(
            sampler=self.time_idx_sampler,
            shape=(batch_size,),
            rng=rng,
            device=state.device,
            dtype=torch.float32,
            name="time_idx",
        ).round()
        if torch.any(times_idx < 0) or torch.any(times_idx > self.seq_len - 1):
            raise ValueError("ControlledSingleEventNode time_idx is outside the sequence")

        params = torch.zeros(
            (batch_size, 1, len(self.schema.param_names)),
            device=state.device,
            dtype=torch.float32,
        )
        amplitude = sampler_sample(
            sampler=self.amplitude_sampler,
            shape=(batch_size,),
            rng=rng,
            device=state.device,
            dtype=torch.float32,
            name="amplitude",
        )
        params[:, 0, self.amplitude_index] = amplitude
        extra_values: dict[str, torch.Tensor] = {}
        for name, sampler in self.extra_param_samplers.items():
            value = sampler_sample(
                sampler=sampler,
                shape=(batch_size,),
                rng=rng,
                device=state.device,
                dtype=torch.float32,
                name=name,
            )
            params[:, 0, self.schema.param_id(name)] = value
            extra_values[name] = value

        state.data[self.out_key] = EventBatch(
            times=times_idx[:, None].to(torch.float32),
            type_ids=torch.full(
                (batch_size, 1),
                self.type_id,
                device=state.device,
                dtype=torch.int64,
            ),
            params=params,
            mask=torch.ones((batch_size, 1), device=state.device, dtype=torch.bool),
            schema=self.schema,
            meta={"seq_len": self.seq_len},
        )

        samples_base = f"{SAMPLES_PREFIX}/SingleEventNode:{self.out_key}"
        state.data[f"{samples_base}/time_idx"] = times_idx.to(torch.int64)
        state.data[f"{samples_base}/amplitude"] = amplitude
        for name, value in extra_values.items():
            state.data[f"{samples_base}/{name}"] = value
        return state


def _resolve_periodic_contract(
    *,
    seq_len: int,
    sampling_frequency: int,
    periodic_cfg: dict[str, Any],
    benchmark_family: str,
    benchmark_version: str,
) -> tuple[ResolvedAionoBasicComponentsPeriodicContract, str]:
    if benchmark_version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported benchmark_version={benchmark_version!r}")
    periodic_contract = resolve_aiono_basic_components_periodic_contract(
        seq_len=int(seq_len),
        sampling_frequency_hz=int(sampling_frequency),
        config=AionoBasicComponentsPeriodicConfig.from_mapping(periodic_cfg),
        benchmark_family=benchmark_family,
        benchmark_version="v1",
    )
    return periodic_contract, "v1"


def _midpoint(low: float, high: float) -> float:
    return 0.5 * (float(low) + float(high))


def _positive_nonzero_anchor(low: float, high: float) -> float:
    if high <= 0:
        return _midpoint(low, high)
    positive_low = max(float(low), 0.0)
    value = 0.5 * (positive_low + float(high))
    if value == 0.0:
        return float(high)
    return float(value)


def _periodic_specs(
    periodic_contract: ResolvedAionoBasicComponentsPeriodicContract,
    component: str,
) -> dict[str, dict[str, object]]:
    return periodic_contract.signal(component).sampler_specs()


def _range_from_spec(spec: dict[str, object], *, name: str) -> tuple[float, float]:
    if spec.get("kind") != "uniform":
        raise ValueError(f"{name} must use a uniform sampler spec, got {spec!r}")
    low = spec.get("low")
    high = spec.get("high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise ValueError(f"{name} sampler spec must define numeric low/high, got {spec!r}")
    return float(low), float(high)


def _view_range(component: str, param: str) -> tuple[float, float]:
    ranges = {
        ("linear_trend", "slope"): (-2.0, 2.0),
        ("linear_trend", "intercept"): (-0.5, 0.5),
        ("quadratic_trend", "a"): (-4.0, 4.0),
        ("quadratic_trend", "b"): (-2.0, 2.0),
        ("quadratic_trend", "c"): (-0.5, 0.5),
        ("log_trend", "amplitude"): (-2.0, 2.0),
        ("log_trend", "offset"): (-0.5, 0.5),
        ("sigmoid_trend", "amplitude"): (-2.0, 2.0),
        ("sigmoid_trend", "center"): (0.2, 0.8),
        ("sigmoid_trend", "sharpness"): (5.0, 20.0),
        ("sigmoid_trend", "offset"): (-0.5, 0.5),
        ("gaussian_noise", "noise_std"): (0.02, 0.15),
        ("uniform_noise", "amplitude"): (0.05, 0.25),
        ("random_walk_noise", "step_std"): (0.01, 0.08),
    }
    try:
        return ranges[(component, param)]
    except KeyError as exc:
        raise ValueError(f"Unsupported controlled view range for {component}.{param}") from exc


def _expanded_view_range(
    *,
    default_low: float,
    default_high: float,
    max_abs: float,
    log_min_abs: float,
) -> tuple[float, float]:
    if max_abs <= 0:
        raise ValueError(f"max_abs must be positive, got {max_abs}")
    if log_min_abs <= 0:
        raise ValueError(f"log_min_abs must be positive, got {log_min_abs}")
    if default_low < 0 < default_high:
        return -float(max_abs), float(max_abs)
    if default_low >= 0:
        return float(min(log_min_abs, max(default_low, 0.0))), float(max_abs)
    return -float(max_abs), -float(min(log_min_abs, abs(min(default_high, 0.0))))


def _target_geometry(target_name: str) -> str:
    if target_name.endswith("_phase"):
        return "circle"
    if target_name.endswith("_frequency_hz"):
        return "positive_scalar_log"
    if target_name.endswith("_time_frac"):
        return "interval"
    return "interval"


def _resolve_target(
    *,
    target_name: str,
    dense_targets_cfg: list[dict[str, Any]],
    periodic_contract: ResolvedAionoBasicComponentsPeriodicContract,
    seq_len: int,
    view_grid_mode: str = "linear",
    view_range_max_abs: float | None = None,
    view_log_min_abs: float = 1e-6,
) -> _ResolvedTarget:
    target_cfg_by_name = {str(item["name"]): dict(item) for item in dense_targets_cfg}
    if target_name not in target_cfg_by_name:
        raise ValueError(f"Unknown dense target {target_name!r}")
    target_cfg = target_cfg_by_name[target_name]
    component = str(target_cfg["enabled_key"])
    param = str(target_cfg["param"])
    source = str(target_cfg["source"])
    node_or_view = str(target_cfg["node"] if source == "process" else target_cfg["view"])
    geometry = _target_geometry(target_name)
    period = None
    dtype = "float"
    grid_mode = "linear"
    range_policy = "benchmark_contract"
    if param == "time_idx":
        physical_low = float(int(seq_len * 0.15))
        physical_high = float(int(seq_len * 0.85))
        dtype = "int"
        coordinate_name = "time_frac"
        range_policy = "central_15_85_percent"
    elif component in {"sine", "sawtooth", "square"}:
        low, high = _range_from_spec(
            _periodic_specs(periodic_contract, component)[param],
            name=f"{component}.{param}",
        )
        physical_low, physical_high = low, high
        coordinate_name = "log_" + param if geometry == "positive_scalar_log" else param
        if geometry == "circle":
            period = float(high - low)
        range_policy = "resolved_periodic_contract"
    else:
        default_low, default_high = _view_range(component, param)
        if view_range_max_abs is None:
            physical_low, physical_high = default_low, default_high
            range_policy = "manifold_default_view_range"
        else:
            physical_low, physical_high = _expanded_view_range(
                default_low=default_low,
                default_high=default_high,
                max_abs=float(view_range_max_abs),
                log_min_abs=float(view_log_min_abs),
            )
            range_policy = f"wide_abs_{float(view_range_max_abs):g}"
        grid_mode = str(view_grid_mode)
        coordinate_name = param
        if grid_mode == "signed_log":
            coordinate_name = "signed_log_" + param
        elif grid_mode == "log":
            coordinate_name = "log_" + param
    return _ResolvedTarget(
        geometry=ManifoldTargetGeometry(
            target_name=target_name,
            component=component,
            parameter=param,
            geometry=geometry,
            source=source,
            node_or_view=node_or_view,
            coordinate_name=coordinate_name,
            period=period,
        ),
        target_cfg=target_cfg,
        physical_low=float(physical_low),
        physical_high=float(physical_high),
        physical_grid_dtype=dtype,
        grid_mode=grid_mode,
        range_policy=range_policy,
        log_min_abs=float(view_log_min_abs),
    )


def _log_positions(*, count: int, split: str) -> torch.Tensor:
    if count <= 0:
        return torch.empty((0,), dtype=torch.float64)
    if split == "train":
        if count == 1:
            return torch.zeros((1,), dtype=torch.float64)
        return torch.linspace(0.0, 1.0, steps=count, dtype=torch.float64)
    if split == "val":
        return (torch.arange(count, dtype=torch.float64) + 0.5) / float(count)
    raise ValueError(f"split must be 'train' or 'val', got {split!r}")


def _positive_log_grid(
    *,
    low: float,
    high: float,
    grid_size: int,
    split: str,
    min_abs: float,
) -> torch.Tensor:
    resolved_low = max(float(low), float(min_abs))
    resolved_high = float(high)
    if resolved_high <= 0:
        raise ValueError(f"log grid high must be positive, got {resolved_high}")
    if resolved_low >= resolved_high:
        raise ValueError(
            f"log grid low must be less than high, got low={resolved_low} high={resolved_high}"
        )
    positions = _log_positions(count=int(grid_size), split=split)
    return torch.exp(
        math.log(resolved_low) + positions * (math.log(resolved_high) - math.log(resolved_low))
    )


def _signed_log_grid(
    *,
    low: float,
    high: float,
    grid_size: int,
    split: str,
    min_abs: float,
) -> torch.Tensor:
    if low < 0 < high:
        include_zero = int(grid_size) % 2 == 1
        neg_count = int(grid_size) // 2
        pos_count = int(grid_size) - neg_count - int(include_zero)
        neg_magnitudes = _positive_log_grid(
            low=float(min_abs),
            high=abs(float(low)),
            grid_size=neg_count,
            split=split,
            min_abs=float(min_abs),
        )
        pos_magnitudes = _positive_log_grid(
            low=float(min_abs),
            high=float(high),
            grid_size=pos_count,
            split=split,
            min_abs=float(min_abs),
        )
        pieces = [-neg_magnitudes.flip(0)]
        if include_zero:
            pieces.append(torch.zeros((1,), dtype=torch.float64))
        pieces.append(pos_magnitudes)
        return torch.cat(pieces, dim=0)
    if low >= 0:
        return _positive_log_grid(
            low=low,
            high=high,
            grid_size=grid_size,
            split=split,
            min_abs=min_abs,
        )
    if high <= 0:
        magnitudes = _positive_log_grid(
            low=max(abs(high), min_abs),
            high=abs(low),
            grid_size=grid_size,
            split=split,
            min_abs=min_abs,
        )
        return -magnitudes.flip(0)
    raise ValueError(f"Unsupported signed-log range low={low} high={high}")


def _grid_values(
    *,
    resolved_target: _ResolvedTarget,
    grid_size: int,
    split: str,
) -> torch.Tensor:
    if grid_size < 4:
        raise ValueError(f"grid_size must be >= 4, got {grid_size}")
    low = float(resolved_target.physical_low)
    high = float(resolved_target.physical_high)
    if resolved_target.geometry.geometry == "interval" and resolved_target.grid_mode in {
        "log",
        "signed_log",
    }:
        if resolved_target.grid_mode == "log":
            values = _positive_log_grid(
                low=low,
                high=high,
                grid_size=int(grid_size),
                split=split,
                min_abs=float(resolved_target.log_min_abs),
            )
        else:
            values = _signed_log_grid(
                low=low,
                high=high,
                grid_size=int(grid_size),
                split=split,
                min_abs=float(resolved_target.log_min_abs),
            )
        if resolved_target.physical_grid_dtype == "int":
            values = values.round().clamp(min=low, max=high).to(torch.int64).to(torch.float64)
        return values
    if split == "train":
        if resolved_target.geometry.geometry == "circle":
            values = low + torch.arange(grid_size, dtype=torch.float64) * ((high - low) / grid_size)
        elif resolved_target.geometry.geometry == "positive_scalar_log":
            values = torch.exp(torch.linspace(math.log(low), math.log(high), steps=grid_size, dtype=torch.float64))
        else:
            values = torch.linspace(low, high, steps=grid_size, dtype=torch.float64)
    elif split == "val":
        offsets = (torch.arange(grid_size, dtype=torch.float64) + 0.5) / float(grid_size)
        if resolved_target.geometry.geometry == "positive_scalar_log":
            values = torch.exp(math.log(low) + offsets * (math.log(high) - math.log(low)))
        else:
            values = low + offsets * (high - low)
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if resolved_target.physical_grid_dtype == "int":
        values = values.round().clamp(min=low, max=high).to(torch.int64).to(torch.float64)
    return values


def _latent_coordinates(
    *,
    resolved_target: _ResolvedTarget,
    physical_values: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    values = physical_values.to(dtype=torch.float64)
    param = resolved_target.geometry.parameter
    geometry = resolved_target.geometry.geometry
    if param == "time_idx":
        return values / float(seq_len - 1)
    if geometry == "interval" and resolved_target.grid_mode == "signed_log":
        abs_values = torch.clamp(torch.abs(values), min=float(resolved_target.log_min_abs))
        return torch.sign(values) * torch.log1p(abs_values / float(resolved_target.log_min_abs))
    if geometry == "interval" and resolved_target.grid_mode == "log":
        abs_values = torch.clamp(values, min=float(resolved_target.log_min_abs))
        return torch.log(abs_values)
    if geometry == "positive_scalar_log":
        return torch.log(values)
    if geometry == "circle":
        return values - float(resolved_target.physical_low)
    return values


def _repeat_grid(values: torch.Tensor, repeats_per_grid_point: int) -> tuple[torch.Tensor, torch.Tensor]:
    if repeats_per_grid_point < 1:
        raise ValueError("repeats_per_grid_point must be >= 1")
    grid_index = torch.arange(int(values.numel()), dtype=torch.int64).repeat_interleave(
        int(repeats_per_grid_point)
    )
    return values.repeat_interleave(int(repeats_per_grid_point)), grid_index


def _constant_or_sequence(
    *,
    component: str,
    param: str,
    target_component: str,
    target_param: str,
    controlled_values: torch.Tensor,
    default_value: float,
) -> Sampler:
    if component == target_component and param == target_param:
        return SequenceSampler(controlled_values)
    return ConstantSampler(float(default_value))


def _periodic_anchor_values(
    *,
    component: str,
    periodic_contract: ResolvedAionoBasicComponentsPeriodicContract,
) -> dict[str, float]:
    specs = _periodic_specs(periodic_contract, component)
    amp_low, amp_high = _range_from_spec(specs["amplitude"], name=f"{component}.amplitude")
    freq_low, freq_high = _range_from_spec(specs["frequency_hz"], name=f"{component}.frequency_hz")
    phase_low, _ = _range_from_spec(specs["phase"], name=f"{component}.phase")
    offset_low, offset_high = _range_from_spec(specs["offset"], name=f"{component}.offset")
    out = {
        "amplitude": _positive_nonzero_anchor(amp_low, amp_high),
        "frequency_hz": math.exp(0.5 * (math.log(freq_low) + math.log(freq_high))),
        "phase": float(phase_low),
        "offset": _midpoint(offset_low, offset_high),
    }
    if "duty_cycle" in specs:
        duty_low, duty_high = _range_from_spec(specs["duty_cycle"], name=f"{component}.duty_cycle")
        out["duty_cycle"] = _midpoint(duty_low, duty_high)
    return out


def _build_controlled_pipeline(
    *,
    seq_len: int,
    sample_rate_hz: float,
    view_name: str,
    component_keys: list[str],
    target: _ResolvedTarget,
    controlled_values: torch.Tensor,
    periodic_contract: ResolvedAionoBasicComponentsPeriodicContract,
    device: torch.device,
) -> tuple[SynthPipeline, dict[str, Any]]:
    target_component = target.geometry.component
    target_param = target.geometry.parameter
    if target_component not in component_keys:
        raise ValueError(f"Target component {target_component!r} is missing from component_keys")
    component_id = int(component_keys.index(target_component))

    schema = EventSchema(
        type_names=["spike", "level_change", "gaussian"],
        param_names=["amplitude", "sigma_sec"],
        time_unit="samples",
    )
    process_nodes: list[torch.nn.Module] = [
        EnableComponentsNode(
            component_keys=component_keys,
            num_enabled=1,
            component_id=ConstantSampler(component_id),
        ),
        ConstantLatentNode(
            seq_len=seq_len,
            channels=1,
            value=0.0,
            enabled_key="constant",
            out_key="latent",
        ),
    ]
    event_in_keys: list[str] = []
    fixed_values: dict[str, Any] = {}
    if target_component in _EVENT_COMPONENTS:
        event_fixed = {
            "spike": {"amplitude": 1.0},
            "level_change": {"amplitude": 1.0},
            "gaussian": {"amplitude": 1.0, "sigma_sec": 0.035},
        }[target_component]
        time_sampler: Sampler | float = event_fixed.get("time_idx", round(seq_len * 0.5))
        amplitude_sampler: Sampler | float = event_fixed["amplitude"]
        extra_params: dict[str, Sampler | float] = {}
        if target_param == "time_idx":
            time_sampler = SequenceSampler(controlled_values)
        elif target_param == "amplitude":
            amplitude_sampler = SequenceSampler(controlled_values)
        if target_component == "gaussian":
            sigma_value: Sampler | float = event_fixed["sigma_sec"]
            if target_param == "sigma_sec":
                sigma_value = SequenceSampler(controlled_values)
            extra_params["sigma_sec"] = sigma_value
        process_nodes.extend(
            [
                ControlledSingleEventNode(
                    seq_len=seq_len,
                    schema=schema,
                    type_name=target_component,
                    time_idx=time_sampler,
                    amplitude=amplitude_sampler,
                    amplitude_param="amplitude",
                    extra_params=extra_params,
                    out_key=target_component,
                ),
                GateEventsByEnabledNode(
                    in_key=target_component,
                    enabled_key=target_component,
                    out_key=f"{target_component}.gated",
                ),
            ]
        )
        event_in_keys.append(f"{target_component}.gated")
        fixed_values[target_component] = {
            key: float(value) for key, value in event_fixed.items() if key != target_param
        }

    outputs = {"latent"}
    if event_in_keys:
        from aiono import UnionEventsNode

        process_nodes.append(UnionEventsNode(in_keys=event_in_keys, out_key="events"))
        outputs.add("events")

    process = ProcessGraph(
        name="BasicComponentsProcess",
        outputs=outputs,
        base_meta={
            "seq_len": int(seq_len),
            "sample_rate_hz": float(sample_rate_hz),
            "component_keys": list(component_keys),
            "num_enabled": 1,
            "controlled_slice": {
                "target": target.geometry.target_name,
                "component": target_component,
                "parameter": target_param,
            },
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
    for component in component_keys:
        if component == "gaussian_noise":
            views.append(GaussianNoiseView(noise_std=ConstantSampler(0.0), enabled_key=component))
            fixed_values[component] = {"noise_std": 0.0}
        elif component == "uniform_noise":
            views.append(UniformNoiseView(amplitude=ConstantSampler(0.0), enabled_key=component))
            fixed_values[component] = {"amplitude": 0.0}
        elif component == "random_walk_noise":
            views.append(RandomWalkNoiseView(step_std=ConstantSampler(0.0), enabled_key=component))
            fixed_values[component] = {"step_std": 0.0}
        elif component == "linear_trend":
            fixed = {"slope": 1.0, "intercept": 0.0}
            views.append(
                LinearTrendView(
                    seq_len=seq_len,
                    slope=_constant_or_sequence(
                        component=component,
                        param="slope",
                        target_component=target_component,
                        target_param=target_param,
                        controlled_values=controlled_values,
                        default_value=fixed["slope"],
                    ),
                    intercept=_constant_or_sequence(
                        component=component,
                        param="intercept",
                        target_component=target_component,
                        target_param=target_param,
                        controlled_values=controlled_values,
                        default_value=fixed["intercept"],
                    ),
                    enabled_key=component,
                )
            )
            fixed_values[component] = {key: value for key, value in fixed.items() if key != target_param}
        elif component == "quadratic_trend":
            views.append(
                QuadraticTrendView(
                    seq_len=seq_len,
                    a=ConstantSampler(1.0),
                    b=ConstantSampler(0.0),
                    c=ConstantSampler(0.0),
                    enabled_key=component,
                )
            )
            fixed_values[component] = {"a": 1.0, "b": 0.0, "c": 0.0}
        elif component == "log_trend":
            views.append(
                LogTrendView(
                    seq_len=seq_len,
                    amplitude=ConstantSampler(1.0),
                    offset=ConstantSampler(0.0),
                    epsilon=1e-3,
                    enabled_key=component,
                )
            )
            fixed_values[component] = {"amplitude": 1.0, "offset": 0.0}
        elif component == "sigmoid_trend":
            views.append(
                SigmoidTrendView(
                    seq_len=seq_len,
                    amplitude=ConstantSampler(1.0),
                    center=ConstantSampler(0.5),
                    sharpness=ConstantSampler(10.0),
                    offset=ConstantSampler(0.0),
                    enabled_key=component,
                )
            )
            fixed_values[component] = {
                "amplitude": 1.0,
                "center": 0.5,
                "sharpness": 10.0,
                "offset": 0.0,
            }
        elif component in {"sine", "sawtooth", "square"}:
            fixed = _periodic_anchor_values(
                component=component,
                periodic_contract=periodic_contract,
            )
            kwargs = {
                "seq_len": seq_len,
                "amplitude": _constant_or_sequence(
                    component=component,
                    param="amplitude",
                    target_component=target_component,
                    target_param=target_param,
                    controlled_values=controlled_values,
                    default_value=fixed["amplitude"],
                ),
                "frequency_hz": _constant_or_sequence(
                    component=component,
                    param="frequency_hz",
                    target_component=target_component,
                    target_param=target_param,
                    controlled_values=controlled_values,
                    default_value=fixed["frequency_hz"],
                ),
                "phase": _constant_or_sequence(
                    component=component,
                    param="phase",
                    target_component=target_component,
                    target_param=target_param,
                    controlled_values=controlled_values,
                    default_value=fixed["phase"],
                ),
                "offset": _constant_or_sequence(
                    component=component,
                    param="offset",
                    target_component=target_component,
                    target_param=target_param,
                    controlled_values=controlled_values,
                    default_value=fixed["offset"],
                ),
                "enabled_key": component,
            }
            if component == "sine":
                views.append(SineWaveView(**kwargs))
            elif component == "sawtooth":
                views.append(SawtoothWaveView(**kwargs))
            else:
                kwargs["duty_cycle"] = _constant_or_sequence(
                    component=component,
                    param="duty_cycle",
                    target_component=target_component,
                    target_param=target_param,
                    controlled_values=controlled_values,
                    default_value=fixed["duty_cycle"],
                )
                views.append(SquareWaveView(**kwargs))
            fixed_values[component] = {
                key: float(value) for key, value in fixed.items() if key != target_param
            }
    if not views:
        raise ValueError("Controlled manifold slice requires at least one view")
    pipeline = SynthPipeline(process=process, views={view_name: ViewChain(*views)}).to(device)
    return pipeline, fixed_values


def _build_dataset_manifest(
    *,
    cfg_text: str,
    cfg: dict[str, Any],
    seq_len: int,
    device: torch.device,
    dense_targets_cfg: list[dict[str, Any]],
    dense_target_names: list[str],
    component_keys: list[str],
    periodic_contract: ResolvedAionoBasicComponentsPeriodicContract,
    periodic_contract_benchmark_version: str,
) -> dict[str, Any]:
    aiono = _require_dict(cfg, "aiono")
    sampling_frequency = _require_int(cfg, "sampling_frequency")
    benchmark_family = _require_str(cfg, "benchmark_family")
    benchmark_version = _require_str(cfg, "benchmark_version")
    periodic_manifest_fields = dict(periodic_contract.manifest_fields())
    periodic_manifest_fields["benchmark_family"] = benchmark_family
    periodic_manifest_fields["benchmark_version"] = benchmark_version
    dense_specs = _build_dense_target_specs(dense_targets_cfg)
    return {
        "dataset_name": "aiono_basic_components_balanced",
        **periodic_manifest_fields,
        "periodic_contract_benchmark_version": periodic_contract_benchmark_version,
        "view_name": _require_str(aiono, "view_name"),
        "sampling_frequency": int(sampling_frequency),
        "channels": [str(channel) for channel in cfg.get("channels", ["I"])],
        "default_channel_size": int(_require_int(cfg, "default_channel_size")),
        "channel_size": int(seq_len),
        "channel_size_policy": "manifold_adapter_exact",
        "channel_size_source": "manifold.adapter_exact",
        "train_seed": int(_require_int(aiono, "train_seed")),
        "validation_seed_values": [0],
        "validation_seed_offset": 0,
        "validation_generator_seeds": [0],
        "validation_seed_to_generator_seed": {"0": 0},
        "validation_seed_count": 1,
        "batch_size": 0,
        "train_batches": 0,
        "val_batches": 0,
        "num_enabled": 1,
        "num_enabled_values": [1],
        "component_keys": list(component_keys),
        "class_names": list(component_keys),
        "group_to_classes": _build_group_to_classes(component_keys),
        "dense_target_names": list(dense_target_names),
        "dense_targets": [asdict(spec) for spec in dense_specs],
        "config_sha256": hashlib.sha256(cfg_text.encode("utf-8")).hexdigest(),
        "created_at_unix": float(time.time()),
        "generator_device": str(device),
        "generation_mode": "controlled_manifold_slice_materialized",
    }


def build_controlled_manifold_slice(
    *,
    config_path: Path = DATASET_CONFIG_PATH,
    target_name: str,
    seq_len: int,
    grid_size: int,
    split: str,
    repeats_per_grid_point: int = 1,
    seed: int = 0,
    device: torch.device | None = None,
    view_grid_mode: str = "linear",
    view_range_max_abs: float | None = None,
    view_log_min_abs: float = 1e-6,
) -> ControlledSlice:
    actual_device = device or torch.device("cpu")
    cfg_text = config_path.read_text(encoding="utf-8")
    cfg = _load_dataset_config(config_path)
    aiono, component_keys, dense_targets_cfg = _parse_manifest_components(cfg)
    dense_target_names = aiono_dense_targets_validate_config(targets_cfg=dense_targets_cfg)
    sampling_frequency = _require_int(cfg, "sampling_frequency")
    periodic_contract, periodic_contract_benchmark_version = _resolve_periodic_contract(
        seq_len=int(seq_len),
        sampling_frequency=int(sampling_frequency),
        periodic_cfg=_require_dict(_require_dict(aiono, "basic_components"), "periodic"),
        benchmark_family=_require_str(cfg, "benchmark_family"),
        benchmark_version=_require_str(cfg, "benchmark_version"),
    )
    resolved_target = _resolve_target(
        target_name=target_name,
        dense_targets_cfg=dense_targets_cfg,
        periodic_contract=periodic_contract,
        seq_len=int(seq_len),
        view_grid_mode=str(view_grid_mode),
        view_range_max_abs=view_range_max_abs,
        view_log_min_abs=float(view_log_min_abs),
    )
    grid_physical = _grid_values(
        resolved_target=resolved_target,
        grid_size=int(grid_size),
        split=split,
    )
    physical_values, grid_index = _repeat_grid(
        grid_physical,
        repeats_per_grid_point=int(repeats_per_grid_point),
    )
    latent_coordinates = _latent_coordinates(
        resolved_target=resolved_target,
        physical_values=physical_values,
        seq_len=int(seq_len),
    )

    pipeline, fixed_values = _build_controlled_pipeline(
        seq_len=int(seq_len),
        sample_rate_hz=float(sampling_frequency),
        view_name=_require_str(aiono, "view_name"),
        component_keys=component_keys,
        target=resolved_target,
        controlled_values=physical_values.to(dtype=torch.float32),
        periodic_contract=periodic_contract,
        device=actual_device,
    )
    batch_size = int(physical_values.numel())
    dataset = SynthBatchIterableDataset(
        pipeline=pipeline,
        batch_size=batch_size,
        device=actual_device,
        seed=int(seed),
        max_batches=1,
    )
    loader = DataLoader(dataset=dataset, batch_size=None, shuffle=False, num_workers=0)
    views = next(iter(loader))
    obs = views[_require_str(aiono, "view_name")]
    y_cls = _extract_class_targets(obs=obs, class_names=component_keys, device=actual_device)
    y_dense, _ = aiono_dense_targets_extract(
        obs=obs,
        targets_cfg=dense_targets_cfg,
        target_names=dense_target_names,
    )
    target_index = int(dense_target_names.index(target_name))
    dataset_manifest = _build_dataset_manifest(
        cfg_text=cfg_text,
        cfg=cfg,
        seq_len=int(seq_len),
        device=actual_device,
        dense_targets_cfg=dense_targets_cfg,
        dense_target_names=dense_target_names,
        component_keys=component_keys,
        periodic_contract=periodic_contract,
        periodic_contract_benchmark_version=periodic_contract_benchmark_version,
    )
    manifest = {
        "schema_version": "manifold_controlled_slice_v0",
        "split": split,
        "target": resolved_target.geometry.to_payload(),
        "sweep": {
            "grid_mode": resolved_target.grid_mode,
            "range_policy": resolved_target.range_policy,
            "physical_low": float(resolved_target.physical_low),
            "physical_high": float(resolved_target.physical_high),
            "log_min_abs": float(resolved_target.log_min_abs),
        },
        "num_enabled": 1,
        "grid_size": int(grid_size),
        "repeats_per_grid_point": int(repeats_per_grid_point),
        "seed": int(seed),
        "fixed_factor_policy": "canonical_non_degenerate",
        "nuisance_policy": "none_for_canonical_curve",
        "physical_grid": [float(value) for value in grid_physical.tolist()],
        "physical_values": [float(value) for value in physical_values.tolist()],
        "latent_coordinates": [float(value) for value in latent_coordinates.tolist()],
        "grid_index": [int(value) for value in grid_index.tolist()],
        "fixed_values": fixed_values,
        "target_index": int(target_index),
        "dense_target_names": list(dense_target_names),
        "dataset_manifest": dataset_manifest,
        "periodic_sampler_specs": dataset_manifest.get("periodic_sampler_specs"),
    }
    return ControlledSlice(
        split={
            "x": obs.x.detach().to("cpu"),
            "y_cls": y_cls.detach().to("cpu"),
            "y_dense": y_dense.detach().to("cpu"),
        },
        manifest=manifest,
        dense_target_names=list(dense_target_names),
        target_index=target_index,
        grid_index=grid_index.detach().cpu(),
        latent_coordinates=latent_coordinates.to(dtype=torch.float32).detach().cpu(),
        physical_values=physical_values.to(dtype=torch.float32).detach().cpu(),
    )
