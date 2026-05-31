from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from time import perf_counter

import torch
from torch.nn import functional as F


BASELINE_LAYER = 0
DEFAULT_FEATURE_BATCH_SIZE = 2048
PAPER_CRITICAL_BASELINES = (
    "MetricFloor",
    "RawDownsample512",
    "FFTLogPower512",
    "StatsWindowed",
    "StatsFFTCombined",
    "RandomProjection512",
    "OracleEnabledMask",
    "OracleDenseParams",
)
BASELINE_ALIASES = {
    "majority": "MetricFloor",
    "prevalence": "MetricFloor",
    "meantarget": "MetricFloor",
    "mean-target": "MetricFloor",
    "mean_target": "MetricFloor",
    "paper-critical": "paper-critical",
    "paper_critical": "paper-critical",
    "all": "all",
}


FeatureFn = Callable[
    [torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, object], int],
    torch.Tensor,
]


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    slug: str
    family: str
    description: str
    uses_probe: bool
    extractor: FeatureFn | None
    feature_dim: int | None = None
    uses_targets: bool = False
    is_oracle: bool = False


def _series(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected x to be [B, C, L], got {tuple(x.shape)}")
    if int(x.size(1)) != 1:
        raise ValueError(f"Current baselines expect one channel, got shape {tuple(x.shape)}")
    return x[:, 0, :].to(dtype=torch.float32)


def _downsample(series: torch.Tensor, *, output_dim: int) -> torch.Tensor:
    if series.dim() != 2:
        raise ValueError(f"Expected series to be [B, L], got {tuple(series.shape)}")
    if int(series.size(1)) == int(output_dim):
        return series.contiguous()
    return F.interpolate(
        series.unsqueeze(1),
        size=int(output_dim),
        mode="linear",
        align_corners=False,
    ).squeeze(1)


def _safe_std(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    return values.std(dim=dim, unbiased=False)


def _linear_trend_features(series: torch.Tensor) -> torch.Tensor:
    length = int(series.size(1))
    t = torch.linspace(-0.5, 0.5, length, device=series.device, dtype=series.dtype)
    denom = torch.sum(t * t).clamp_min(1.0e-12)
    mean = series.mean(dim=1, keepdim=True)
    slope = torch.sum((series - mean) * t.unsqueeze(0), dim=1) / denom
    intercept = mean.squeeze(1)
    return torch.stack([slope, intercept], dim=1)


def _diff_summary(diff: torch.Tensor) -> torch.Tensor:
    if int(diff.size(1)) < 1:
        zeros = torch.zeros((int(diff.size(0)), 4), dtype=diff.dtype, device=diff.device)
        return zeros
    return torch.stack(
        [
            diff.mean(dim=1),
            _safe_std(diff, dim=1),
            diff.abs().mean(dim=1),
            torch.sqrt(torch.mean(diff * diff, dim=1).clamp_min(0.0)),
        ],
        dim=1,
    )


def _global_stats(series: torch.Tensor) -> torch.Tensor:
    first_diff = series[:, 1:] - series[:, :-1]
    second_diff = first_diff[:, 1:] - first_diff[:, :-1] if int(first_diff.size(1)) > 1 else first_diff[:, :0]
    signs = torch.sign(series)
    sign_changes = (signs[:, 1:] * signs[:, :-1] < 0).to(dtype=series.dtype)
    zero_cross_fraction = sign_changes.mean(dim=1, keepdim=True)
    base = torch.stack(
        [
            series.mean(dim=1),
            _safe_std(series, dim=1),
            series.amin(dim=1),
            series.amax(dim=1),
            series.median(dim=1).values,
            series.abs().mean(dim=1),
            torch.sqrt(torch.mean(series * series, dim=1).clamp_min(0.0)),
            torch.mean(series * series, dim=1),
        ],
        dim=1,
    )
    return torch.cat(
        [
            base,
            _diff_summary(first_diff),
            _diff_summary(second_diff),
            zero_cross_fraction,
            _linear_trend_features(series),
        ],
        dim=1,
    )


def _windowed_stats(series: torch.Tensor, *, windows: int = 16) -> torch.Tensor:
    chunks = torch.chunk(series, int(windows), dim=1)
    features = []
    for chunk in chunks:
        if int(chunk.size(1)) < 1:
            continue
        features.extend(
            [
                chunk.mean(dim=1),
                _safe_std(chunk, dim=1),
                chunk.amin(dim=1),
                chunk.amax(dim=1),
                chunk[:, -1] - chunk[:, 0],
            ]
        )
    return torch.stack(features, dim=1)


def _fft_power(series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = series - series.mean(dim=1, keepdim=True)
    length = int(centered.size(1))
    window = torch.hann_window(length, periodic=True, dtype=centered.dtype, device=centered.device)
    spectrum = torch.fft.rfft(centered * window.unsqueeze(0), dim=1)
    power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
    sample_rate = 500.0
    freqs = torch.fft.rfftfreq(length, d=1.0 / sample_rate).to(device=series.device, dtype=series.dtype)
    return power, freqs


def _fft_log_power(series: torch.Tensor, *, output_dim: int = 512) -> torch.Tensor:
    power, _ = _fft_power(series)
    log_power = torch.log1p(power)
    if int(log_power.size(1)) == int(output_dim):
        return log_power.contiguous()
    return F.interpolate(
        log_power.unsqueeze(1),
        size=int(output_dim),
        mode="linear",
        align_corners=False,
    ).squeeze(1)


def _band_energy_features(power: torch.Tensor, *, bands: int = 8) -> torch.Tensor:
    total = power.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    chunks = torch.chunk(power, int(bands), dim=1)
    energies = [chunk.sum(dim=1) / total.squeeze(1) for chunk in chunks if int(chunk.size(1)) > 0]
    return torch.stack(energies, dim=1)


def _fft_stats(series: torch.Tensor) -> torch.Tensor:
    power, freqs = _fft_power(series)
    total = power.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    probabilities = power / total
    centroid = torch.sum(probabilities * freqs.unsqueeze(0), dim=1)
    centered_freq = freqs.unsqueeze(0) - centroid.unsqueeze(1)
    bandwidth = torch.sqrt(torch.sum(probabilities * centered_freq * centered_freq, dim=1).clamp_min(0.0))
    entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(1.0e-12)), dim=1)
    entropy = entropy / torch.log(torch.tensor(float(power.size(1)), device=series.device, dtype=series.dtype))
    peak_count = min(5, int(power.size(1)))
    peak_values, peak_indices = torch.topk(power, k=peak_count, dim=1)
    peak_freqs = freqs.index_select(0, peak_indices.reshape(-1)).reshape_as(peak_values)
    peak_values = torch.log1p(peak_values)
    if peak_count < 5:
        pad_cols = 5 - peak_count
        peak_values = F.pad(peak_values, (0, pad_cols))
        peak_freqs = F.pad(peak_freqs, (0, pad_cols))
    return torch.cat(
        [
            centroid.unsqueeze(1),
            bandwidth.unsqueeze(1),
            entropy.unsqueeze(1),
            _band_energy_features(power, bands=8),
            peak_freqs,
            peak_values,
        ],
        dim=1,
    )


def _raw_downsample_512(
    x: torch.Tensor,
    y_cls: torch.Tensor | None,
    y_dense: torch.Tensor | None,
    manifest: dict[str, object],
    seed: int,
) -> torch.Tensor:
    del y_cls, y_dense, manifest, seed
    return _downsample(_series(x), output_dim=512)


def _fft_log_power_512(
    x: torch.Tensor,
    y_cls: torch.Tensor | None,
    y_dense: torch.Tensor | None,
    manifest: dict[str, object],
    seed: int,
) -> torch.Tensor:
    del y_cls, y_dense, manifest, seed
    return _fft_log_power(_series(x), output_dim=512)


def _stats_windowed(
    x: torch.Tensor,
    y_cls: torch.Tensor | None,
    y_dense: torch.Tensor | None,
    manifest: dict[str, object],
    seed: int,
) -> torch.Tensor:
    del y_cls, y_dense, manifest, seed
    series = _series(x)
    return torch.cat([_global_stats(series), _windowed_stats(series, windows=16)], dim=1)


def _stats_fft_combined(
    x: torch.Tensor,
    y_cls: torch.Tensor | None,
    y_dense: torch.Tensor | None,
    manifest: dict[str, object],
    seed: int,
) -> torch.Tensor:
    del y_cls, y_dense, manifest, seed
    series = _series(x)
    return torch.cat(
        [
            _global_stats(series),
            _windowed_stats(series, windows=16),
            _fft_log_power(series, output_dim=512),
            _fft_stats(series),
        ],
        dim=1,
    )


def _random_projection_512(
    x: torch.Tensor,
    y_cls: torch.Tensor | None,
    y_dense: torch.Tensor | None,
    manifest: dict[str, object],
    seed: int,
) -> torch.Tensor:
    del y_cls, y_dense, manifest
    features = _downsample(_series(x), output_dim=512)
    features = features - features.mean(dim=1, keepdim=True)
    denom = _safe_std(features, dim=1).unsqueeze(1).clamp_min(1.0e-6)
    features = features / denom
    generator_device = "cuda" if features.device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed) + 1_048_573)
    projection = torch.randn(
        (int(features.size(1)), 512),
        generator=generator,
        device=features.device,
        dtype=features.dtype,
    ) / sqrt(float(features.size(1)))
    return features @ projection


def _oracle_enabled_mask(
    x: torch.Tensor,
    y_cls: torch.Tensor | None,
    y_dense: torch.Tensor | None,
    manifest: dict[str, object],
    seed: int,
) -> torch.Tensor:
    del x, y_dense, manifest, seed
    if y_cls is None:
        raise ValueError("OracleEnabledMask requires class targets")
    return y_cls.to(dtype=torch.float32)


def _oracle_dense_params(
    x: torch.Tensor,
    y_cls: torch.Tensor | None,
    y_dense: torch.Tensor | None,
    manifest: dict[str, object],
    seed: int,
) -> torch.Tensor:
    del x, manifest, seed
    if y_cls is None or y_dense is None:
        raise ValueError("OracleDenseParams requires class and dense targets")
    dense = y_dense.to(dtype=torch.float32)
    valid = torch.isfinite(dense).to(dtype=torch.float32)
    return torch.cat([y_cls.to(dtype=torch.float32), torch.nan_to_num(dense, nan=0.0), valid], dim=1)


_BASELINE_SPECS: dict[str, BaselineSpec] = {
    "MetricFloor": BaselineSpec(
        name="MetricFloor",
        slug="MetricFloor",
        family="metric_floor",
        description="Categorical prevalence and dense train-mean prediction without a learned probe.",
        uses_probe=False,
        extractor=None,
        feature_dim=1,
    ),
    "RawDownsample512": BaselineSpec(
        name="RawDownsample512",
        slug="RawDownsample512",
        family="raw_waveform",
        description="Observed waveform linearly resampled to 512 features.",
        uses_probe=True,
        extractor=_raw_downsample_512,
        feature_dim=512,
    ),
    "FFTLogPower512": BaselineSpec(
        name="FFTLogPower512",
        slug="FFTLogPower512",
        family="spectral",
        description="Hann-windowed log-power rFFT resampled to 512 frequency bins.",
        uses_probe=True,
        extractor=_fft_log_power_512,
        feature_dim=512,
    ),
    "StatsWindowed": BaselineSpec(
        name="StatsWindowed",
        slug="StatsWindowed",
        family="statistical",
        description="Global and fixed-window time-domain summary statistics.",
        uses_probe=True,
        extractor=_stats_windowed,
    ),
    "StatsFFTCombined": BaselineSpec(
        name="StatsFFTCombined",
        slug="StatsFFTCombined",
        family="statistical_spectral",
        description="Windowed statistics plus FFT log-power and spectral summaries.",
        uses_probe=True,
        extractor=_stats_fft_combined,
    ),
    "RandomProjection512": BaselineSpec(
        name="RandomProjection512",
        slug="RandomProjection512",
        family="random_feature",
        description="Fixed Gaussian projection of normalized downsampled waveform to 512 features.",
        uses_probe=True,
        extractor=_random_projection_512,
        feature_dim=512,
    ),
    "OracleEnabledMask": BaselineSpec(
        name="OracleEnabledMask",
        slug="OracleEnabledMask",
        family="oracle",
        description="Generator component-enabled indicators exposed as features.",
        uses_probe=True,
        extractor=_oracle_enabled_mask,
        uses_targets=True,
        is_oracle=True,
    ),
    "OracleDenseParams": BaselineSpec(
        name="OracleDenseParams",
        slug="OracleDenseParams",
        family="oracle",
        description="Generator component indicators, dense parameters, and dense-valid masks exposed as features.",
        uses_probe=True,
        extractor=_oracle_dense_params,
        uses_targets=True,
        is_oracle=True,
    ),
}


def baseline_names() -> list[str]:
    return list(_BASELINE_SPECS)


def get_baseline_spec(name: str) -> BaselineSpec:
    key = _canonical_baseline_name(name)
    if key in ("paper-critical", "all"):
        raise ValueError(f"{name!r} expands to multiple baselines; use resolve_baseline_names")
    try:
        return _BASELINE_SPECS[key]
    except KeyError as error:
        raise KeyError(f"Unknown baseline {name!r}; expected one of {baseline_names()}") from error


def _canonical_baseline_name(name: str) -> str:
    raw = str(name).strip()
    if not raw:
        raise ValueError("baseline name must be non-empty")
    lowered = raw.lower()
    alias = BASELINE_ALIASES.get(lowered)
    if alias is not None:
        return alias
    for key in _BASELINE_SPECS:
        if lowered == key.lower():
            return key
    return raw


def resolve_baseline_names(names: list[str] | tuple[str, ...]) -> list[str]:
    if not names:
        raise ValueError("At least one baseline name is required")
    resolved: list[str] = []
    for name in names:
        key = _canonical_baseline_name(name)
        if key == "paper-critical":
            candidates = list(PAPER_CRITICAL_BASELINES)
        elif key == "all":
            candidates = baseline_names()
        else:
            candidates = [get_baseline_spec(key).name]
        for candidate in candidates:
            if candidate not in resolved:
                resolved.append(candidate)
    return resolved


def collect_split_features(
    *,
    spec: BaselineSpec,
    split: dict[str, torch.Tensor],
    manifest: dict[str, object],
    seed: int,
    device: torch.device,
    batch_size: int = DEFAULT_FEATURE_BATCH_SIZE,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if spec.extractor is None:
        raise ValueError(f"{spec.name} does not define a feature extractor")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    x = split["x"]
    y_cls = split.get("y_cls")
    y_dense = split.get("y_dense")
    if not isinstance(x, torch.Tensor):
        raise TypeError("split['x'] must be a torch.Tensor")
    feature_batches: list[torch.Tensor] = []
    total_start = perf_counter()
    forward_s = 0.0
    batches = 0
    samples = int(x.size(0))
    with torch.inference_mode():
        for start in range(0, samples, int(batch_size)):
            end = min(samples, start + int(batch_size))
            batch_x = x[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
            batch_y_cls = (
                None
                if y_cls is None
                else y_cls[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
            )
            batch_y_dense = (
                None
                if y_dense is None
                else y_dense[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
            )
            forward_start = perf_counter()
            features = spec.extractor(batch_x, batch_y_cls, batch_y_dense, manifest, int(seed))
            forward_s += perf_counter() - forward_start
            if features.dim() != 2:
                raise ValueError(
                    f"{spec.name} produced non-2D features for rows {start}:{end}: {tuple(features.shape)}"
                )
            if torch.any(~torch.isfinite(features)):
                raise ValueError(f"{spec.name} produced NaN/Inf features for rows {start}:{end}")
            feature_batches.append(features.detach().to(device="cpu", dtype=torch.float32))
            batches += 1
    if not feature_batches:
        raise ValueError("Feature extraction produced no batches")
    features_all = torch.cat(feature_batches, dim=0)
    return features_all, {
        "total_s": float(perf_counter() - total_start),
        "forward_s": float(forward_s),
        "batches": int(batches),
        "samples": int(samples),
        "feature_dim": int(features_all.size(1)),
    }
