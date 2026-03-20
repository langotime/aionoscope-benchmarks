from __future__ import annotations

import argparse
import gc
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, time
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from .constants import DATASET_CONFIG_PATH, REPO_ROOT
from .model_registry import create_adapter
from .offline_probe import collect_probe_features_by_layer
from .runtime_dataset import build_runtime_splits


def _utc_timestamp() -> str:
    return time().__format__(".3f")


def _log(model_name: str, message: str) -> None:
    print(f"[{_utc_timestamp()}] [{model_name}] {message}", flush=True)


@dataclass(frozen=True)
class CandidateResult:
    batch_size: int
    status: str
    warmup_iterations: int
    timed_iterations: int
    mean_wall_s: float | None
    samples_per_s: float | None
    peak_memory_gib: float | None
    model_memory_gib: float | None
    error: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Canonical model name or slug")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DATASET_CONFIG_PATH,
        help="Dataset config used to materialize a short runtime split",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device for adapter runtime",
    )
    parser.add_argument(
        "--num-enabled",
        type=int,
        default=2,
        help="Representative num_enabled value used for the short throughput split",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Warmup iterations per candidate batch size",
    )
    parser.add_argument(
        "--timed-iters",
        type=int,
        default=2,
        help="Timed iterations per candidate batch size",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4096,
        help="Global hard cap for tested encode batch sizes",
    )
    parser.add_argument(
        "--candidate-batch-size",
        action="append",
        dest="candidate_batch_sizes",
        type=int,
        default=None,
        help="Optional explicit candidate batch size; can be repeated",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "encode_batch_tuning",
        help="Directory for per-model tuning JSON outputs",
    )
    return parser.parse_args()


def _set_perf_flags() -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _candidate_batch_sizes(*, runtime_default: int, max_batch_size: int) -> list[int]:
    base = int(runtime_default)
    if base <= 0:
        raise ValueError(f"runtime_default must be > 0, got {base}")
    hard_cap = int(max_batch_size)
    if hard_cap <= 0:
        raise ValueError(f"max_batch_size must be > 0, got {hard_cap}")
    if base >= hard_cap:
        return [base]
    upper = min(hard_cap, max(512, base * 4))
    candidates: list[int] = []
    value = base
    while value <= upper:
        candidates.append(int(value))
        value *= 2
    if candidates[-1] != upper and upper > candidates[-1]:
        candidates.append(int(upper))
    return sorted(set(int(value) for value in candidates))


def _cuda_gib(value: int) -> float:
    return float(value) / float(1024**3)


def _is_cuda_oom(error: BaseException) -> bool:
    if isinstance(error, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(error).lower()


def _clear_cuda_state(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _device_memory_gib(adapter: torch.nn.Module, device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    try:
        seen_storage_ptrs: set[int] = set()
        total_bytes = 0
        for parameter in adapter.parameters():
            if parameter.device.type != "cuda":
                continue
            storage = parameter.untyped_storage()
            storage_ptr = int(storage.data_ptr())
            if storage_ptr in seen_storage_ptrs:
                continue
            seen_storage_ptrs.add(storage_ptr)
            total_bytes += int(storage.nbytes())
        return _cuda_gib(total_bytes)
    except Exception:
        return None


def _build_short_split(
    *,
    adapter,
    dataset_config_path: Path,
    num_enabled: int,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return build_runtime_splits(
        config_path=dataset_config_path,
        device=torch.device("cpu"),
        batch_size=int(batch_size),
        channel_size_override=adapter.exact_benchmark_sequence_length(),
        channel_size_policy_override="model_native_exact",
        channel_size_source_override=f"adapter.{adapter.benchmark_sequence_length_source}",
        train_batches=1,
        val_batches=1,
        num_enabled=int(num_enabled),
    )


def _make_loader(
    *,
    split: dict[str, torch.Tensor],
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(split["x"], split["y_cls"], split["y_dense"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def _detect_attention_backend(adapter, *, device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device),
        "device_type": str(device.type),
        "torch_version": str(torch.__version__),
    }
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else int(torch.cuda.current_device())
        info["cuda_device_index"] = int(cuda_index)
        info["cuda_device_name"] = str(torch.cuda.get_device_name(cuda_index))
        info["torch_flash_sdp_enabled"] = bool(torch.backends.cuda.flash_sdp_enabled())
        info["torch_mem_efficient_sdp_enabled"] = bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()
        )
        info["torch_math_sdp_enabled"] = bool(torch.backends.cuda.math_sdp_enabled())
    try:
        import flash_attn  # noqa: F401
    except Exception as error:
        info["flash_attn_importable"] = False
        info["flash_attn_import_error"] = repr(error)
    else:
        info["flash_attn_importable"] = True

    adapter_attention_impl = getattr(adapter, "attention_implementation", None)
    if adapter_attention_impl is not None:
        info["adapter_attention_implementation"] = str(adapter_attention_impl)

    model = getattr(adapter, "model", None)
    if model is not None:
        info["model_class"] = str(model.__class__.__name__)
        model_config = getattr(model, "config", None)
        if model_config is not None:
            info["model_config_class"] = str(model_config.__class__.__name__)
            attn_impl = getattr(model_config, "_attn_implementation", None)
            if attn_impl is not None:
                info["model_config_attn_implementation"] = str(attn_impl)

    decoder_model = getattr(adapter, "decoder_model", None)
    if decoder_model is not None:
        info["decoder_model_class"] = str(decoder_model.__class__.__name__)

    decoder = getattr(adapter, "decoder", None)
    if decoder is not None:
        info["decoder_class"] = str(decoder.__class__.__name__)

    return info


def _benchmark_candidate(
    *,
    adapter,
    x_cpu: torch.Tensor,
    layers: tuple[int, ...],
    device: torch.device,
    warmup_iters: int,
    timed_iters: int,
) -> CandidateResult:
    if int(x_cpu.size(0)) <= 0:
        raise ValueError(f"Need at least one sample, got {tuple(x_cpu.shape)}")
    model_memory_gib = _device_memory_gib(adapter, device)
    _clear_cuda_state(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(int(warmup_iters)):
            x_device = x_cpu.to(device, non_blocking=True)
            with adapter.autocast_context(device):
                _ = adapter.forward_layer_dict(x_device, layers=layers)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            del x_device

        elapsed_times: list[float] = []
        peak_memory_gib = 0.0
        for _ in range(int(timed_iters)):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            start = perf_counter()
            x_device = x_cpu.to(device, non_blocking=True)
            with adapter.autocast_context(device):
                _ = adapter.forward_layer_dict(x_device, layers=layers)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_memory_gib = max(
                    peak_memory_gib,
                    _cuda_gib(torch.cuda.max_memory_allocated(device)),
                )
            elapsed_times.append(float(perf_counter() - start))
            del x_device

    mean_wall_s = sum(elapsed_times) / float(len(elapsed_times))
    return CandidateResult(
        batch_size=int(x_cpu.size(0)),
        status="ok",
        warmup_iterations=int(warmup_iters),
        timed_iterations=int(timed_iters),
        mean_wall_s=float(mean_wall_s),
        samples_per_s=float(int(x_cpu.size(0)) / mean_wall_s),
        peak_memory_gib=(None if device.type != "cuda" else float(peak_memory_gib)),
        model_memory_gib=None if model_memory_gib is None else float(model_memory_gib),
    )


def _benchmark_cached_candidate(
    *,
    adapter,
    split: dict[str, torch.Tensor],
    layers: tuple[int, ...],
    split_name: str,
    batch_size: int,
    device: torch.device,
    warmup_iters: int,
    timed_iters: int,
) -> CandidateResult:
    model_memory_gib = _device_memory_gib(adapter, device)
    _clear_cuda_state(device)

    def _run_once() -> tuple[int, float, float | None]:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        collected = collect_probe_features_by_layer(
            encoder=adapter,
            representation_fn=adapter.make_representation_fn(layers=layers, split=split_name),
            layers=layers,
            loader=_make_loader(split=split, batch_size=int(batch_size)),
            device=device,
            auto_mixed_precision=adapter.autocast_context(device),
            allow_crops=False,
        )
        peak_memory_gib = (
            None
            if device.type != "cuda"
            else _cuda_gib(torch.cuda.max_memory_allocated(device))
        )
        return (
            int(collected.timings["samples"]),
            float(collected.timings["total_s"]),
            peak_memory_gib,
        )

    with torch.inference_mode():
        for _ in range(int(warmup_iters)):
            _run_once()

        elapsed_times: list[float] = []
        peak_memory_gib = 0.0
        samples = 0
        for _ in range(int(timed_iters)):
            run_samples, run_elapsed_s, run_peak_memory_gib = _run_once()
            samples = int(run_samples)
            elapsed_times.append(float(run_elapsed_s))
            if run_peak_memory_gib is not None:
                peak_memory_gib = max(peak_memory_gib, float(run_peak_memory_gib))

    mean_wall_s = sum(elapsed_times) / float(len(elapsed_times))
    return CandidateResult(
        batch_size=int(batch_size),
        status="ok",
        warmup_iterations=int(warmup_iters),
        timed_iterations=int(timed_iters),
        mean_wall_s=float(mean_wall_s),
        samples_per_s=float(int(samples) / mean_wall_s),
        peak_memory_gib=(None if device.type != "cuda" else float(peak_memory_gib)),
        model_memory_gib=None if model_memory_gib is None else float(model_memory_gib),
    )


def _recommend_batch_size(results: list[CandidateResult]) -> tuple[int, float] | None:
    ok_results = [result for result in results if result.status == "ok" and result.samples_per_s is not None]
    if not ok_results:
        return None
    best_throughput = max(float(result.samples_per_s) for result in ok_results)
    threshold = 0.98 * best_throughput
    recommended = min(
        (
            result
            for result in ok_results
            if float(result.samples_per_s) >= threshold
        ),
        key=lambda result: int(result.batch_size),
    )
    return int(recommended.batch_size), float(best_throughput)


def _should_stop_after_candidate(results: list[CandidateResult]) -> bool:
    ok_results = [result for result in results if result.status == "ok" and result.samples_per_s is not None]
    if len(ok_results) < 2:
        return False
    last_result = ok_results[-1]
    previous_best = max(ok_results[:-1], key=lambda result: float(result.samples_per_s))
    if int(last_result.batch_size) < 2 * int(previous_best.batch_size):
        return False
    return float(last_result.samples_per_s) <= 1.02 * float(previous_best.samples_per_s)


def main() -> None:
    args = _parse_args()
    _set_perf_flags()
    actual_device = torch.device(str(args.device))
    if actual_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    spec, adapter = create_adapter(str(args.model))
    model_name = str(spec.name)
    adapter = adapter.to(actual_device)
    adapter.eval()

    _log(model_name, "phase: bootstrap adapter state")
    manifest, train_split, val_split = _build_short_split(
        adapter=adapter,
        dataset_config_path=args.dataset_config,
        num_enabled=int(args.num_enabled),
        batch_size=256,
    )
    adapter.prepare(manifest=manifest, train_split=train_split, val_split=val_split)
    adapter.prepare_runtime(device=actual_device)

    runtime_default_batch_size = int(adapter.default_encode_batch_size)
    candidate_batch_sizes = (
        sorted(set(int(value) for value in args.candidate_batch_sizes))
        if args.candidate_batch_sizes
        else _candidate_batch_sizes(
            runtime_default=runtime_default_batch_size,
            max_batch_size=int(args.max_batch_size),
        )
    )
    max_candidate_batch_size = max(int(value) for value in candidate_batch_sizes)

    _log(
        model_name,
        "phase: build representative short split "
        f"(runtime_default_batch_size={runtime_default_batch_size} candidates={candidate_batch_sizes})",
    )
    manifest, train_split, val_split = _build_short_split(
        adapter=adapter,
        dataset_config_path=args.dataset_config,
        num_enabled=int(args.num_enabled),
        batch_size=max_candidate_batch_size,
    )
    adapter.prepare(manifest=manifest, train_split=train_split, val_split=val_split)
    adapter.prepare_runtime(device=actual_device)

    attention_info = _detect_attention_backend(adapter, device=actual_device)
    available_layers = tuple(int(layer) for layer in adapter.available_layers)
    results: list[CandidateResult] = []
    for candidate_batch_size in candidate_batch_sizes:
        _log(
            model_name,
            f"candidate: batch_size={int(candidate_batch_size)} layers={len(available_layers)}",
        )
        try:
            if str(spec.env) == "tabular":
                probe_train_split = getattr(adapter, "probe_train_split", None) or train_split
                candidate_result = _benchmark_cached_candidate(
                    adapter=adapter,
                    split=probe_train_split,
                    layers=available_layers,
                    split_name="train",
                    batch_size=int(candidate_batch_size),
                    device=actual_device,
                    warmup_iters=int(args.warmup_iters),
                    timed_iters=int(args.timed_iters),
                )
            else:
                x_cpu = train_split["x"][: int(candidate_batch_size)].contiguous()
                candidate_result = _benchmark_candidate(
                    adapter=adapter,
                    x_cpu=x_cpu,
                    layers=available_layers,
                    device=actual_device,
                    warmup_iters=int(args.warmup_iters),
                    timed_iters=int(args.timed_iters),
                )
        except Exception as error:
            if _is_cuda_oom(error) or any(result.status == "ok" for result in results):
                _clear_cuda_state(actual_device)
                candidate_result = CandidateResult(
                    batch_size=int(candidate_batch_size),
                    status=("oom" if _is_cuda_oom(error) else "limit"),
                    warmup_iterations=int(args.warmup_iters),
                    timed_iterations=int(args.timed_iters),
                    mean_wall_s=None,
                    samples_per_s=None,
                    peak_memory_gib=None,
                    model_memory_gib=_device_memory_gib(adapter, actual_device),
                    error=repr(error),
                )
                results.append(candidate_result)
                _log(
                    model_name,
                    f"candidate: batch_size={int(candidate_batch_size)} "
                    f"-> {candidate_result.status} ({type(error).__name__})",
                )
                break
            raise
        results.append(candidate_result)
        _log(
            model_name,
            "candidate: "
            f"batch_size={int(candidate_batch_size)} "
            f"samples_per_s={candidate_result.samples_per_s:.2f} "
            f"peak_memory_gib={candidate_result.peak_memory_gib}",
        )
        if _should_stop_after_candidate(results):
            _log(
                model_name,
                f"candidate: batch_size={int(candidate_batch_size)} -> early stop on throughput plateau",
            )
            break

    recommendation = _recommend_batch_size(results)
    payload: dict[str, Any] = {
        "model": {
            "name": str(spec.name),
            "slug": str(spec.slug),
            "env": str(spec.env),
            "checkpoint": str(spec.checkpoint),
            "import_path": str(spec.import_path),
        },
        "device": str(actual_device),
        "dataset_config": str(args.dataset_config),
        "num_enabled": int(args.num_enabled),
        "runtime_default_batch_size": int(runtime_default_batch_size),
        "candidate_batch_sizes": list(candidate_batch_sizes),
        "available_layers": list(available_layers),
        "attention_info": attention_info,
        "results": [asdict(result) for result in results],
        "recommended_batch_size": None if recommendation is None else int(recommendation[0]),
        "best_samples_per_s": None if recommendation is None else float(recommendation[1]),
        "created_at_unix": float(time()),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.slug}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _log(
        model_name,
        f"done: wrote {out_path} recommended_batch_size={payload['recommended_batch_size']} "
        f"attention={payload['attention_info'].get('adapter_attention_implementation') or payload['attention_info'].get('model_config_attn_implementation')}",
    )
    print(out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
