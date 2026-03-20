#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_REPO_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from aionoscope_benchmarks.constants import DATASET_CONFIG_PATH, LIBRARY_ROOT, REPO_ROOT
from aionoscope_benchmarks.model_registry import MODEL_SPECS, all_foundational_model_names


ENV_TO_PYTHON = {
    "core": REPO_ROOT / ".venv" / "bin" / "python",
    "tabular": REPO_ROOT / ".venv-tabular" / "bin" / "python",
    "moment": REPO_ROOT / ".venv-moment311" / "bin" / "python",
    "mantis": REPO_ROOT / ".venv-mantis2" / "bin" / "python",
    "chronos": REPO_ROOT / ".venv-chronos" / "bin" / "python",
    "ttm": REPO_ROOT / ".venv-ttm" / "bin" / "python",
    "timemoe": REPO_ROOT / ".venv-timemoe" / "bin" / "python",
    "tempopfn": REPO_ROOT / ".venv-tempopfn" / "bin" / "python",
    "tirex": REPO_ROOT / ".venv-tirex" / "bin" / "python",
    "moirai": REPO_ROOT / ".venv-moirai" / "bin" / "python",
    "toto": REPO_ROOT / ".venv-toto" / "bin" / "python",
    "tivit": REPO_ROOT / ".venv-tivit" / "bin" / "python",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Specific model name or slug; can be repeated",
    )
    parser.add_argument(
        "--start-at",
        type=str,
        default=None,
        help="Start from this model name or slug within the foundational list",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Execution device passed to the tuner",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DATASET_CONFIG_PATH,
        help="Dataset config used for short encode-throughput splits",
    )
    parser.add_argument(
        "--num-enabled",
        type=int,
        default=2,
        help="Representative num_enabled value used for short encode-throughput splits",
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
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "encode_batch_tuning",
        help="Directory for per-model tuning JSON outputs",
    )
    return parser.parse_args()


def _resolve_models(args: argparse.Namespace) -> list[str]:
    models = [str(model) for model in (args.models or all_foundational_model_names())]
    if args.start_at is None:
        return models
    start_token = str(args.start_at)
    for index, model in enumerate(models):
        spec = MODEL_SPECS[model]
        if start_token in {model, spec.slug}:
            return models[index:]
    raise KeyError(f"Unknown start model: {start_token!r}")


def main() -> None:
    args = _parse_args()
    models = _resolve_models(args)
    if not models:
        raise SystemExit("No models selected")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for index, model in enumerate(models, start=1):
        spec = MODEL_SPECS[model]
        python_path = ENV_TO_PYTHON.get(spec.env)
        if python_path is None:
            raise KeyError(f"No Python executable configured for env={spec.env!r} model={model!r}")
        if not python_path.is_file():
            raise FileNotFoundError(f"Missing Python executable for env={spec.env!r}: {python_path}")

        cmd = [
            str(python_path),
            "-m",
            "aionoscope_benchmarks.tune_encode_batch",
            "--model",
            model,
            "--dataset-config",
            str(args.dataset_config),
            "--device",
            str(args.device),
            "--num-enabled",
            str(int(args.num_enabled)),
            "--warmup-iters",
            str(int(args.warmup_iters)),
            "--timed-iters",
            str(int(args.timed_iters)),
            "--max-batch-size",
            str(int(args.max_batch_size)),
            "--out-dir",
            str(args.out_dir),
        ]
        env = os.environ.copy()
        library_pythonpath = str(LIBRARY_ROOT)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            library_pythonpath
            if not existing_pythonpath
            else library_pythonpath + os.pathsep + existing_pythonpath
        )
        print(f"[run {index}/{len(models)}] {model}: {' '.join(cmd)}", flush=True)
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            env=env,
        )

        payload_path = args.out_dir / f"{spec.slug}.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        summary_row = {
            "model": str(spec.name),
            "slug": str(spec.slug),
            "env": str(spec.env),
            "runtime_default_batch_size": int(payload["runtime_default_batch_size"]),
            "recommended_batch_size": payload["recommended_batch_size"],
            "best_samples_per_s": payload["best_samples_per_s"],
            "attention": payload["attention_info"].get("adapter_attention_implementation")
            or payload["attention_info"].get("model_config_attn_implementation"),
        }
        summary_rows.append(summary_row)
        print(
            f"[ok {index}/{len(models)}] {model}: "
            f"default={summary_row['runtime_default_batch_size']} "
            f"recommended={summary_row['recommended_batch_size']} "
            f"attention={summary_row['attention']}",
            flush=True,
        )

    summary_path = args.out_dir / "foundational_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except Exception as exc:  # pragma: no cover - CLI path
        print(f"[error] {exc}", file=sys.stderr)
        raise
