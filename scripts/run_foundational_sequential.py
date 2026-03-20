#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from aionoscope_benchmarks.constants import DATASET_CONFIG_PATH, LIBRARY_ROOT, MODEL_RESULTS_ROOT, REPO_ROOT
from aionoscope_benchmarks.model_registry import MODEL_SPECS, all_foundational_model_names
from aionoscope_benchmarks.run_model import result_output_path
from aionoscope_benchmarks.runtime_dataset import resolve_requested_num_enabled_values


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
        help="Execution device passed through to run_model",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DATASET_CONFIG_PATH,
        help="Dataset config YAML passed through to run_model",
    )
    parser.add_argument(
        "--num-enabled",
        action="append",
        dest="num_enabled_values",
        type=int,
        default=None,
        help="Optional active num_enabled value override; can be repeated",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run models even if their JSON already exists",
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
    requested_num_enabled_values = resolve_requested_num_enabled_values(
        config_path=args.dataset_config,
        requested_num_enabled_values=args.num_enabled_values,
    )

    for index, model in enumerate(models, start=1):
        spec = MODEL_SPECS[model]
        pending_num_enabled_values = [
            int(num_enabled)
            for num_enabled in requested_num_enabled_values
            if args.force
            or not result_output_path(
                out_dir=MODEL_RESULTS_ROOT,
                model_slug=spec.slug,
                num_enabled=int(num_enabled),
            ).is_file()
        ]
        if not pending_num_enabled_values:
            existing_paths = [
                result_output_path(
                    out_dir=MODEL_RESULTS_ROOT,
                    model_slug=spec.slug,
                    num_enabled=int(num_enabled),
                )
                for num_enabled in requested_num_enabled_values
            ]
            print(
                f"[skip {index}/{len(models)}] {model}: "
                + ", ".join(str(path) for path in existing_paths)
            )
            continue

        python_path = ENV_TO_PYTHON.get(spec.env)
        if python_path is None:
            raise KeyError(f"No Python executable configured for env={spec.env!r} model={model!r}")
        if not python_path.is_file():
            raise FileNotFoundError(f"Missing Python executable for env={spec.env!r}: {python_path}")

        cmd = [
            str(python_path),
            "-m",
            "aionoscope_benchmarks.run_model",
            "--model",
            model,
            "--dataset-config",
            str(args.dataset_config),
            "--device",
            str(args.device),
        ]
        for num_enabled in pending_num_enabled_values:
            cmd.extend(["--num-enabled", str(int(num_enabled))])
        env = os.environ.copy()
        library_pythonpath = str(LIBRARY_ROOT)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            library_pythonpath
            if not existing_pythonpath
            else library_pythonpath + os.pathsep + existing_pythonpath
        )
        print(f"[run {index}/{len(models)}] {model}: {' '.join(cmd)}")
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            env=env,
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except Exception as exc:  # pragma: no cover - CLI path
        print(f"[error] {exc}", file=sys.stderr)
        raise
