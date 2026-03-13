#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from aionoscope_benchmarks.constants import LIBRARY_ROOT, MODEL_RESULTS_ROOT, REPO_ROOT
from aionoscope_benchmarks.model_registry import MODEL_SPECS, all_foundational_model_names


ENV_TO_PYTHON = {
    "core": REPO_ROOT / ".venv" / "bin" / "python",
    "tabular": REPO_ROOT / ".venv-tabular" / "bin" / "python",
    "moment": REPO_ROOT / ".venv-moment311" / "bin" / "python",
    "mantis": REPO_ROOT / ".venv-mantis2" / "bin" / "python",
    "chronos": REPO_ROOT / ".venv-chronos" / "bin" / "python",
    "ttm": REPO_ROOT / ".venv-ttm" / "bin" / "python",
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

    for index, model in enumerate(models, start=1):
        spec = MODEL_SPECS[model]
        result_path = MODEL_RESULTS_ROOT / f"{spec.slug}.json"
        if result_path.is_file() and not args.force:
            print(f"[skip {index}/{len(models)}] {model}: {result_path}")
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
            "--device",
            str(args.device),
        ]
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
