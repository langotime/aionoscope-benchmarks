#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from aionoscope_benchmarks.constants import DATASET_CONFIG_PATH, LIBRARY_ROOT, REPO_ROOT, RESULTS_ROOT
from aionoscope_benchmarks.manifold_config import (
    DEFAULT_MANIFOLD_CALIBRATION_MODELS,
    DEFAULT_MANIFOLD_CALIBRATION_TARGETS,
)
from aionoscope_benchmarks.manifold_viewer import build_viewer
from aionoscope_benchmarks.model_registry import MODEL_SPECS, canonical_model_name


ENV_DIR_BY_NAME = {
    "core": ".venv",
    "tabular": ".venv-tabular",
    "moment": ".venv-moment311",
    "mantis": ".venv-mantis2",
    "chronos": ".venv-chronos",
    "ttm": ".venv-ttm",
    "timemoe": ".venv-timemoe",
    "tempopfn": ".venv-tempopfn",
    "tirex": ".venv-tirex",
    "moirai": ".venv-moirai",
    "toto": ".venv-toto",
    "tivit": ".venv-tivit",
}


def _artifact_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models", default=None)
    parser.add_argument("--target", action="append", dest="targets", default=None)
    parser.add_argument("--layer", action="append", dest="layers", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--repeats-per-grid-point", type=int, default=None)
    parser.add_argument("--pca-dim", type=int, default=None)
    parser.add_argument("--geodesic-neighbor", action="append", dest="geodesic_neighbors", type=int, default=None)
    parser.add_argument("--view-grid-mode", choices=("linear", "log", "signed_log"), default=None)
    parser.add_argument("--view-range-max-abs", type=float, default=None)
    parser.add_argument("--view-log-min-abs", type=float, default=None)
    parser.add_argument("--plot-max-points", type=int, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=None)
    parser.add_argument("--dataset-config", type=Path, default=DATASET_CONFIG_PATH)
    parser.add_argument("--out-root", type=Path, default=RESULTS_ROOT / "manifold_calibration")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--generation-device", type=str, default="cpu")
    parser.add_argument(
        "--env-root",
        type=Path,
        default=REPO_ROOT,
        help="Directory containing the per-model .venv-* environments.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_args()


def _python_for_model(*, model_name: str, env_root: Path) -> Path:
    key = canonical_model_name(model_name)
    spec = MODEL_SPECS[key]
    env_dir = ENV_DIR_BY_NAME.get(spec.env)
    if env_dir is None:
        raise KeyError(f"No Python environment directory configured for env={spec.env!r}")
    python_path = env_root / env_dir / "bin" / "python"
    if not python_path.is_file():
        raise FileNotFoundError(
            f"Missing Python executable for model={key!r} env={spec.env!r}: {python_path}"
        )
    return python_path


def _extend_flag(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(REPO_ROOT), str(LIBRARY_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def main() -> None:
    args = _parse_args()
    models = tuple(args.models or DEFAULT_MANIFOLD_CALIBRATION_MODELS)
    targets = tuple(args.targets or DEFAULT_MANIFOLD_CALIBRATION_TARGETS)
    run_id = args.run_id or _artifact_run_id()
    run_root = args.out_root / run_id

    if not models:
        raise SystemExit("No models selected")
    if not targets:
        raise SystemExit("No targets selected")

    script_path = REPO_ROOT / "scripts" / "run_manifold_calibration.py"
    env = _subprocess_env()
    for index, model_name in enumerate(models, start=1):
        canonical_name = canonical_model_name(model_name)
        python_path = _python_for_model(model_name=canonical_name, env_root=args.env_root)
        cmd = [
            str(python_path),
            str(script_path),
            "--model",
            canonical_name,
            "--dataset-config",
            str(args.dataset_config),
            "--out-root",
            str(args.out_root),
            "--run-id",
            run_id,
            "--device",
            str(args.device),
            "--generation-device",
            str(args.generation_device),
            "--no-viewer",
        ]
        for target in targets:
            cmd.extend(["--target", str(target)])
        for layer in args.layers or ():
            cmd.extend(["--layer", str(int(layer))])
        for geodesic_neighbor in args.geodesic_neighbors or ():
            cmd.extend(["--geodesic-neighbor", str(int(geodesic_neighbor))])
        _extend_flag(cmd, "--max-layers", args.max_layers)
        _extend_flag(cmd, "--grid-size", args.grid_size)
        _extend_flag(cmd, "--repeats-per-grid-point", args.repeats_per_grid_point)
        _extend_flag(cmd, "--pca-dim", args.pca_dim)
        _extend_flag(cmd, "--view-grid-mode", args.view_grid_mode)
        _extend_flag(cmd, "--view-range-max-abs", args.view_range_max_abs)
        _extend_flag(cmd, "--view-log-min-abs", args.view_log_min_abs)
        _extend_flag(cmd, "--plot-max-points", args.plot_max_points)
        _extend_flag(cmd, "--encode-batch-size", args.encode_batch_size)
        if args.skip_plots or args.no_plots:
            cmd.append("--no-plots")

        print(f"[run {index}/{len(models)}] {canonical_name}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)

    if not args.no_viewer:
        viewer_path = run_root / "index.html"
        build_viewer(artifact_root=run_root, out_path=viewer_path)
        print(viewer_path)
    else:
        print(run_root)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except Exception as exc:  # pragma: no cover - CLI path
        print(f"[error] {exc}", file=sys.stderr)
        raise
