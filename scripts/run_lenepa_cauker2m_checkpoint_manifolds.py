#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from aionoscope_benchmarks.constants import REPO_ROOT, RESULTS_ROOT
from aionoscope_benchmarks.manifold_viewer import build_viewer_manifest


BASE_CHECKPOINT_DIR = Path(
    "/mnt/t0-train-shared/lejepa/pretrain/"
    "CAUKER2M_L5000_LENEPA_SIGREGT2p5_L0-8_PD0_PROJ_LR2x_MSSE_PATCHNORM_D256_"
    "OPTOYTSBCBAL_s0_UCRinterp5000"
)
CONTINUATION_CHECKPOINT_DIR = Path(
    "/mnt/t0-train-shared/lejepa/pretrain/"
    "CAUKER2M_L5000_LENEPA_SIGREGT2p5_L0-8_PD0_PROJ_LR2x_MSSE_PATCHNORM_D256_"
    "OPTOYTSBCBAL_s0_UCRinterp5000_CONT200K_LOCAL"
)

LINEAR_TARGETS = (
    "sine_phase",
    "sine_frequency_hz",
    "sine_amplitude",
    "spike_time_frac",
    "gaussian_time_frac",
    "linear_trend_slope",
)
SIGNED_LOG_TARGETS = ("linear_trend_slope",)
TARGET_ARTIFACTS = (*LINEAR_TARGETS, "linear_trend_slope__signed_log")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        type=Path,
        default=None,
        help="Directory containing chkpt_<step>.pt files. Defaults to the two CauKer2M run dirs.",
    )
    parser.add_argument("--out-root", type=Path, default=RESULTS_ROOT / "manifolds")
    parser.add_argument("--run-id", type=str, default=".")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--generation-device", type=str, default="cpu")
    parser.add_argument("--grid-size", type=int, default=1024)
    parser.add_argument("--repeats-per-grid-point", type=int, default=1)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--plot-max-points", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=220)
    parser.add_argument("--force", action="store_true", help="Rerun checkpoints even when all metrics exist.")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--manifest-every",
        type=int,
        default=10,
        help="Rebuild results/manifolds/manifest.json after every N completed checkpoints.",
    )
    return parser.parse_args()


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("chkpt_"):
        raise ValueError(f"Unexpected checkpoint filename: {path}")
    return int(stem.removeprefix("chkpt_"))


def _discover_checkpoints(dirs: tuple[Path, ...]) -> list[tuple[int, Path]]:
    by_step: dict[int, Path] = {}
    for directory in dirs:
        for path in directory.glob("chkpt_*.pt"):
            by_step[_checkpoint_step(path)] = path
    checkpoints = [(step, by_step[step]) for step in sorted(by_step)]
    if not checkpoints:
        raise FileNotFoundError(f"No chkpt_*.pt files found under {[str(d) for d in dirs]}")
    return [(index, path) for index, (_, path) in enumerate(checkpoints, start=1)]


def _expected_metrics(out_root: Path, checkpoint_path: Path) -> list[Path]:
    checkpoint_step = _checkpoint_step(checkpoint_path)
    checkpoint_dir = f"ckpt_{checkpoint_step:06d}"
    return [
        out_root / "LeNEPA-CauKer2M" / checkpoint_dir / target / "metrics.json"
        for target in TARGET_ARTIFACTS
    ]


def _extend_flag(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _runner_command(
    *,
    script_path: Path,
    args: argparse.Namespace,
    checkpoint_index: int,
    checkpoint_path: Path,
    targets: tuple[str, ...],
    view_grid_mode: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--model",
        "LeNEPA-CauKer2M",
        "--out-root",
        str(args.out_root),
        "--run-id",
        str(args.run_id),
        "--device",
        str(args.device),
        "--generation-device",
        str(args.generation_device),
        "--lenepa-training-checkpoint",
        str(checkpoint_path),
        "--checkpoint-index",
        str(int(checkpoint_index)),
        "--view-grid-mode",
        view_grid_mode,
        "--view-range-max-abs",
        "1000000",
        "--no-viewer",
    ]
    for target in targets:
        cmd.extend(["--target", target])
    _extend_flag(cmd, "--grid-size", args.grid_size)
    _extend_flag(cmd, "--repeats-per-grid-point", args.repeats_per_grid_point)
    _extend_flag(cmd, "--pca-dim", args.pca_dim)
    _extend_flag(cmd, "--plot-max-points", args.plot_max_points)
    _extend_flag(cmd, "--encode-batch-size", args.encode_batch_size)
    if args.skip_plots:
        cmd.append("--skip-plots")
    return cmd


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing else os.pathsep.join([str(REPO_ROOT), existing])
    return env


def main() -> None:
    args = _parse_args()
    checkpoint_dirs = tuple(args.checkpoint_dir or (BASE_CHECKPOINT_DIR, CONTINUATION_CHECKPOINT_DIR))
    checkpoints = _discover_checkpoints(checkpoint_dirs)
    selected = [
        (index, path)
        for index, path in checkpoints
        if int(args.start_index) <= int(index) <= int(args.end_index)
    ]
    if not selected:
        raise SystemExit(
            f"No checkpoints selected for index range {args.start_index}..{args.end_index}"
        )

    script_path = REPO_ROOT / "scripts" / "run_manifold_calibration.py"
    env = _subprocess_env()
    completed_since_manifest = 0
    for ordinal, (checkpoint_index, checkpoint_path) in enumerate(selected, start=1):
        expected_metrics = _expected_metrics(args.out_root, checkpoint_path)
        if not args.force and all(path.is_file() for path in expected_metrics):
            print(
                f"[skip {ordinal}/{len(selected)}] checkpoint #{checkpoint_index:03d} "
                f"{checkpoint_path.name}: all target metrics already exist",
                flush=True,
            )
            continue
        print(
            f"[checkpoint {ordinal}/{len(selected)}] #{checkpoint_index:03d} "
            f"{checkpoint_path.name}",
            flush=True,
        )
        for targets, view_grid_mode in (
            (LINEAR_TARGETS, "linear"),
            (SIGNED_LOG_TARGETS, "signed_log"),
        ):
            cmd = _runner_command(
                script_path=script_path,
                args=args,
                checkpoint_index=checkpoint_index,
                checkpoint_path=checkpoint_path,
                targets=targets,
                view_grid_mode=view_grid_mode,
            )
            print(f"[run] {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
        completed_since_manifest += 1
        if args.manifest_every > 0 and completed_since_manifest >= args.manifest_every:
            manifest_path = build_viewer_manifest(artifact_root=args.out_root)
            print(f"[manifest] {manifest_path}", flush=True)
            completed_since_manifest = 0

    manifest_path = build_viewer_manifest(artifact_root=args.out_root)
    print(f"[manifest] {manifest_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
