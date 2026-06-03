from __future__ import annotations

import argparse
from pathlib import Path

from aionoscope_benchmarks.constants import RESULTS_ROOT
from aionoscope_benchmarks.manifold_viewer import build_viewer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=RESULTS_ROOT / "manifolds",
        help="Root directory containing manifold artifacts.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_ROOT / "manifolds" / "index.html",
        help="Output static HTML viewer path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_viewer(artifact_root=args.artifact_root, out_path=args.out)
    print(args.out)


if __name__ == "__main__":
    main()
