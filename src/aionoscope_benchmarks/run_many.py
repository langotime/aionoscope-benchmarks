from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .constants import MODEL_RESULTS_ROOT
from .model_registry import all_foundational_model_names
from .run_model import run_single_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Specific model name or slug; can be repeated. Use `all` for the full foundational list.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MODEL_RESULTS_ROOT,
        help="Output directory for JSON result files",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running other models after a failure",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.models or "all" in args.models:
        models = all_foundational_model_names()
    else:
        models = [str(model) for model in args.models]

    failures: list[tuple[str, str]] = []
    for model in models:
        try:
            out_path = run_single_model(
                model_name=model,
                out_dir=args.out_dir,
                device=torch.device(str(args.device)),
            )
            print(f"[ok] {model}: {out_path}")
        except Exception as exc:  # pragma: no cover - CLI path
            failures.append((model, repr(exc)))
            print(f"[failed] {model}: {exc}")
            if not args.continue_on_error:
                raise

    if failures:
        joined = "; ".join(f"{model} -> {error}" for model, error in failures)
        raise SystemExit(f"Failed models: {joined}")


if __name__ == "__main__":
    main()
