from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .constants import DATASET_CONFIG_PATH, MODEL_RESULTS_ROOT
from .model_registry import all_foundational_model_names
from .run_model import run_single_model_for_num_enabled
from .runtime_dataset import resolve_requested_num_enabled_values


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
        "--dataset-config",
        type=Path,
        default=DATASET_CONFIG_PATH,
        help="Dataset config YAML used to resolve active num_enabled runs",
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

    requested_num_enabled_values = resolve_requested_num_enabled_values(
        config_path=args.dataset_config,
        requested_num_enabled_values=args.num_enabled_values,
    )
    failures: list[tuple[str, int, str]] = []
    for model in models:
        for num_enabled in requested_num_enabled_values:
            try:
                out_path = run_single_model_for_num_enabled(
                    model_name=model,
                    num_enabled=int(num_enabled),
                    dataset_config_path=args.dataset_config,
                    out_dir=args.out_dir,
                    device=torch.device(str(args.device)),
                )
                print(f"[ok] {model} num_enabled={int(num_enabled)}: {out_path}")
            except Exception as exc:  # pragma: no cover - CLI path
                failures.append((model, int(num_enabled), repr(exc)))
                print(f"[failed] {model} num_enabled={int(num_enabled)}: {exc}")
                if not args.continue_on_error:
                    raise

    if failures:
        joined = "; ".join(
            f"{model} num_enabled={int(num_enabled)} -> {error}"
            for model, num_enabled, error in failures
        )
        raise SystemExit(f"Failed models: {joined}")


if __name__ == "__main__":
    main()
