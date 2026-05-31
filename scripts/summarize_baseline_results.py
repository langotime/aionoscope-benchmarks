#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aionoscope_benchmarks.constants import MODEL_RESULTS_ROOT, REPO_ROOT


def _stat_median(value: object) -> float | None:
    if isinstance(value, dict) and "median" in value:
        median = value["median"]
        return None if median is None else float(median)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _stat_std(value: object) -> float | None:
    if isinstance(value, dict) and "std" in value:
        std = value["std"]
        return None if std is None else float(std)
    return None


def _compact(value: float | None) -> str:
    return "" if value is None else f"{float(value):.6g}"


def _baseline_rows(results_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("Baseline-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("model", {})
        if not isinstance(model, dict) or model.get("type") != "baseline":
            continue
        dataset = payload.get("dataset", {})
        results = payload.get("results", {})
        summary = results.get("summary", {}) if isinstance(results, dict) else {}
        if not isinstance(dataset, dict) or not isinstance(summary, dict):
            continue
        dense_layers = summary.get("macro_best_layers", {})
        dense_r2 = dense_layers.get("r2", {}) if isinstance(dense_layers, dict) else {}
        dense_pearson = dense_layers.get("pearson", {}) if isinstance(dense_layers, dict) else {}
        best_auc = summary.get("best_auc", {})
        best_auprc = summary.get("best_auprc", {})
        baseline = model.get("baseline", {}) if isinstance(model.get("baseline"), dict) else {}
        rows.append(
            {
                "file": path.name,
                "baseline": baseline.get("name", model.get("name")),
                "baseline_family": baseline.get("family", ""),
                "channel_size": dataset.get("channel_size"),
                "num_enabled": dataset.get("num_enabled"),
                "macro_auroc_median": _stat_median(best_auc.get("macro_auc") if isinstance(best_auc, dict) else None),
                "macro_auroc_std": _stat_std(best_auc.get("macro_auc") if isinstance(best_auc, dict) else None),
                "macro_auprc_median": _stat_median(
                    best_auprc.get("macro_auprc") if isinstance(best_auprc, dict) else None
                ),
                "macro_auprc_std": _stat_std(best_auprc.get("macro_auprc") if isinstance(best_auprc, dict) else None),
                "macro_r2_median": _stat_median(dense_r2.get("value") if isinstance(dense_r2, dict) else None),
                "macro_r2_std": _stat_std(dense_r2.get("value") if isinstance(dense_r2, dict) else None),
                "macro_pearson_median": _stat_median(
                    dense_pearson.get("value") if isinstance(dense_pearson, dict) else None
                ),
                "macro_pearson_std": _stat_std(
                    dense_pearson.get("value") if isinstance(dense_pearson, dict) else None
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "baseline",
        "baseline_family",
        "channel_size",
        "num_enabled",
        "macro_auroc_median",
        "macro_auroc_std",
        "macro_auprc_median",
        "macro_auprc_std",
        "macro_r2_median",
        "macro_r2_std",
        "macro_pearson_median",
        "macro_pearson_std",
        "file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _compact(row[key]) if isinstance(row.get(key), float) else row.get(key, "")
                    for key in fieldnames
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=MODEL_RESULTS_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "baseline_calibration_by_length.csv",
    )
    args = parser.parse_args()
    rows = _baseline_rows(args.results_dir)
    if not rows:
        raise SystemExit(f"No baseline JSON files found in {args.results_dir}")
    _write_csv(args.out, rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
