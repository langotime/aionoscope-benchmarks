from __future__ import annotations

import argparse
import json
import sys
import threading
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from .constants import RESULTS_ROOT


def _discover_result_files(results_root: Path) -> list[str]:
    entries = sorted(path.name for path in (results_root / "models").glob("*.json"))
    if not entries:
        raise ValueError(f"{results_root / 'models'} does not list any result artifacts.")
    return entries


def _validate_result_payload(path: Path, payload: dict[str, object]) -> None:
    model = payload.get("model")
    dataset = payload.get("dataset")
    results = payload.get("results")
    if not isinstance(model, dict) or not isinstance(dataset, dict) or not isinstance(results, dict):
        raise ValueError(f"{path} is missing model/dataset/results sections.")
    for field in ("slug", "family", "checkpoint_name"):
        if field not in model:
            raise ValueError(f"{path} is missing model.{field}.")
    architecture = model.get("architecture")
    training = model.get("training")
    if not isinstance(architecture, dict) or "backbone" not in architecture:
        raise ValueError(f"{path} is missing model.architecture.backbone.")
    if not isinstance(training, dict) or "paradigm" not in training:
        raise ValueError(f"{path} is missing model.training.paradigm.")
    for field in ("benchmark_family", "benchmark_version", "num_enabled"):
        if field not in dataset:
            raise ValueError(f"{path} is missing dataset.{field}.")
    for field in ("categorical", "dense", "shared", "summary"):
        if field not in results:
            raise ValueError(f"{path} is missing results.{field}.")


@contextmanager
def _serve_directory(directory: Path):
    class _QuietHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format: str, *args) -> None:  # pragma: no cover
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), _QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _fetch_text(url: str) -> str:
    with urlopen(url, timeout=10) as response:  # noqa: S310
        return response.read().decode("utf-8")


def _copy_dev_results_tree(source_root: Path, destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    (destination_root / "dashboard.html").write_text(
        (source_root / "dashboard.html").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    destination_models = destination_root / "models"
    destination_models.mkdir(parents=True, exist_ok=True)
    for source_path in sorted((source_root / "models").glob("*.json")):
        (destination_models / source_path.name).write_text(
            source_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def run_dashboard_smoke(*, results_root: Path = RESULTS_ROOT, max_files: int = 3) -> dict[str, object]:
    dashboard_path = results_root / "dashboard.html"
    if not dashboard_path.exists():
        raise ValueError(f"Missing dashboard HTML: {dashboard_path}")
    dashboard_html = dashboard_path.read_text(encoding="utf-8")
    if 'const MODELS_LIST_PATH = "/models/list.txt";' not in dashboard_html:
        raise ValueError("Dashboard HTML no longer points at /models/list.txt.")
    if (results_root / "models" / "list.txt").exists():
        raise ValueError(
            "results/models/list.txt should not exist in the dev results tree; it is a deployment artifact."
        )

    discovered_files = _discover_result_files(results_root)
    selected_files = discovered_files[: max(1, int(max_files))]
    benchmark_pairs: set[tuple[str, str]] = set()

    with TemporaryDirectory() as temp_dir:
        served_root = Path(temp_dir)
        _copy_dev_results_tree(results_root, served_root)
        with _serve_directory(served_root) as base_url:
            served_dashboard_html = _fetch_text(f"{base_url}/dashboard.html")
            if "<title>Aionoscope Benchmarks Dashboard</title>" not in served_dashboard_html:
                raise ValueError("Served dashboard HTML did not contain the expected title banner.")
            for filename in selected_files:
                response_text = _fetch_text(f"{base_url}/models/{filename}")
                payload = json.loads(response_text)
                _validate_result_payload(results_root / "models" / filename, payload)
                dataset = payload["dataset"]
                benchmark_pairs.add((str(dataset["benchmark_family"]), str(dataset["benchmark_version"])))

    return {
        "dashboard_path": "/dashboard.html",
        "discovery_mode": "directory_listing_fallback",
        "checked_files": len(selected_files),
        "benchmark_pairs": sorted(benchmark_pairs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve results/ and verify dashboard discovery against real JSON artifacts.")
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_ROOT),
        help="Path to the results/ site root. Defaults to the repo's checked-in results directory.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=3,
        help="Maximum number of JSON artifacts to fetch through the local HTTP smoke server.",
    )
    args = parser.parse_args(argv)
    summary = run_dashboard_smoke(results_root=Path(args.results_root), max_files=int(args.max_files))
    print(
        "Dashboard smoke passed: "
        f"served {summary['dashboard_path']} via {summary['discovery_mode']}, "
        f"validated {summary['checked_files']} result files from {summary['benchmark_pairs']}.",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
