from __future__ import annotations

from aionoscope_benchmarks.constants import RESULTS_ROOT
from aionoscope_benchmarks.dashboard_smoke import run_dashboard_smoke


def test_dashboard_smoke_serves_checked_in_results() -> None:
    summary = run_dashboard_smoke(results_root=RESULTS_ROOT, max_files=2)

    assert summary["dashboard_path"] == "/dashboard.html"
    assert summary["discovery_mode"] == "directory_listing_fallback"
    assert summary["checked_files"] == 2
    assert summary["benchmark_pairs"] == [("aiono_basic_components", "v2")]
