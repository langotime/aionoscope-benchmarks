from __future__ import annotations

from aionoscope_benchmarks.manifold_config import ManifoldEvalConfig


def test_manifold_config_accepts_scalar_cli_like_values() -> None:
    config = ManifoldEvalConfig.from_mapping(
        {
            "models": "MantisV2",
            "targets": "sine_phase",
            "geodesic_neighbors": 2,
            "view_grid_mode": "signed_log",
            "view_range_max_abs": 1.0e6,
        }
    )

    assert config.models == ("MantisV2",)
    assert config.targets == ("sine_phase",)
    assert config.geodesic_neighbors == (2,)
    assert config.view_grid_mode == "signed_log"
    assert config.view_range_max_abs == 1.0e6
