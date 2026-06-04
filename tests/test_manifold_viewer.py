from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aionoscope_benchmarks.manifold_eval import compute_manifold_layer_evaluation
from aionoscope_benchmarks.manifold_viewer import build_viewer
from aionoscope_benchmarks.manifold_viz import write_json, write_visualization_bundle


def test_manifold_plot_json_is_strict_browser_json(tmp_path: Path) -> None:
    path = tmp_path / "plot_data.json"
    write_json(
        path,
        {
            "finite": 1.0,
            "positive_inf": float("inf"),
            "negative_inf": -float("inf"),
            "nan": float("nan"),
            "nested": [[0.0, float("inf")]],
        },
    )

    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    assert json.loads(text) == {
        "finite": 1.0,
        "positive_inf": None,
        "negative_inf": None,
        "nan": None,
        "nested": [[0.0, None]],
    }


def test_manifold_visualization_bundle_and_viewer_are_static_artifacts(tmp_path: Path) -> None:
    coords = np.linspace(0.0, 1.0, 8)
    features = np.column_stack([coords, np.zeros_like(coords)])
    evaluation = compute_manifold_layer_evaluation(
        train_features=features,
        train_grid_index=np.arange(coords.size),
        train_coordinates=coords,
        val_features=features,
        val_coordinates=coords,
        geometry="interval",
        geodesic_neighbors=(2,),
        pca_dim=8,
    )
    run_root = tmp_path / "run"
    target_dir = run_root / "ToyModel" / "linear_trend_slope"
    visualizations = write_visualization_bundle(
        out_dir=target_dir / "plots",
        stem="toy__linear_trend_slope__layer_0",
        plot_data=evaluation.plot_data,
        metrics=evaluation.metrics,
        title="Toy / linear_trend_slope / layer 0",
    )
    assert set(visualizations) == {"plot_data_json", "distance_data_json"}
    metrics_path = target_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "model": {"name": "ToyModel", "slug": "ToyModel"},
                "target": {
                    "target_name": "linear_trend_slope",
                    "geometry": "interval",
                },
                "summary": {},
                "by_layer": {"0": evaluation.metrics},
                "visualizations": {"0": visualizations},
            }
        ),
        encoding="utf-8",
    )

    viewer_path = run_root / "index.html"
    build_viewer(artifact_root=run_root, out_path=viewer_path)

    assert viewer_path.exists()
    manifest_path = run_root / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "manifold_viewer_manifest_v1"
    assert manifest["records"] == [
        {
            "run": "run",
            "model": "ToyModel",
            "model_slug": "ToyModel",
            "target": "linear_trend_slope",
            "target_name": "linear_trend_slope",
            "sweep": {},
            "geometry": "interval",
            "metrics_json": "ToyModel/linear_trend_slope/metrics.json",
            "layers": [
                {
                    "layer": "0",
                    "paths": {
                        "plot_data_json": "ToyModel/linear_trend_slope/plots/toy__linear_trend_slope__layer_0_plot_data.json",
                        "distance_data_json": "ToyModel/linear_trend_slope/plots/toy__linear_trend_slope__layer_0_distance_data.json",
                    },
                }
            ],
        }
    ]
    html = viewer_path.read_text(encoding="utf-8")
    assert "Aionoscope Manifold Viewer" in html
    assert "ToyModel" not in html
    assert "const records =" not in html
    assert 'const MANIFEST_PATH = "manifest.json"' in html
    assert "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js" in html
    assert "Centroid path" in html
    assert "Distance scatter" in html
    assert "Distance heatmap" in html
    assert "distance data JSON" in html
    assert "distance-details" in html
    assert "loadDistanceData" in html
    assert "Spearman: latent vs linear" in html
    assert "plot data JSON" in html
    # multi-select comparison mechanism (overlay + side-by-side)
    assert 'id="comparison"' in html
    assert "add-comparison" in html
    assert 'data-remove=' in html
    assert 'class="sbs-grid"' in html
    assert "renderSideBySide" in html
    assert "procrustesRotation" in html
    assert html.index('<select id="geometry">') < html.index('<select id="target">')
    assert "all geometries" in html
    plot_data_path = Path(visualizations["plot_data_json"])
    distance_data_path = Path(visualizations["distance_data_json"])
    assert plot_data_path.exists()
    assert distance_data_path.exists()
    assert plot_data_path.stat().st_size > 0
    assert distance_data_path.stat().st_size > 0
    plot_payload = json.loads(plot_data_path.read_text(encoding="utf-8"))
    distance_payload = json.loads(distance_data_path.read_text(encoding="utf-8"))
    assert "distance_data_json" in plot_payload
    assert "latent_distance" not in plot_payload
    assert "linear_distance" not in plot_payload
    assert "geodesic_distance" not in plot_payload
    assert "distance_grid_points" not in plot_payload
    assert "plot_indices" not in plot_payload
    assert plot_payload["centroid_grid_points"] == coords.size
    assert "latent_distance" in distance_payload
    assert "linear_distance" in distance_payload
    assert distance_payload["distance_grid_points"] == coords.size
    assert not list((target_dir / "plots").glob("*.png"))
