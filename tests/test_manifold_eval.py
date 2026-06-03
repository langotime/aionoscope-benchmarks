from __future__ import annotations

import math

import numpy as np

from aionoscope_benchmarks.manifold_eval import (
    compute_manifold_layer_evaluation,
    knn_geodesic_distance_matrix,
    maybe_pca,
    target_distance_matrix,
)


def test_circle_target_distance_wraps_by_period() -> None:
    coords = np.asarray([0.0, math.pi / 2.0, 3.0 * math.pi / 2.0])

    distances = target_distance_matrix(coords, geometry="circle", period=2.0 * math.pi)

    assert distances[0, 2] == math.pi / 2.0
    assert distances[1, 2] == math.pi


def test_interval_manifold_metrics_are_near_perfect_for_straight_line() -> None:
    coords = np.linspace(0.0, 1.0, 16)
    features = np.column_stack([coords, np.zeros_like(coords)])

    evaluation = compute_manifold_layer_evaluation(
        train_features=features,
        train_grid_index=np.arange(coords.size),
        train_coordinates=coords,
        val_features=features,
        val_coordinates=coords,
        geometry="interval",
        geodesic_neighbors=(2, 4),
        pca_dim=8,
    )

    assert evaluation.metrics["spearman_latent_vs_linear"] == 1.0
    assert evaluation.metrics["spearman_latent_vs_geodesic"] > 0.999
    assert evaluation.metrics["knn_recall_at_1"] == 1.0
    assert evaluation.metrics["projection_r2"] == 1.0
    assert evaluation.metrics["graph_by_k"]["2"]["connected"] is True


def test_plot_downsampling_only_limits_distance_matrices() -> None:
    coords = np.linspace(0.0, 1.0, 16)
    features = np.column_stack([coords, np.zeros_like(coords)])

    evaluation = compute_manifold_layer_evaluation(
        train_features=features,
        train_grid_index=np.arange(coords.size),
        train_coordinates=coords,
        geometry="interval",
        geodesic_neighbors=(2,),
        pca_dim=8,
        plot_max_points=4,
    )
    plot_data = evaluation.plot_data

    assert plot_data["source_grid_points"] == 16
    assert plot_data["centroid_grid_points"] == 16
    assert plot_data["distance_grid_points"] == 4
    assert len(plot_data["centroids"]) == 16
    assert len(plot_data["centroid_coordinates"]) == 16
    assert len(plot_data["latent_distance"]) == 4
    assert len(plot_data["linear_distance"]) == 4


def test_circle_manifold_metrics_detect_closed_neighbors() -> None:
    coords = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    features = np.column_stack([np.cos(coords), np.sin(coords)])

    evaluation = compute_manifold_layer_evaluation(
        train_features=features,
        train_grid_index=np.arange(coords.size),
        train_coordinates=coords,
        geometry="circle",
        period=2.0 * math.pi,
        geodesic_neighbors=(2, 4),
        pca_dim=8,
    )

    assert evaluation.metrics["cycle_neighbor_wrap_score"] == 1.0
    assert evaluation.metrics["graph_by_k"]["2"]["connected"] is True
    assert evaluation.metrics["spearman_latent_vs_geodesic"] > 0.95


def test_zero_neighbor_geodesic_graph_reports_disconnected() -> None:
    linear_distances = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )

    geodesic, payload = knn_geodesic_distance_matrix(linear_distances, k=0)

    assert payload["connected"] is False
    assert payload["finite_pair_fraction"] == 0.0
    assert not np.isfinite(geodesic[0, 1])


def test_maybe_pca_sanitizes_nonfinite_inputs() -> None:
    train = np.asarray(
        [
            [0.0, 1.0, np.nan, 3.0],
            [1.0, 2.0, np.inf, 4.0],
            [2.0, 3.0, -np.inf, 5.0],
            [3.0, 4.0, 6.0, 6.0],
            [4.0, 5.0, 7.0, 7.0],
        ]
    )
    val = train + 0.5

    train_pca, val_pca, payload = maybe_pca(train, pca_dim=2, val_features=val)

    assert payload["applied"] is True
    assert payload["train_sanitize"]["finite_input_fraction"] < 1.0
    assert np.isfinite(train_pca).all()
    assert val_pca is not None
    assert np.isfinite(val_pca).all()


def test_maybe_pca_falls_back_when_svd_does_not_converge(monkeypatch) -> None:
    class FailingPCA:
        def __init__(self, *, n_components: int, svd_solver: str) -> None:
            self.n_components = n_components
            self.svd_solver = svd_solver

        def fit_transform(self, train: np.ndarray) -> np.ndarray:
            raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr("aionoscope_benchmarks.manifold_eval.PCA", FailingPCA)
    train = np.arange(20, dtype=np.float64).reshape(5, 4)

    train_pca, val_pca, payload = maybe_pca(train, pca_dim=2, val_features=train)

    assert payload["applied"] is False
    assert payload["pca_error"] == "SVD did not converge"
    assert train_pca.shape == train.shape
    assert val_pca is not None
    assert val_pca.shape == train.shape
