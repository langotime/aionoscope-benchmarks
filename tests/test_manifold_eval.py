from __future__ import annotations

import math

import numpy as np

from aionoscope_benchmarks.manifold_eval import (
    compute_manifold_layer_evaluation,
    knn_geodesic_distance_matrix,
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
