from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class ManifoldLayerEvaluation:
    metrics: dict[str, Any]
    plot_data: dict[str, Any]


def _as_numpy_2d(features: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
    if arr.ndim != 2:
        raise ValueError(f"features must be 2D [N, D], got {arr.shape}")
    return arr.astype(np.float64, copy=False)


def _as_numpy_1d(values: torch.Tensor | np.ndarray | list[float] | list[int]) -> np.ndarray:
    arr = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"values must be 1D, got {arr.shape}")
    return arr.astype(np.float64, copy=False)


def _json_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value_float = float(value)
    return value_float if math.isfinite(value_float) else None


def _sanitize_feature_array(features: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(features, dtype=np.float64)
    finite = np.isfinite(arr)
    if bool(finite.all()):
        return arr, {
            "finite_input_fraction": 1.0,
            "nonfinite_replacement": None,
        }

    finite_values = arr[finite]
    if finite_values.size:
        posinf = float(np.max(finite_values))
        neginf = float(np.min(finite_values))
    else:
        posinf = 0.0
        neginf = 0.0
    sanitized = np.nan_to_num(arr, nan=0.0, posinf=posinf, neginf=neginf)
    return sanitized, {
        "finite_input_fraction": _json_float(float(np.mean(finite))),
        "nonfinite_replacement": {
            "nan": 0.0,
            "posinf": posinf,
            "neginf": neginf,
        },
    }


def _upper_triangular_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"distance matrix must be square, got {matrix.shape}")
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return matrix[rows, cols]


def _plot_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, num=int(max_points)).round().astype(np.int64))


def _safe_corr(x: np.ndarray, y: np.ndarray, *, kind: str) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    x_valid = x[mask]
    y_valid = y[mask]
    if float(np.std(x_valid)) == 0.0 or float(np.std(y_valid)) == 0.0:
        return None
    if kind == "pearson":
        return _json_float(pearsonr(x_valid, y_valid).statistic)
    if kind == "spearman":
        return _json_float(spearmanr(x_valid, y_valid).statistic)
    raise ValueError(f"Unsupported correlation kind: {kind}")


def target_distance_matrix(
    coordinates: np.ndarray,
    *,
    geometry: str,
    period: float | None = None,
) -> np.ndarray:
    coords = _as_numpy_1d(coordinates)
    diff = np.abs(coords[:, None] - coords[None, :])
    if geometry == "circle":
        resolved_period = float(period) if period is not None else float(np.max(coords) - np.min(coords))
        if resolved_period <= 0:
            raise ValueError(f"circle period must be positive, got {resolved_period}")
        return np.minimum(diff, resolved_period - diff)
    if geometry in {"interval", "positive_scalar_log"}:
        return diff
    raise ValueError(f"Unsupported target geometry: {geometry!r}")


def euclidean_distance_matrix(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"points must be 2D, got {arr.shape}")
    diff = arr[:, None, :] - arr[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def maybe_pca(
    train_features: np.ndarray,
    *,
    pca_dim: int,
    val_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    train, train_sanitize_payload = _sanitize_feature_array(np.asarray(train_features, dtype=np.float64))
    val = None
    val_sanitize_payload = None
    if val_features is not None:
        val, val_sanitize_payload = _sanitize_feature_array(np.asarray(val_features, dtype=np.float64))
    if pca_dim <= 0 or train.shape[1] <= pca_dim or train.shape[0] <= 2:
        return train, val, {
            "applied": False,
            "input_dim": int(train.shape[1]),
            "output_dim": int(train.shape[1]),
            "explained_variance_ratio_sum": None,
            "train_sanitize": train_sanitize_payload,
            "val_sanitize": val_sanitize_payload,
        }
    n_components = min(int(pca_dim), int(train.shape[0] - 1), int(train.shape[1]))
    pca = PCA(n_components=n_components, svd_solver="full")
    try:
        train_pca = pca.fit_transform(train)
        val_pca = None if val is None else pca.transform(val)
    except np.linalg.LinAlgError as exc:
        return train, val, {
            "applied": False,
            "input_dim": int(train.shape[1]),
            "output_dim": int(train.shape[1]),
            "explained_variance_ratio_sum": None,
            "pca_error": str(exc),
            "train_sanitize": train_sanitize_payload,
            "val_sanitize": val_sanitize_payload,
        }
    return train_pca, val_pca, {
        "applied": True,
        "input_dim": int(train.shape[1]),
        "output_dim": int(n_components),
        "explained_variance_ratio_sum": _json_float(np.sum(pca.explained_variance_ratio_)),
        "train_sanitize": train_sanitize_payload,
        "val_sanitize": val_sanitize_payload,
    }


def compute_centroids(
    features: np.ndarray,
    grid_index: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feats = np.asarray(features, dtype=np.float64)
    groups = np.asarray(grid_index, dtype=np.int64)
    coords = np.asarray(coordinates, dtype=np.float64)
    if feats.shape[0] != groups.shape[0] or feats.shape[0] != coords.shape[0]:
        raise ValueError("features, grid_index, and coordinates must have the same length")
    unique_groups = np.unique(groups)
    centroids = []
    centroid_coords = []
    counts = []
    within_sse = []
    for group in unique_groups:
        mask = groups == group
        group_features = feats[mask]
        centroid = np.mean(group_features, axis=0)
        centroids.append(centroid)
        centroid_coords.append(float(np.mean(coords[mask])))
        counts.append(int(mask.sum()))
        within_sse.append(float(np.mean(np.sum((group_features - centroid) ** 2, axis=1))))
    return (
        np.vstack(centroids),
        np.asarray(centroid_coords, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
        np.asarray(within_sse, dtype=np.float64),
    )


def knn_geodesic_distance_matrix(
    linear_distances: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    dist = np.asarray(linear_distances, dtype=np.float64)
    n = int(dist.shape[0])
    if dist.shape != (n, n):
        raise ValueError(f"linear_distances must be square, got {dist.shape}")
    if n < 2:
        return dist.copy(), {"k": int(k), "connected": True, "effective_k": 0}
    effective_k = min(max(int(k), 0), n - 1)
    if effective_k == 0:
        graph = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(graph, 0.0)
    else:
        graph = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(graph, 0.0)
        for row in range(n):
            order = np.argsort(dist[row])
            neighbors = [idx for idx in order if idx != row][:effective_k]
            graph[row, neighbors] = dist[row, neighbors]
        graph = np.minimum(graph, graph.T)
    sparse = csr_matrix(np.where(np.isfinite(graph), graph, 0.0))
    geodesic = shortest_path(sparse, directed=False, unweighted=False)
    finite_off_diag = _upper_triangular_values(geodesic)
    connected = bool(np.all(np.isfinite(finite_off_diag)))
    return geodesic, {
        "k": int(k),
        "effective_k": int(effective_k),
        "connected": connected,
        "finite_pair_fraction": float(np.mean(np.isfinite(finite_off_diag))) if finite_off_diag.size else 1.0,
    }


def scaled_stress(target_distances: np.ndarray, representation_distances: np.ndarray) -> float | None:
    target = _upper_triangular_values(target_distances)
    rep = _upper_triangular_values(representation_distances)
    mask = np.isfinite(target) & np.isfinite(rep)
    if int(mask.sum()) < 1:
        return None
    target = target[mask]
    rep = rep[mask]
    denom = float(np.dot(rep, rep))
    target_norm = float(np.dot(target, target))
    if denom <= 0 or target_norm <= 0:
        return None
    scale = float(np.dot(target, rep) / denom)
    return _json_float(math.sqrt(float(np.sum((target - scale * rep) ** 2)) / target_norm))


def _knn_sets(distances: np.ndarray, k: int) -> list[set[int]]:
    n = int(distances.shape[0])
    effective_k = min(max(int(k), 0), n - 1)
    sets = []
    for row in range(n):
        order = np.argsort(distances[row])
        neighbors = [int(idx) for idx in order if int(idx) != row][:effective_k]
        sets.append(set(neighbors))
    return sets


def knn_recall(target_distances: np.ndarray, representation_distances: np.ndarray, *, k: int) -> float | None:
    n = int(target_distances.shape[0])
    effective_k = min(max(int(k), 0), n - 1)
    if effective_k == 0:
        return None
    target_sets = _knn_sets(target_distances, effective_k)
    rep_sets = _knn_sets(representation_distances, effective_k)
    recalls = [
        len(target_sets[index] & rep_sets[index]) / float(effective_k)
        for index in range(n)
    ]
    return _json_float(float(np.mean(recalls)))


def trustworthiness_score(
    target_distances: np.ndarray,
    representation_distances: np.ndarray,
    *,
    k: int,
) -> float | None:
    n = int(target_distances.shape[0])
    effective_k = min(max(int(k), 0), n - 1)
    if effective_k <= 0 or n <= effective_k + 1:
        return None
    target_rank = np.argsort(np.argsort(target_distances, axis=1), axis=1)
    rep_sets = _knn_sets(representation_distances, effective_k)
    target_sets = _knn_sets(target_distances, effective_k)
    penalty = 0.0
    for i in range(n):
        intrusions = rep_sets[i] - target_sets[i]
        for j in intrusions:
            penalty += max(0.0, float(target_rank[i, j] - effective_k))
    denom = n * effective_k * (2 * n - 3 * effective_k - 1)
    if denom <= 0:
        return None
    return _json_float(1.0 - (2.0 / denom) * penalty)


def continuity_score(
    target_distances: np.ndarray,
    representation_distances: np.ndarray,
    *,
    k: int,
) -> float | None:
    return trustworthiness_score(
        representation_distances,
        target_distances,
        k=k,
    )


def topology_metrics(
    *,
    geometry: str,
    target_distances: np.ndarray,
    linear_distances: np.ndarray,
    geodesic_distances: np.ndarray | None,
) -> dict[str, Any]:
    n = int(linear_distances.shape[0])
    adjacent = [float(linear_distances[i, i + 1]) for i in range(n - 1)]
    if geometry == "circle" and n > 2:
        adjacent.append(float(linear_distances[-1, 0]))
    adjacent_arr = np.asarray(adjacent, dtype=np.float64)
    adjacent_median = float(np.median(adjacent_arr[np.isfinite(adjacent_arr)])) if adjacent_arr.size else math.nan
    if geometry == "circle":
        first_last = float(linear_distances[0, -1]) if n >= 2 else math.nan
        closure_ratio = first_last / adjacent_median if adjacent_median > 0 else math.nan
        geodesic = geodesic_distances if geodesic_distances is not None else linear_distances
        return {
            "cycle_closure_ratio": _json_float(closure_ratio),
            "cycle_closure_error": _json_float(abs(closure_ratio - 1.0) if math.isfinite(closure_ratio) else None),
            "cycle_neighbor_wrap_score": _json_float(_wrap_neighbor_score(linear_distances)),
            "circular_order_score": _safe_corr(
                _upper_triangular_values(target_distances),
                _upper_triangular_values(geodesic),
                kind="spearman",
            ),
        }
    endpoint = float(linear_distances[0, -1]) if n >= 2 else math.nan
    endpoint_ratio = endpoint / adjacent_median if adjacent_median > 0 else math.nan
    return {
        "endpoint_separation": _json_float(endpoint_ratio),
        "monotone_order_score": _json_float(_monotone_order_score(linear_distances)),
        "foldover_rate": _json_float(_foldover_rate(linear_distances)),
    }


def _wrap_neighbor_score(linear_distances: np.ndarray) -> float | None:
    n = int(linear_distances.shape[0])
    if n < 3:
        return None
    first_neighbors = [int(idx) for idx in np.argsort(linear_distances[0]) if int(idx) != 0][:2]
    last_neighbors = [int(idx) for idx in np.argsort(linear_distances[-1]) if int(idx) != n - 1][:2]
    return 0.5 * (float((n - 1) in first_neighbors) + float(0 in last_neighbors))


def _monotone_order_score(
    linear_distances: np.ndarray,
    *,
    max_triplets: int = 200_000,
) -> float | None:
    n = int(linear_distances.shape[0])
    total_possible = n * (n - 1) * (n - 2) // 6
    if total_possible == 0:
        return None
    if total_possible > int(max_triplets):
        rng = np.random.default_rng(0)
        sampled: list[np.ndarray] = []
        sampled_count = 0
        while sampled_count < int(max_triplets):
            candidates = np.sort(
                rng.integers(0, n, size=(int(max_triplets) - sampled_count, 3)),
                axis=1,
            )
            mask = (
                (candidates[:, 0] < candidates[:, 1])
                & (candidates[:, 1] < candidates[:, 2])
            )
            valid = candidates[mask]
            if valid.size == 0:
                continue
            needed = int(max_triplets) - sampled_count
            valid = valid[:needed]
            sampled.append(valid)
            sampled_count += int(valid.shape[0])
        triplets = np.vstack(sampled)
        i = triplets[:, 0]
        j = triplets[:, 1]
        k = triplets[:, 2]
        good = (
            (linear_distances[i, k] >= linear_distances[i, j])
            & (linear_distances[i, k] >= linear_distances[j, k])
        )
        return float(np.mean(good))

    total = 0
    good_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                total += 1
                if linear_distances[i, k] >= linear_distances[i, j] and linear_distances[i, k] >= linear_distances[j, k]:
                    good_count += 1
    if total == 0:
        return None
    return good_count / float(total)


def _foldover_rate(linear_distances: np.ndarray) -> float | None:
    n = int(linear_distances.shape[0])
    if n < 4:
        return None
    adjacent = np.asarray([linear_distances[i, i + 1] for i in range(n - 1)], dtype=np.float64)
    threshold = float(np.median(adjacent))
    total = 0
    folded = 0
    for i in range(n):
        for j in range(i + 2, n):
            total += 1
            if linear_distances[i, j] < threshold:
                folded += 1
    if total == 0:
        return None
    return folded / float(total)


def fiber_metrics(
    *,
    features: np.ndarray,
    grid_index: np.ndarray,
    centroids: np.ndarray,
    counts: np.ndarray,
) -> dict[str, Any]:
    if int(np.min(counts)) < 2:
        return {
            "fiber_applicable": False,
            "mean_fiber_ratio": None,
            "median_fiber_ratio": None,
            "max_fiber_ratio": None,
            "between_to_within_snr": None,
            "usable_grid_points": int(len(counts)),
            "min_grid_point_count": int(np.min(counts)),
            "median_grid_point_count": float(np.median(counts)),
        }
    within = []
    for group_position, group in enumerate(np.unique(grid_index)):
        group_features = features[grid_index == group]
        centroid = centroids[group_position]
        within.append(float(np.mean(np.sum((group_features - centroid) ** 2, axis=1))))
    within_arr = np.asarray(within, dtype=np.float64)
    global_centroid = np.mean(centroids, axis=0)
    between_arr = np.sum((centroids - global_centroid) ** 2, axis=1)
    ratios = within_arr / np.maximum(between_arr, 1e-12)
    mean_within = float(np.mean(within_arr))
    mean_between = float(np.mean(between_arr))
    return {
        "fiber_applicable": True,
        "mean_fiber_ratio": _json_float(float(np.mean(ratios))),
        "median_fiber_ratio": _json_float(float(np.median(ratios))),
        "max_fiber_ratio": _json_float(float(np.max(ratios))),
        "between_to_within_snr": _json_float(mean_between / max(mean_within, 1e-12)),
        "usable_grid_points": int(len(counts)),
        "min_grid_point_count": int(np.min(counts)),
        "median_grid_point_count": float(np.median(counts)),
    }


def _project_to_polyline(
    samples: np.ndarray,
    centroids: np.ndarray,
    centroid_coords: np.ndarray,
    *,
    geometry: str,
    period: float | None,
) -> np.ndarray:
    if samples.size == 0:
        return np.empty((0,), dtype=np.float64)
    points = np.asarray(centroids, dtype=np.float64)
    coords = np.asarray(centroid_coords, dtype=np.float64)
    segment_pairs: list[tuple[int, int, float, float]] = []
    for idx in range(len(points) - 1):
        segment_pairs.append((idx, idx + 1, float(coords[idx]), float(coords[idx + 1])))
    resolved_period = float(period) if period is not None else None
    if geometry == "circle" and len(points) > 2:
        if resolved_period is None:
            resolved_period = float(np.max(coords) - np.min(coords) + (coords[1] - coords[0]))
        segment_pairs.append((len(points) - 1, 0, float(coords[-1]), float(coords[0] + resolved_period)))
    predictions = []
    for sample in samples:
        best_dist = math.inf
        best_coord = float(coords[0])
        for start, end, coord_start, coord_end in segment_pairs:
            a = points[start]
            b = points[end]
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom <= 0:
                t = 0.0
            else:
                t = min(1.0, max(0.0, float(np.dot(sample - a, ab) / denom)))
            projection = a + t * ab
            dist = float(np.sum((sample - projection) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_coord = coord_start + t * (coord_end - coord_start)
        if geometry == "circle" and resolved_period is not None:
            best_coord = best_coord % resolved_period
        predictions.append(best_coord)
    return np.asarray(predictions, dtype=np.float64)


def projection_metrics(
    *,
    val_features: np.ndarray,
    val_coordinates: np.ndarray,
    centroids: np.ndarray,
    centroid_coords: np.ndarray,
    geometry: str,
    period: float | None,
) -> dict[str, Any]:
    if val_features is None or val_coordinates is None:
        return {
            "projection_mae": None,
            "projection_rmse": None,
            "projection_r2": None,
            "projection_pearson": None,
            "projection_circular_mae": None,
        }
    pred = _project_to_polyline(
        val_features,
        centroids,
        centroid_coords,
        geometry=geometry,
        period=period,
    )
    truth = np.asarray(val_coordinates, dtype=np.float64)
    if geometry == "circle":
        resolved_period = float(period) if period is not None else float(np.max(centroid_coords) - np.min(centroid_coords))
        abs_err = np.abs(pred - truth)
        abs_err = np.minimum(abs_err, resolved_period - abs_err)
        return {
            "projection_mae": _json_float(float(np.mean(abs_err))),
            "projection_rmse": _json_float(float(np.sqrt(np.mean(abs_err ** 2)))),
            "projection_r2": None,
            "projection_pearson": None,
            "projection_circular_mae": _json_float(float(np.mean(abs_err))),
        }
    err = pred - truth
    ss_res = float(np.sum(err ** 2))
    centered = truth - float(np.mean(truth))
    ss_tot = float(np.sum(centered ** 2))
    return {
        "projection_mae": _json_float(float(np.mean(np.abs(err)))),
        "projection_rmse": _json_float(float(np.sqrt(np.mean(err ** 2)))),
        "projection_r2": _json_float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "projection_pearson": _safe_corr(pred, truth, kind="pearson"),
        "projection_circular_mae": None,
    }


def compute_manifold_layer_evaluation(
    *,
    train_features: torch.Tensor | np.ndarray,
    train_grid_index: torch.Tensor | np.ndarray,
    train_coordinates: torch.Tensor | np.ndarray,
    geometry: str,
    period: float | None = None,
    val_features: torch.Tensor | np.ndarray | None = None,
    val_coordinates: torch.Tensor | np.ndarray | None = None,
    pca_dim: int = 64,
    geodesic_neighbors: tuple[int, ...] = (4, 6, 8),
    plot_max_points: int = 256,
) -> ManifoldLayerEvaluation:
    train = _as_numpy_2d(train_features)
    val = None if val_features is None else _as_numpy_2d(val_features)
    train_pca, val_pca, pca_payload = maybe_pca(train, pca_dim=int(pca_dim), val_features=val)
    train_groups = _as_numpy_1d(train_grid_index).astype(np.int64)
    train_coords = _as_numpy_1d(train_coordinates)
    centroids, centroid_coords, counts, _ = compute_centroids(
        train_pca,
        train_groups,
        train_coords,
    )
    latent_dist = target_distance_matrix(
        centroid_coords,
        geometry=geometry,
        period=period,
    )
    linear_dist = euclidean_distance_matrix(centroids)
    latent_values = _upper_triangular_values(latent_dist)
    linear_values = _upper_triangular_values(linear_dist)

    graph_payloads: dict[str, Any] = {}
    geodesic_by_k: dict[int, np.ndarray] = {}
    best_geodesic = None
    best_k = None
    best_spearman = None
    for k in geodesic_neighbors:
        geodesic, graph_payload = knn_geodesic_distance_matrix(linear_dist, k=int(k))
        geodesic_by_k[int(k)] = geodesic
        geo_values = _upper_triangular_values(geodesic)
        spearman = _safe_corr(latent_values, geo_values, kind="spearman")
        graph_payload["spearman_latent_vs_geodesic"] = spearman
        graph_payloads[str(int(k))] = graph_payload
        if graph_payload["connected"] and spearman is not None:
            if best_spearman is None or float(spearman) > float(best_spearman):
                best_spearman = float(spearman)
                best_geodesic = geodesic
                best_k = int(k)
    if best_geodesic is None and geodesic_by_k:
        best_k = int(next(iter(geodesic_by_k)))
        best_geodesic = geodesic_by_k[best_k]

    geodesic_values = (
        _upper_triangular_values(best_geodesic)
        if best_geodesic is not None
        else np.full_like(latent_values, np.nan)
    )
    spearman_linear = _safe_corr(latent_values, linear_values, kind="spearman")
    spearman_geodesic = _safe_corr(latent_values, geodesic_values, kind="spearman")
    metrics: dict[str, Any] = {
        "usable_grid_points": int(len(centroid_coords)),
        "min_grid_point_count": int(np.min(counts)),
        "median_grid_point_count": float(np.median(counts)),
        "pca": pca_payload,
        "spearman_latent_vs_linear": spearman_linear,
        "pearson_latent_vs_linear": _safe_corr(latent_values, linear_values, kind="pearson"),
        "spearman_latent_vs_geodesic": spearman_geodesic,
        "pearson_latent_vs_geodesic": _safe_corr(latent_values, geodesic_values, kind="pearson"),
        "geodesic_gain": (
            _json_float(float(spearman_geodesic) - float(spearman_linear))
            if spearman_geodesic is not None and spearman_linear is not None
            else None
        ),
        "stress_scaled": scaled_stress(latent_dist, linear_dist),
        "selected_geodesic_k": best_k,
        "graph_by_k": graph_payloads,
        "knn_recall_at_1": knn_recall(latent_dist, linear_dist, k=1),
        "knn_recall_at_3": knn_recall(latent_dist, linear_dist, k=3),
        "knn_recall_at_5": knn_recall(latent_dist, linear_dist, k=5),
        "trustworthiness": trustworthiness_score(latent_dist, linear_dist, k=5),
        "continuity": continuity_score(latent_dist, linear_dist, k=5),
    }
    metrics.update(
        topology_metrics(
            geometry=geometry,
            target_distances=latent_dist,
            linear_distances=linear_dist,
            geodesic_distances=best_geodesic,
        )
    )
    metrics.update(
        fiber_metrics(
            features=train_pca,
            grid_index=train_groups,
            centroids=centroids,
            counts=counts,
        )
    )
    if val_pca is not None and val_coordinates is not None:
        val_coords = _as_numpy_1d(val_coordinates)
    else:
        val_coords = None
    metrics.update(
        projection_metrics(
            val_features=val_pca,
            val_coordinates=val_coords,
            centroids=centroids,
            centroid_coords=centroid_coords,
            geometry=geometry,
            period=period,
        )
    )

    distance_indices = _plot_indices(len(centroid_coords), int(plot_max_points))
    plot_latent_dist = latent_dist[np.ix_(distance_indices, distance_indices)]
    plot_linear_dist = linear_dist[np.ix_(distance_indices, distance_indices)]
    plot_geodesic = (
        None
        if best_geodesic is None
        else np.asarray(best_geodesic)[np.ix_(distance_indices, distance_indices)]
    )
    plot_data = {
        "downsampled": bool(len(distance_indices) != len(centroid_coords)),
        "source_grid_points": int(len(centroid_coords)),
        "centroid_grid_points": int(len(centroid_coords)),
        "distance_grid_points": int(len(distance_indices)),
        "distance_downsampled": bool(len(distance_indices) != len(centroid_coords)),
        "distance_plot_indices": [int(value) for value in distance_indices.tolist()],
        "plot_grid_points": int(len(distance_indices)),
        "plot_indices": [int(value) for value in distance_indices.tolist()],
        "centroid_coordinates": [float(value) for value in centroid_coords.tolist()],
        "centroid_counts": [int(value) for value in counts.tolist()],
        "centroids": centroids.astype(float).tolist(),
        "latent_distance": plot_latent_dist.astype(float).tolist(),
        "linear_distance": plot_linear_dist.astype(float).tolist(),
        "geodesic_distance": (
            None if plot_geodesic is None else np.asarray(plot_geodesic, dtype=float).tolist()
        ),
        "selected_geodesic_k": best_k,
        "geometry": geometry,
        "period": period,
    }
    return ManifoldLayerEvaluation(metrics=metrics, plot_data=plot_data)


def summarize_layer_metrics(
    *,
    by_layer: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if not by_layer:
        return {}

    def best_layer(metric_name: str, *, direction: str) -> dict[str, Any]:
        scored = []
        for layer, payload in by_layer.items():
            value = payload.get(metric_name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                scored.append((int(layer), float(value)))
        if not scored:
            return {"layer": None, "value": None}
        if direction == "min":
            layer, value = min(scored, key=lambda item: item[1])
        else:
            layer, value = max(scored, key=lambda item: item[1])
        return {"layer": int(layer), "value": float(value)}

    return {
        "best_isometry_layer": best_layer("spearman_latent_vs_geodesic", direction="max"),
        "best_neighborhood_layer": best_layer("knn_recall_at_5", direction="max"),
        "best_projection_layer": best_layer("projection_r2", direction="max"),
        "best_fiber_layer": best_layer("mean_fiber_ratio", direction="min"),
    }
