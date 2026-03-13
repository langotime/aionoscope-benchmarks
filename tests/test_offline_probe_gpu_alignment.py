from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import aionoscope_benchmarks.offline_probe as offline_probe_module
from aionoscope_benchmarks.offline_probe import (
    CollectedProbeFeatures,
    OfflineProbeConfig,
    offline_probe_run_linear_multihead_by_layer_multi_val_from_collected,
)
from aionoscope_benchmarks.probe_metrics import probe_compute_metrics


CUDA_AVAILABLE = torch.cuda.is_available()


def _build_probe_config() -> OfflineProbeConfig:
    return OfflineProbeConfig(
        steps=2,
        batch_size=4,
        learning_rate=0.05,
        final_learning_rate=0.05,
        learning_rate_warmup_steps=0,
        weight_decay=0.0,
        opt_betas=(0.9, 0.999),
        gradient_clip=1.0,
        checkpoint_interval=1,
    )


def _make_collected(
    *,
    feature_shift: float,
) -> CollectedProbeFeatures:
    class_targets = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    dense_targets = torch.tensor(
        [
            [0.0, 0.5],
            [1.0, 1.5],
            [0.5, 1.0],
            [-1.0, 0.0],
            [0.25, 0.75],
            [1.25, 1.75],
            [0.75, 1.25],
            [-0.75, -0.25],
        ],
        dtype=torch.float32,
    )
    bias = torch.ones((class_targets.size(0), 1), dtype=torch.float32)
    base_features = torch.cat(
        [
            (class_targets * 2.0) - 1.0,
            dense_targets,
            bias,
        ],
        dim=1,
    )
    layer0 = base_features + feature_shift
    layer1 = (base_features * 0.5) + feature_shift
    return CollectedProbeFeatures(
        features_by_layer={0: layer0, 1: layer1},
        class_targets=class_targets,
        dense_targets=dense_targets,
        has_crops=False,
        timings={},
    )


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
        ),
    ],
)
def test_probe_compute_metrics_matches_sklearn(device: torch.device) -> None:
    class_names = ["a", "b", "c"]
    targets_np = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=np.int64,
    )
    predictions_np = np.array(
        [
            [0.92, 0.08, 0.81],
            [0.15, 0.84, 0.23],
            [0.73, 0.79, 0.31],
            [0.18, 0.27, 0.77],
            [0.69, 0.24, 0.28],
            [0.21, 0.76, 0.72],
        ],
        dtype=np.float32,
    )
    targets = torch.tensor(targets_np, dtype=torch.float32, device=device)
    predictions = torch.tensor(predictions_np, dtype=torch.float32, device=device)

    macro_auc, per_class_auc, macro_auprc, per_class_auprc = probe_compute_metrics(
        targets=targets,
        predictions=predictions,
        class_names=class_names,
    )

    expected_auc = {
        class_name: float(roc_auc_score(targets_np[:, index], predictions_np[:, index]))
        for index, class_name in enumerate(class_names)
    }
    expected_auprc = {
        class_name: float(average_precision_score(targets_np[:, index], predictions_np[:, index]))
        for index, class_name in enumerate(class_names)
    }

    for class_name in class_names:
        assert per_class_auc[class_name] == pytest.approx(expected_auc[class_name], abs=1e-6)
        assert per_class_auprc[class_name] == pytest.approx(expected_auprc[class_name], abs=1e-6)
    assert macro_auc == pytest.approx(sum(expected_auc.values()) / len(expected_auc), abs=1e-6)
    assert macro_auprc == pytest.approx(
        sum(expected_auprc.values()) / len(expected_auprc),
        abs=1e-6,
    )


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
        ),
    ],
)
def test_multi_val_runner_stages_each_layer_once(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
) -> None:
    train_collected = _make_collected(feature_shift=0.0)
    val_collected_by_seed = {
        11: _make_collected(feature_shift=0.1),
        13: _make_collected(feature_shift=-0.1),
    }
    stage_calls: list[int] = []
    original_stage_layer = offline_probe_module._stage_probe_layer_multi_val

    def wrapped_stage_layer(*args, **kwargs):
        stage_calls.append(int(kwargs["layer"]))
        return original_stage_layer(*args, **kwargs)

    monkeypatch.setattr(offline_probe_module, "_stage_probe_layer_multi_val", wrapped_stage_layer)

    results = offline_probe_run_linear_multihead_by_layer_multi_val_from_collected(
        train_collected=train_collected,
        val_collected_by_seed=val_collected_by_seed,
        num_classes=2,
        class_names=["alpha", "beta"],
        eval_config=_build_probe_config(),
        device=device,
        layers_categorical=(0, 1),
        layers_dense=(0, 1),
        layers_confusion=tuple(),
        dense_target_names=["delta", "theta"],
        dense_log_per_target=True,
        probe_seed=0,
    )

    assert stage_calls == [0, 1]
    assert set(results) == {11, 13}
    for seed_value, seed_results in results.items():
        assert set(seed_results["categorical"]) == {0, 1}
        assert set(seed_results["dense"]) == {0, 1}
        for layer in (0, 1):
            classification_timings = seed_results["categorical"][layer]["timings"]["classification"]
            regression_timings = seed_results["dense"][layer]["timings"]["regression"]
            assert classification_timings["layer_staging_s"] >= 0.0
            assert regression_timings["layer_staging_s"] >= 0.0
            assert classification_timings["eval_numpy_s"] == 0.0
            assert classification_timings["eval_torchmetrics_s"] >= 0.0
            assert seed_results["categorical"][layer]["best_auc"]["best_probe_step"] in {1, 2}
            assert seed_results["dense"][layer]["per_target_best_step"]["delta"] in {1, 2}
