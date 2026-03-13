from __future__ import annotations

from collections import defaultdict
import sys

import torch

try:
    from torchmetrics.functional.classification import (
        multilabel_auroc,
        multilabel_average_precision,
    )
except ImportError as exc:  # pragma: no cover - exercised when env is unsynced
    multilabel_auroc = None
    multilabel_average_precision = None
    _TORCHMETRICS_IMPORT_ERROR = exc
else:
    _TORCHMETRICS_IMPORT_ERROR = None


def ensure_probe_metric_dependencies_available() -> None:
    if _TORCHMETRICS_IMPORT_ERROR is None:
        return
    raise ImportError(
        "Offline probe categorical metrics require torchmetrics in the active benchmark environment. "
        f"Interpreter: {sys.executable}. "
        "Install benchmark dependencies into this model-specific venv before running the benchmark."
    ) from _TORCHMETRICS_IMPORT_ERROR


def probe_compute_metrics(
    *, targets: torch.Tensor, predictions: torch.Tensor, class_names: list[str]
) -> tuple[float, dict[str, float], float, dict[str, float]]:
    ensure_probe_metric_dependencies_available()
    if not isinstance(targets, torch.Tensor):
        target_device = predictions.device if isinstance(predictions, torch.Tensor) else None
        targets = torch.as_tensor(targets, device=target_device)
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.as_tensor(predictions, device=targets.device)
    if targets.shape != predictions.shape:
        raise ValueError(
            "Probe targets/predictions shape mismatch: "
            f"targets={tuple(targets.shape)}, predictions={tuple(predictions.shape)}"
        )
    if targets.dim() != 2:
        raise ValueError(f"Probe targets must be 2D [N, C], got: {tuple(targets.shape)}")
    if targets.device != predictions.device:
        raise ValueError(
            "Probe targets/predictions device mismatch: "
            f"targets={targets.device}, predictions={predictions.device}"
        )
    if not class_names:
        raise ValueError("class_names is required to compute probe metrics")

    num_classes = int(targets.size(1))
    if len(class_names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, got {len(class_names)}")

    targets_f = targets.to(dtype=torch.float32)
    unique_targets = torch.unique(targets_f)
    invalid_targets = torch.any((unique_targets != 0.0) & (unique_targets != 1.0))
    if bool(invalid_targets):
        raise ValueError(
            "AUROC/AUPRC require binary targets in {0,1}. "
            f"Got unique values: {unique_targets.tolist()!r}"
        )

    invalid_classes = []
    sample_count = int(targets_f.size(0))
    for class_index, class_name in enumerate(class_names):
        positive_count = int(targets_f[:, class_index].sum().item())
        negative_count = sample_count - positive_count
        if positive_count == 0 or negative_count == 0:
            invalid_classes.append(f"{class_name} (pos={positive_count}, neg={negative_count})")
    if invalid_classes:
        raise ValueError(
            "AUC is undefined for classes with only one label in the validation set: "
            + ", ".join(invalid_classes)
        )

    predictions_f = predictions.to(dtype=torch.float32)
    if torch.any(predictions_f < 0.0) or torch.any(predictions_f > 1.0):
        raise ValueError("AUROC/AUPRC require predicted probabilities in [0, 1].")

    targets_i = targets_f.to(dtype=torch.int64)
    per_class_auc_t = multilabel_auroc(
        predictions_f,
        targets_i,
        num_labels=num_classes,
        average=None,
        thresholds=None,
        validate_args=False,
    )
    per_class_auprc_t = multilabel_average_precision(
        predictions_f,
        targets_i,
        num_labels=num_classes,
        average=None,
        thresholds=None,
        validate_args=False,
    )
    per_class_auc = {
        class_name: float(value)
        for class_name, value in zip(class_names, per_class_auc_t.tolist(), strict=True)
    }
    per_class_auprc = {
        class_name: float(value)
        for class_name, value in zip(class_names, per_class_auprc_t.tolist(), strict=True)
    }
    macro_auc = float(per_class_auc_t.mean().item())
    macro_auprc = float(per_class_auprc_t.mean().item())
    return macro_auc, per_class_auc, macro_auprc, per_class_auprc


def probe_compute_pairwise_confusion_torch(
    *,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    class_names: list[str],
) -> dict[str, object]:
    if targets.shape != predictions.shape:
        raise ValueError(
            "Probe targets/predictions shape mismatch: "
            f"targets={tuple(targets.shape)}, predictions={tuple(predictions.shape)}"
        )
    if targets.dim() != 2:
        raise ValueError(f"Probe targets must be 2D [N, C], got: {tuple(targets.shape)}")
    if targets.device != predictions.device:
        raise ValueError(
            "Probe targets/predictions device mismatch: "
            f"targets={targets.device}, predictions={predictions.device}"
        )
    if not class_names:
        raise ValueError("class_names is required to compute pairwise confusion")

    num_classes = int(targets.size(1))
    if len(class_names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, got {len(class_names)}")

    targets_f = targets.to(dtype=torch.float32)
    unique_targets = torch.unique(targets_f)
    invalid_targets = torch.any((unique_targets != 0.0) & (unique_targets != 1.0))
    if bool(invalid_targets):
        raise ValueError(
            "Pairwise confusion requires binary targets in {0,1}. "
            f"Got unique values: {unique_targets.tolist()!r}"
        )

    predictions_f = predictions.to(dtype=torch.float32)
    if torch.any(predictions_f < 0.0) or torch.any(predictions_f > 1.0):
        raise ValueError("Pairwise confusion requires predicted probabilities in [0, 1].")

    present = targets_f
    absent = 1.0 - targets_f
    counts_offdiag = torch.matmul(present.transpose(0, 1), absent)
    numerator_offdiag = torch.matmul(present.transpose(0, 1), absent * predictions_f)
    mean_pred = numerator_offdiag / counts_offdiag
    counts = counts_offdiag.to(torch.int64)

    diag_counts = torch.sum(present, dim=0).to(torch.int64)
    diag_numerator = torch.sum(present * predictions_f, dim=0)
    diag_mean = diag_numerator / diag_counts
    diag_idx = torch.arange(num_classes, device=targets.device)
    mean_pred[diag_idx, diag_idx] = diag_mean
    counts[diag_idx, diag_idx] = diag_counts

    return {
        "class_names": list(class_names),
        "mean_pred": mean_pred.cpu().tolist(),
        "count": counts.cpu().tolist(),
    }


def probe_build_group_to_classes(
    *, class_to_groups: dict[str, list[str]]
) -> dict[str, list[str]]:
    group_to_classes: dict[str, list[str]] = defaultdict(list)
    for class_name, groups in class_to_groups.items():
        for group_name in groups:
            group_to_classes[group_name].append(class_name)
    return {group: sorted(classes) for group, classes in group_to_classes.items()}
