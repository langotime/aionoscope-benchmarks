from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .probe_metrics import probe_compute_metrics, probe_compute_pairwise_confusion_torch
from .schedules import cosine_schedule, update_learning_rate_


@dataclass(frozen=True)
class OfflineProbeConfig:
    steps: int
    batch_size: int
    learning_rate: float
    final_learning_rate: float
    learning_rate_warmup_steps: int
    weight_decay: float
    opt_betas: tuple[float, float]
    gradient_clip: float
    checkpoint_interval: int


def _validate_offline_probe_config(*, eval_config: OfflineProbeConfig) -> None:
    if eval_config.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {eval_config.batch_size}")
    if eval_config.steps <= 0:
        raise ValueError(f"steps must be > 0, got {eval_config.steps}")
    if eval_config.checkpoint_interval <= 0:
        raise ValueError(
            "checkpoint_interval must be > 0, "
            f"got {eval_config.checkpoint_interval}"
        )
    if eval_config.gradient_clip < 0:
        raise ValueError(f"gradient_clip must be >= 0, got {eval_config.gradient_clip}")


def _split_offline_probe_batch(
    batch: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not isinstance(batch, (tuple, list)):
        raise ValueError(f"Offline probe batches must be tuples, got {type(batch).__name__}")
    if len(batch) == 2:
        x, y_cls = batch
        y_dense = None
    elif len(batch) == 3:
        x, y_cls, y_dense = batch
    else:
        raise ValueError(f"Offline probe batches must have 2 or 3 items, got {len(batch)}")
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Offline probe inputs must be tensors, got {type(x).__name__}")
    if not isinstance(y_cls, torch.Tensor):
        raise ValueError(f"Offline probe class targets must be tensors, got {type(y_cls).__name__}")
    if y_dense is not None and not isinstance(y_dense, torch.Tensor):
        raise ValueError(f"Offline probe dense targets must be tensors, got {type(y_dense).__name__}")
    return x, y_cls, y_dense


def _collect_probe_features_by_layer(
    *,
    encoder: nn.Module,
    representation_fn: Callable[[torch.Tensor], dict[int, torch.Tensor]],
    layers: tuple[int, ...],
    loader: Iterable[tuple[torch.Tensor, ...]],
    device: torch.device,
    auto_mixed_precision,
    allow_crops: bool,
    timings: dict[str, float | int] | None = None,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor | None, bool]:
    if not layers:
        raise ValueError("layers must be non-empty")
    if len(set(layers)) != len(layers):
        raise ValueError(f"layers must be unique, got {layers}")

    total_start = perf_counter()
    feature_batches: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    class_target_batches: list[torch.Tensor] = []
    dense_target_batches: list[torch.Tensor] = []
    has_crops = False
    saw_dense_targets = None
    batches = 0
    samples = 0
    to_device_s = 0.0
    forward_s = 0.0
    cpu_copy_s = 0.0

    encoder_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            batches += 1
            x, y_cls, y_dense = _split_offline_probe_batch(batch)
            samples += int(x.size(0))
            if y_dense is None:
                if saw_dense_targets is True:
                    raise ValueError("Loader mixes dense and non-dense batches")
                saw_dense_targets = False
            else:
                if saw_dense_targets is False:
                    raise ValueError("Loader mixes dense and non-dense batches")
                saw_dense_targets = True

            if x.dim() == 4:
                if not allow_crops:
                    raise ValueError(f"Expected 3D inputs without crops, got {tuple(x.shape)}")
                has_crops = True
                batch_size, num_crops, num_channels, channel_size = x.size()
                x = x.reshape(batch_size * num_crops, num_channels, channel_size)
                to_device_start = perf_counter()
                x = x.to(device, non_blocking=True)
                to_device_s += perf_counter() - to_device_start
                forward_start = perf_counter()
                with auto_mixed_precision:
                    layer_reps = representation_fn(x)
                forward_s += perf_counter() - forward_start
                if not isinstance(layer_reps, dict):
                    raise ValueError(
                        "representation_fn must return dict[int, Tensor], "
                        f"got {type(layer_reps).__name__}"
                    )
                cpu_copy_start = perf_counter()
                for layer in layers:
                    rep = layer_reps.get(layer)
                    if rep is None:
                        raise ValueError(
                            f"representation_fn missing requested layer={layer}; "
                            f"returned keys={sorted(layer_reps.keys())}"
                        )
                    if rep.dim() != 2:
                        raise ValueError(
                            f"representation_fn output for layer={layer} must be [B, D], got {tuple(rep.shape)}"
                        )
                    rep = rep.float().reshape(batch_size, num_crops, -1).cpu()
                    feature_batches[layer].append(rep)
                class_target_batches.append(y_cls.float().cpu())
                if y_dense is not None:
                    dense_target_batches.append(y_dense.float().cpu())
                cpu_copy_s += perf_counter() - cpu_copy_start
            elif x.dim() == 3:
                if has_crops:
                    raise ValueError("Mixed cropped and non-cropped batches in loader")
                to_device_start = perf_counter()
                x = x.to(device, non_blocking=True)
                to_device_s += perf_counter() - to_device_start
                forward_start = perf_counter()
                with auto_mixed_precision:
                    layer_reps = representation_fn(x)
                forward_s += perf_counter() - forward_start
                if not isinstance(layer_reps, dict):
                    raise ValueError(
                        "representation_fn must return dict[int, Tensor], "
                        f"got {type(layer_reps).__name__}"
                    )
                cpu_copy_start = perf_counter()
                for layer in layers:
                    rep = layer_reps.get(layer)
                    if rep is None:
                        raise ValueError(
                            f"representation_fn missing requested layer={layer}; "
                            f"returned keys={sorted(layer_reps.keys())}"
                        )
                    if rep.dim() != 2:
                        raise ValueError(
                            f"representation_fn output for layer={layer} must be [B, D], got {tuple(rep.shape)}"
                        )
                    feature_batches[layer].append(rep.float().cpu())
                class_target_batches.append(y_cls.float().cpu())
                if y_dense is not None:
                    dense_target_batches.append(y_dense.float().cpu())
                cpu_copy_s += perf_counter() - cpu_copy_start
            else:
                raise ValueError(f"Expected inputs with 3 or 4 dims, got {tuple(x.shape)}")
    encoder.train(encoder_training)

    if not class_target_batches:
        raise ValueError("Feature extraction produced no batches")
    cat_start = perf_counter()
    features_by_layer = {}
    for layer, batches_for_layer in feature_batches.items():
        if not batches_for_layer:
            raise ValueError(f"No features produced for layer={layer}")
        features_by_layer[layer] = torch.cat(batches_for_layer, dim=0)
    class_targets = torch.cat(class_target_batches, dim=0)
    dense_targets = torch.cat(dense_target_batches, dim=0) if dense_target_batches else None
    cat_s = perf_counter() - cat_start
    total_s = perf_counter() - total_start
    if timings is not None:
        timings.update(
            {
                "total_s": float(total_s),
                "to_device_s": float(to_device_s),
                "forward_s": float(forward_s),
                "cpu_copy_s": float(cpu_copy_s),
                "cat_s": float(cat_s),
                "batches": int(batches),
                "samples": int(samples),
            }
        )
    return features_by_layer, class_targets, dense_targets, has_crops


def _evaluate_probe_features(
    *,
    probe: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
    class_names: list[str],
    compute_pairwise_confusion: bool = True,
    timings: dict[str, float | int] | None = None,
) -> tuple[
    tuple[float, dict[str, float], float, dict[str, float]],
    dict[str, object] | None,
]:
    total_start = perf_counter()
    probe.eval()
    logits_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []

    feature_loader = DataLoader(
        dataset=TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    forward_start = perf_counter()
    with torch.inference_mode():
        for batch_features, batch_targets in feature_loader:
            if batch_features.dim() == 3:
                size, num_crops, feature_dim = batch_features.size()
                flat_features = batch_features.reshape(size * num_crops, feature_dim).to(
                    device, non_blocking=True
                )
                logits = probe(flat_features).reshape(size, num_crops, -1).mean(dim=1)
            elif batch_features.dim() == 2:
                logits = probe(batch_features.to(device, non_blocking=True))
            else:
                raise ValueError(
                    f"Expected offline probe features to be 2D or 3D, got {tuple(batch_features.shape)}"
                )
            logits_batches.append(logits)
            target_batches.append(batch_targets)
    forward_s = perf_counter() - forward_start

    probe.train()
    numpy_start = perf_counter()
    logits_all = torch.cat(logits_batches, dim=0)
    predictions_t = logits_all.float().sigmoid()
    targets_t = torch.cat(target_batches, dim=0).float()
    predictions = predictions_t.cpu().numpy()
    targets_np = targets_t.cpu().numpy()
    numpy_s = perf_counter() - numpy_start
    metrics_start = perf_counter()
    metrics = probe_compute_metrics(
        targets=targets_np,
        predictions=predictions,
        class_names=class_names,
    )
    metrics_s = perf_counter() - metrics_start
    pairwise_confusion = None
    confusion_s = 0.0
    if compute_pairwise_confusion:
        confusion_start = perf_counter()
        pairwise_confusion = probe_compute_pairwise_confusion_torch(
            targets=targets_t,
            predictions=predictions_t,
            class_names=class_names,
        )
        confusion_s = perf_counter() - confusion_start
    if timings is not None:
        timings.update(
            {
                "total_s": float(perf_counter() - total_start),
                "forward_s": float(forward_s),
                "numpy_s": float(numpy_s),
                "metrics_s": float(metrics_s),
                "pairwise_confusion_s": float(confusion_s),
            }
        )
    return metrics, pairwise_confusion


def _evaluate_regression_features_streaming(
    *,
    probe: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
    target_names: list[str],
    timings: dict[str, float | int] | None = None,
) -> dict[str, torch.Tensor]:
    total_start = perf_counter()
    probe.eval()
    target_dim = int(targets.size(1))
    sum_sq_error = torch.zeros(target_dim, dtype=torch.float64, device=device)
    sum_abs_error = torch.zeros(target_dim, dtype=torch.float64, device=device)
    sum_targets = torch.zeros(target_dim, dtype=torch.float64, device=device)
    sum_target_sq = torch.zeros(target_dim, dtype=torch.float64, device=device)
    sum_predictions = torch.zeros(target_dim, dtype=torch.float64, device=device)
    sum_prediction_sq = torch.zeros(target_dim, dtype=torch.float64, device=device)
    sum_target_prediction = torch.zeros(target_dim, dtype=torch.float64, device=device)
    valid_count = torch.zeros(target_dim, dtype=torch.int64, device=device)

    feature_loader = DataLoader(
        dataset=TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    forward_start = perf_counter()
    with torch.inference_mode():
        for batch_features, batch_targets in feature_loader:
            if batch_features.dim() == 3:
                size, num_crops, feature_dim = batch_features.size()
                flat_features = batch_features.reshape(size * num_crops, feature_dim).to(
                    device, non_blocking=True
                )
                predictions = probe(flat_features).reshape(size, num_crops, -1).mean(dim=1)
            elif batch_features.dim() == 2:
                predictions = probe(batch_features.to(device, non_blocking=True))
            else:
                raise ValueError(
                    f"Expected offline probe features to be 2D or 3D, got {tuple(batch_features.shape)}"
                )
            predictions = predictions.float()
            batch_targets = batch_targets.to(device, non_blocking=True).float()
            valid = torch.isfinite(batch_targets)
            batch_targets = torch.nan_to_num(batch_targets, nan=0.0)
            predictions64 = predictions.to(torch.float64)
            targets64 = batch_targets.to(torch.float64)
            valid64 = valid.to(torch.float64)
            errors64 = (predictions64 - targets64) * valid64
            sum_sq_error += torch.sum(errors64 * errors64, dim=0)
            sum_abs_error += torch.sum(errors64.abs(), dim=0)
            sum_targets += torch.sum(targets64 * valid64, dim=0)
            sum_target_sq += torch.sum(targets64 * targets64 * valid64, dim=0)
            sum_predictions += torch.sum(predictions64 * valid64, dim=0)
            sum_prediction_sq += torch.sum(predictions64 * predictions64 * valid64, dim=0)
            sum_target_prediction += torch.sum(predictions64 * targets64 * valid64, dim=0)
            valid_count += valid.sum(dim=0).to(torch.int64)
    forward_s = perf_counter() - forward_start
    probe.train()

    if int(valid_count.sum().item()) < 1:
        raise ValueError("Validation split has no finite dense targets")
    if torch.any(valid_count == 0):
        missing = [
            name
            for name, count in zip(target_names, valid_count.tolist(), strict=True)
            if count == 0
        ]
        raise ValueError(
            "Validation split has no finite dense samples for targets: " + ", ".join(missing)
        )

    finalize_start = perf_counter()
    count = valid_count.to(torch.float64)
    mean_targets = sum_targets / count
    mean_predictions = sum_predictions / count
    ss_tot = sum_target_sq - count * (mean_targets * mean_targets)
    ss_pred = sum_prediction_sq - count * (mean_predictions * mean_predictions)
    denom = torch.sqrt(ss_tot * ss_pred)

    mse = sum_sq_error / count
    mae = sum_abs_error / count
    r2 = torch.full_like(mse, float("nan"))
    r2_defined = ss_tot > 0
    r2[r2_defined] = 1.0 - (sum_sq_error[r2_defined] / ss_tot[r2_defined])

    pearson = torch.full_like(mse, float("nan"))
    pearson_defined = denom > 0
    covariance = sum_target_prediction - count * mean_targets * mean_predictions
    pearson[pearson_defined] = covariance[pearson_defined] / denom[pearson_defined]
    finalize_s = perf_counter() - finalize_start
    if timings is not None:
        timings.update(
            {
                "total_s": float(perf_counter() - total_start),
                "forward_s": float(forward_s),
                "finalize_s": float(finalize_s),
            }
        )
    return {
        "mse": mse.cpu(),
        "mae": mae.cpu(),
        "r2": r2.cpu(),
        "pearson": pearson.cpu(),
    }


def _train_linear_classification_probe(
    *,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    num_classes: int,
    class_names: list[str],
    eval_config: OfflineProbeConfig,
    device: torch.device,
    val_has_crops: bool,
    compute_pairwise_confusion: bool = True,
) -> dict[str, object]:
    total_start = perf_counter()
    if train_features.dim() != 2:
        raise ValueError(f"Expected train features to be 2D, got {tuple(train_features.shape)}")
    if val_features.dim() not in (2, 3):
        raise ValueError(f"Expected val features to be 2D or 3D, got {tuple(val_features.shape)}")
    if train_targets.dim() != 2 or val_targets.dim() != 2:
        raise ValueError("Targets must be 2D")

    feature_dim = int(train_features.size(1))
    train_size = int(train_features.size(0))
    val_size = int(val_targets.size(0))
    if train_size < eval_config.batch_size:
        raise ValueError(
            "Offline probe train split too small for batch_size="
            f"{eval_config.batch_size}: {train_size} samples"
        )
    if val_size < 1:
        raise ValueError("Offline probe validation split is empty")

    probe = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, num_classes)).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=eval_config.learning_rate,
        betas=eval_config.opt_betas,
        weight_decay=eval_config.weight_decay,
    )
    lr_schedule = cosine_schedule(
        total_steps=eval_config.steps,
        start_value=eval_config.learning_rate,
        final_value=eval_config.final_learning_rate,
        warmup_steps=eval_config.learning_rate_warmup_steps,
        warmup_start_value=1e-6,
    )
    train_feature_loader = DataLoader(
        dataset=TensorDataset(train_features, train_targets),
        batch_size=eval_config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    def cycle(dataloader):
        while True:
            yield from dataloader

    train_iterator = cycle(train_feature_loader)
    best_auc_metrics = None
    best_auc_step = None
    best_auc_pairwise_confusion = None
    best_auprc_metrics = None
    best_auprc_step = None
    best_auprc_pairwise_confusion = None
    eval_calls = 0
    eval_total_s = 0.0
    eval_forward_s = 0.0
    eval_numpy_s = 0.0
    eval_metrics_s = 0.0
    eval_pairwise_confusion_s = 0.0

    for step in range(eval_config.steps):
        update_learning_rate_(optimizer, next(lr_schedule))
        batch_features, batch_targets = next(train_iterator)
        batch_features = batch_features.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        logits = probe(batch_features)
        loss = F.binary_cross_entropy_with_logits(logits, batch_targets)
        loss.backward()
        max_norm = eval_config.gradient_clip if eval_config.gradient_clip > 0 else float("inf")
        torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % eval_config.checkpoint_interval == 0:
            eval_call_timings: dict[str, float | int] = {}
            eval_call_start = perf_counter()
            metrics, pairwise_confusion = _evaluate_probe_features(
                probe=probe,
                features=val_features,
                targets=val_targets,
                device=device,
                batch_size=eval_config.batch_size,
                class_names=class_names,
                compute_pairwise_confusion=compute_pairwise_confusion,
                timings=eval_call_timings,
            )
            eval_total_s += perf_counter() - eval_call_start
            eval_calls += 1
            eval_forward_s += float(eval_call_timings.get("forward_s", 0.0))
            eval_numpy_s += float(eval_call_timings.get("numpy_s", 0.0))
            eval_metrics_s += float(eval_call_timings.get("metrics_s", 0.0))
            eval_pairwise_confusion_s += float(eval_call_timings.get("pairwise_confusion_s", 0.0))
            macro_auc = metrics[0]
            macro_auprc = metrics[2]
            if best_auc_metrics is None or macro_auc > best_auc_metrics[0]:
                best_auc_metrics = metrics
                best_auc_step = step + 1
                best_auc_pairwise_confusion = pairwise_confusion
            if best_auprc_metrics is None or macro_auprc > best_auprc_metrics[2]:
                best_auprc_metrics = metrics
                best_auprc_step = step + 1
                best_auprc_pairwise_confusion = pairwise_confusion

    if (
        best_auc_metrics is None
        or best_auc_step is None
        or best_auprc_metrics is None
        or best_auprc_step is None
    ):
        raise ValueError("Offline probe did not evaluate any checkpoints")

    def _pack_best(
        metrics: tuple[float, dict[str, float], float, dict[str, float]],
        pairwise_confusion: dict[str, object] | None,
        *,
        best_step: int,
    ) -> dict[str, object]:
        macro_auc, per_class_auc, macro_auprc, per_class_auprc = metrics
        packed: dict[str, object] = {
            "macro_auc": float(macro_auc),
            "per_class_auc": per_class_auc,
            "macro_auprc": float(macro_auprc),
            "per_class_auprc": per_class_auprc,
            "best_probe_step": int(best_step),
        }
        if pairwise_confusion is not None:
            packed["pairwise_confusion"] = pairwise_confusion
        return packed

    val_num_crops = int(val_features.size(1)) if val_has_crops else 1
    total_s = perf_counter() - total_start
    return {
        "best_auc": _pack_best(best_auc_metrics, best_auc_pairwise_confusion, best_step=best_auc_step),
        "best_auprc": _pack_best(
            best_auprc_metrics,
            best_auprc_pairwise_confusion,
            best_step=best_auprc_step,
        ),
        "feature_dim": int(feature_dim),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "val_num_crops": int(val_num_crops),
        "timings": {
            "classification": {
                "total_s": float(total_s),
                "train_steps_s": float(total_s - eval_total_s),
                "eval_total_s": float(eval_total_s),
                "eval_calls": int(eval_calls),
                "eval_forward_s": float(eval_forward_s),
                "eval_numpy_s": float(eval_numpy_s),
                "eval_metrics_s": float(eval_metrics_s),
                "eval_pairwise_confusion_s": float(eval_pairwise_confusion_s),
            }
        },
    }


def _clip_linear_per_target_(linear: nn.Linear, max_norm: float) -> None:
    if linear.weight.grad is None:
        raise ValueError("Linear weight gradients missing during dense probe clipping")
    if linear.bias is None or linear.bias.grad is None:
        raise ValueError("Linear bias gradients missing during dense probe clipping")
    if max_norm <= 0:
        return
    weight_grad = linear.weight.grad
    bias_grad = linear.bias.grad
    row_norm_sq = torch.sum(weight_grad * weight_grad, dim=1) + bias_grad * bias_grad
    row_norm = torch.sqrt(row_norm_sq)
    clip_coef = torch.ones_like(row_norm)
    over_limit = row_norm > max_norm
    clip_coef[over_limit] = max_norm / row_norm[over_limit]
    linear.weight.grad.mul_(clip_coef.unsqueeze(1))
    linear.bias.grad.mul_(clip_coef)


def _train_linear_regression_probe(
    *,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    target_names: list[str],
    eval_config: OfflineProbeConfig,
    device: torch.device,
    val_has_crops: bool,
    log_per_target: bool,
) -> dict[str, object]:
    total_start = perf_counter()
    if train_features.dim() != 2:
        raise ValueError(f"Expected train features to be 2D, got {tuple(train_features.shape)}")
    if val_features.dim() not in (2, 3):
        raise ValueError(f"Expected val features to be 2D or 3D, got {tuple(val_features.shape)}")
    if train_targets.dim() != 2 or val_targets.dim() != 2:
        raise ValueError("Dense targets must be 2D")

    feature_dim = int(train_features.size(1))
    train_size = int(train_features.size(0))
    val_size = int(val_targets.size(0))
    if train_size < eval_config.batch_size:
        raise ValueError(
            "Offline probe train split too small for batch_size="
            f"{eval_config.batch_size}: {train_size} samples"
        )
    if val_size < 1:
        raise ValueError("Offline probe validation split is empty")

    train_valid = torch.isfinite(train_targets)
    val_valid = torch.isfinite(val_targets)
    missing_train = [
        name
        for name, count in zip(target_names, train_valid.sum(dim=0).tolist(), strict=True)
        if count == 0
    ]
    if missing_train:
        raise ValueError(
            "Train split has no finite dense samples for targets: " + ", ".join(missing_train)
        )
    missing_val = [
        name
        for name, count in zip(target_names, val_valid.sum(dim=0).tolist(), strict=True)
        if count == 0
    ]
    if missing_val:
        raise ValueError(
            "Validation split has no finite dense samples for targets: " + ", ".join(missing_val)
        )

    dense_dim = int(train_targets.size(1))
    probe = nn.Sequential(
        nn.LayerNorm(feature_dim, elementwise_affine=False),
        nn.Linear(feature_dim, dense_dim),
    ).to(device)
    linear_head = probe[1]
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=eval_config.learning_rate,
        betas=eval_config.opt_betas,
        weight_decay=eval_config.weight_decay,
    )
    lr_schedule = cosine_schedule(
        total_steps=eval_config.steps,
        start_value=eval_config.learning_rate,
        final_value=eval_config.final_learning_rate,
        warmup_steps=eval_config.learning_rate_warmup_steps,
        warmup_start_value=1e-6,
    )
    train_feature_loader = DataLoader(
        dataset=TensorDataset(train_features, train_targets),
        batch_size=eval_config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    def cycle(dataloader):
        while True:
            yield from dataloader

    train_iterator = cycle(train_feature_loader)
    best_metrics = None
    best_steps = None
    eval_calls = 0
    eval_total_s = 0.0
    eval_forward_s = 0.0
    eval_finalize_s = 0.0

    for step in range(eval_config.steps):
        update_learning_rate_(optimizer, next(lr_schedule))
        batch_features, batch_targets = next(train_iterator)
        batch_features = batch_features.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        predictions = probe(batch_features)
        valid = torch.isfinite(batch_targets)
        if not torch.any(valid):
            raise ValueError("Training batch has no finite dense targets")
        batch_targets = torch.nan_to_num(batch_targets, nan=0.0)
        errors = (predictions - batch_targets) * valid.to(dtype=predictions.dtype)
        sq_error = errors * errors
        sum_sq_error = torch.sum(sq_error, dim=0)
        count = torch.sum(valid, dim=0).to(dtype=predictions.dtype)
        valid_targets = count > 0
        if not torch.any(valid_targets):
            raise ValueError("Training batch has no valid targets after masking")
        mse = sum_sq_error[valid_targets] / count[valid_targets]
        loss = mse.mean()
        loss.backward()
        _clip_linear_per_target_(linear_head, float(eval_config.gradient_clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % eval_config.checkpoint_interval == 0:
            eval_call_timings: dict[str, float | int] = {}
            eval_call_start = perf_counter()
            per_target_metrics = _evaluate_regression_features_streaming(
                probe=probe,
                features=val_features,
                targets=val_targets,
                device=device,
                batch_size=eval_config.batch_size,
                target_names=target_names,
                timings=eval_call_timings,
            )
            eval_total_s += perf_counter() - eval_call_start
            eval_calls += 1
            eval_forward_s += float(eval_call_timings.get("forward_s", 0.0))
            eval_finalize_s += float(eval_call_timings.get("finalize_s", 0.0))
            per_target_mse = per_target_metrics["mse"]
            if best_metrics is None:
                best_metrics = {key: value.clone() for key, value in per_target_metrics.items()}
                best_steps = torch.full((dense_dim,), step + 1, dtype=torch.int64)
            else:
                improved = per_target_mse < best_metrics["mse"]
                step_tensor = torch.full((dense_dim,), step + 1, dtype=torch.int64)
                for key, values in per_target_metrics.items():
                    best_metrics[key] = torch.where(improved, values, best_metrics[key])
                best_steps = torch.where(improved, step_tensor, best_steps)

    if best_metrics is None or best_steps is None:
        raise ValueError("Offline dense probe did not evaluate any checkpoints")

    result: dict[str, object] = {
        "macro_mse": float(best_metrics["mse"].mean()),
        "macro_mae": float(best_metrics["mae"].mean()),
        "macro_r2": float(torch.nanmean(best_metrics["r2"])),
        "macro_pearson": float(torch.nanmean(best_metrics["pearson"])),
        "timings": {
            "regression": {
                "total_s": float(perf_counter() - total_start),
                "train_steps_s": float((perf_counter() - total_start) - eval_total_s),
                "eval_total_s": float(eval_total_s),
                "eval_calls": int(eval_calls),
                "eval_forward_s": float(eval_forward_s),
                "eval_finalize_s": float(eval_finalize_s),
            }
        },
    }
    if log_per_target:
        result["per_target_mse"] = {
            name: float(value)
            for name, value in zip(target_names, best_metrics["mse"].tolist(), strict=True)
        }
        result["per_target_mae"] = {
            name: float(value)
            for name, value in zip(target_names, best_metrics["mae"].tolist(), strict=True)
        }
        result["per_target_r2"] = {
            name: float(value)
            for name, value in zip(target_names, best_metrics["r2"].tolist(), strict=True)
        }
        result["per_target_pearson"] = {
            name: float(value)
            for name, value in zip(target_names, best_metrics["pearson"].tolist(), strict=True)
        }
        result["per_target_best_step"] = {
            name: int(value) for name, value in zip(target_names, best_steps.tolist(), strict=True)
        }
    return result


def offline_probe_run_linear_multihead_by_layer(
    *,
    encoder: nn.Module,
    representation_fn: Callable[[torch.Tensor], dict[int, torch.Tensor]] | None = None,
    train_representation_fn: Callable[[torch.Tensor], dict[int, torch.Tensor]] | None = None,
    val_representation_fn: Callable[[torch.Tensor], dict[int, torch.Tensor]] | None = None,
    train_loader: Iterable[tuple[torch.Tensor, ...]],
    val_loader: Iterable[tuple[torch.Tensor, ...]],
    num_classes: int,
    class_names: list[str],
    eval_config: OfflineProbeConfig,
    device: torch.device,
    auto_mixed_precision,
    layers_categorical: tuple[int, ...],
    layers_dense: tuple[int, ...],
    layers_confusion: tuple[int, ...],
    dense_target_names: list[str] | None = None,
    dense_log_per_target: bool | None = None,
) -> dict[str, object]:
    _validate_offline_probe_config(eval_config=eval_config)
    if representation_fn is None and (train_representation_fn is None or val_representation_fn is None):
        raise ValueError(
            "Either representation_fn or both train_representation_fn/val_representation_fn must be provided"
        )
    if train_representation_fn is None:
        train_representation_fn = representation_fn
    if val_representation_fn is None:
        val_representation_fn = representation_fn

    if len(set(layers_categorical)) != len(layers_categorical):
        raise ValueError(f"layers_categorical must be unique, got {layers_categorical}")
    if len(set(layers_dense)) != len(layers_dense):
        raise ValueError(f"layers_dense must be unique, got {layers_dense}")
    if len(set(layers_confusion)) != len(layers_confusion):
        raise ValueError(f"layers_confusion must be unique, got {layers_confusion}")
    if not set(layers_confusion).issubset(set(layers_categorical)):
        raise ValueError("layers_confusion must be a subset of layers_categorical")
    layers_all = tuple(sorted(set(layers_categorical) | set(layers_dense) | set(layers_confusion)))
    if not layers_all:
        raise ValueError("At least one layer must be requested")

    total_start = perf_counter()
    train_collect_timings: dict[str, float | int] = {}
    train_features_by_layer, train_targets, train_dense_targets, train_has_crops = _collect_probe_features_by_layer(
        encoder=encoder,
        representation_fn=train_representation_fn,
        layers=layers_all,
        loader=train_loader,
        device=device,
        auto_mixed_precision=auto_mixed_precision,
        allow_crops=False,
        timings=train_collect_timings,
    )
    if train_has_crops:
        raise ValueError("Offline probe training does not support multi-crop train features")

    val_collect_timings: dict[str, float | int] = {}
    val_features_by_layer, val_targets, val_dense_targets, val_has_crops = _collect_probe_features_by_layer(
        encoder=encoder,
        representation_fn=val_representation_fn,
        layers=layers_all,
        loader=val_loader,
        device=device,
        auto_mixed_precision=auto_mixed_precision,
        allow_crops=True,
        timings=val_collect_timings,
    )

    categorical: dict[int, dict[str, object]] = {}
    for layer in layers_categorical:
        categorical[layer] = _train_linear_classification_probe(
            train_features=train_features_by_layer[layer],
            train_targets=train_targets,
            val_features=val_features_by_layer[layer],
            val_targets=val_targets,
            num_classes=num_classes,
            class_names=class_names,
            eval_config=eval_config,
            device=device,
            val_has_crops=val_has_crops,
            compute_pairwise_confusion=layer in set(layers_confusion),
        )

    dense: dict[int, dict[str, object]] = {}
    if layers_dense and (train_dense_targets is not None or val_dense_targets is not None):
        if train_dense_targets is None or val_dense_targets is None:
            raise ValueError("Dense targets must be present in both train and val splits")
        if dense_target_names is None or dense_log_per_target is None:
            raise ValueError("Dense target metadata must be provided when running dense probes")
        for layer in layers_dense:
            dense[layer] = _train_linear_regression_probe(
                train_features=train_features_by_layer[layer],
                train_targets=train_dense_targets,
                val_features=val_features_by_layer[layer],
                val_targets=val_dense_targets,
                target_names=dense_target_names,
                eval_config=eval_config,
                device=device,
                val_has_crops=val_has_crops,
                log_per_target=bool(dense_log_per_target),
            )

    example_val_features = val_features_by_layer[layers_all[0]]
    val_num_crops = int(example_val_features.size(1)) if val_has_crops else 1
    feature_dim = int(train_features_by_layer[layers_all[0]].size(-1))
    return {
        "categorical": categorical,
        "dense": dense,
        "dense_targets_available": bool(train_dense_targets is not None and val_dense_targets is not None),
        "feature_dim": int(feature_dim),
        "train_size": int(train_targets.size(0)),
        "val_size": int(val_targets.size(0)),
        "val_num_crops": int(val_num_crops),
        "timings": {
            "total_s": float(perf_counter() - total_start),
            "collect_train": train_collect_timings,
            "collect_val": val_collect_timings,
        },
    }
