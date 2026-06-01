from __future__ import annotations

import torch
import pytest

from aionoscope_benchmarks.manifold_config import DEFAULT_MANIFOLD_CALIBRATION_TARGETS
from aionoscope_benchmarks.manifold_slices import build_controlled_manifold_slice


@pytest.mark.parametrize("target_name", DEFAULT_MANIFOLD_CALIBRATION_TARGETS)
def test_default_controlled_targets_have_finite_dense_target(target_name: str) -> None:
    controlled = build_controlled_manifold_slice(
        target_name=target_name,
        seq_len=512,
        grid_size=5,
        split="train",
        device=torch.device("cpu"),
    )

    assert controlled.manifest["target"]["target_name"] == target_name
    assert torch.isfinite(controlled.split["x"]).all()
    assert torch.isfinite(controlled.split["y_dense"][:, controlled.target_index]).all()


def test_controlled_sine_phase_slice_materializes_expected_contract() -> None:
    controlled = build_controlled_manifold_slice(
        target_name="sine_phase",
        seq_len=512,
        grid_size=8,
        split="train",
        repeats_per_grid_point=2,
        device=torch.device("cpu"),
    )

    assert controlled.split["x"].shape == (16, 1, 512)
    assert controlled.split["y_cls"].shape == (16, 14)
    assert controlled.split["y_dense"].shape[0] == 16
    assert controlled.manifest["schema_version"] == "manifold_controlled_slice_v0"
    assert controlled.manifest["target"]["geometry"] == "circle"
    assert controlled.manifest["target"]["component"] == "sine"
    assert controlled.manifest["dataset_manifest"]["generation_mode"] == (
        "controlled_manifold_slice_materialized"
    )
    assert controlled.grid_index.tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
    ]
    assert torch.isfinite(controlled.split["x"]).all()
    assert torch.isfinite(controlled.split["y_dense"][:, controlled.target_index]).all()

    component_keys = controlled.manifest["dataset_manifest"]["component_keys"]
    sine_index = component_keys.index("sine")
    assert torch.all(controlled.split["y_cls"][:, sine_index] == 1.0)
    assert torch.all(controlled.split["y_cls"].sum(dim=1) == 1.0)


def test_controlled_validation_slice_uses_half_grid_offset() -> None:
    train = build_controlled_manifold_slice(
        target_name="linear_trend_slope",
        seq_len=512,
        grid_size=8,
        split="train",
        device=torch.device("cpu"),
    )
    val = build_controlled_manifold_slice(
        target_name="linear_trend_slope",
        seq_len=512,
        grid_size=8,
        split="val",
        device=torch.device("cpu"),
    )

    train_grid = torch.as_tensor(train.manifest["physical_grid"])
    val_grid = torch.as_tensor(val.manifest["physical_grid"])

    assert not torch.equal(train_grid, val_grid)
    assert val.manifest["target"]["geometry"] == "interval"
    assert torch.isfinite(val.split["y_dense"][:, val.target_index]).all()


def test_controlled_view_slice_supports_wide_signed_log_sweep() -> None:
    controlled = build_controlled_manifold_slice(
        target_name="linear_trend_slope",
        seq_len=512,
        grid_size=8,
        split="train",
        device=torch.device("cpu"),
        view_grid_mode="signed_log",
        view_range_max_abs=1.0e6,
    )

    physical = controlled.physical_values
    latent = controlled.latent_coordinates

    assert controlled.manifest["sweep"]["grid_mode"] == "signed_log"
    assert controlled.manifest["sweep"]["range_policy"] == "wide_abs_1e+06"
    assert float(torch.min(physical)) == -1.0e6
    assert float(torch.max(physical)) == 1.0e6
    assert torch.all(torch.diff(physical) > 0)
    assert torch.all(torch.diff(latent) > 0)
    assert torch.isfinite(controlled.split["y_dense"][:, controlled.target_index]).all()


def test_controlled_event_time_slice_has_finite_normalized_target() -> None:
    controlled = build_controlled_manifold_slice(
        target_name="spike_time_frac",
        seq_len=512,
        grid_size=6,
        split="train",
        device=torch.device("cpu"),
    )

    target_values = controlled.split["y_dense"][:, controlled.target_index]

    assert controlled.manifest["target"]["component"] == "spike"
    assert controlled.manifest["target"]["parameter"] == "time_idx"
    assert torch.isfinite(target_values).all()
    assert torch.all((target_values >= 0.0) & (target_values <= 1.0))
    assert torch.isfinite(controlled.split["x"]).all()
