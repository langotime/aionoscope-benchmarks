from __future__ import annotations

from types import SimpleNamespace

import pytest

from aionoscope_benchmarks.run_model import _encoder_forward_runtime_summary


def test_encoder_forward_runtime_summary_records_explicit_totals() -> None:
    train_collected = SimpleNamespace(timings={"forward_s": 1.25})
    val_collected_by_seed = {
        3: SimpleNamespace(timings={"forward_s": 2.75}),
        1: SimpleNamespace(timings={"forward_s": 2.5}),
    }

    summary = _encoder_forward_runtime_summary(
        train_collected=train_collected,
        val_collected_by_seed=val_collected_by_seed,
    )

    assert summary["encoder_forward_train_s"] == pytest.approx(1.25)
    assert summary["encoder_forward_val_total_s"] == pytest.approx(5.25)
    assert summary["encoder_forward_total_s"] == pytest.approx(6.5)
    assert summary["encoder_forward_by_validation_seed_s"] == {
        "1": pytest.approx(2.5),
        "3": pytest.approx(2.75),
    }


def test_encoder_forward_runtime_summary_requires_forward_timing() -> None:
    train_collected = SimpleNamespace(timings={"total_s": 1.0})
    val_collected_by_seed = {
        0: SimpleNamespace(timings={"forward_s": 2.0}),
    }

    with pytest.raises(ValueError, match="Missing numeric timing 'forward_s'"):
        _encoder_forward_runtime_summary(
            train_collected=train_collected,
            val_collected_by_seed=val_collected_by_seed,
        )
