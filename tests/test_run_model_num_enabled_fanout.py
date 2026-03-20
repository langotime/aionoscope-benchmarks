from __future__ import annotations

from pathlib import Path

from aionoscope_benchmarks import run_model


def test_result_output_path_includes_num_enabled_suffix(tmp_path: Path) -> None:
    out_path = run_model.result_output_path(
        out_dir=tmp_path,
        model_slug="Chronos-2",
        num_enabled=3,
    )

    assert out_path == tmp_path / "Chronos-2__num_enabled_3.json"


def test_run_single_model_fans_out_requested_num_enabled_values(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    def _fake_resolve_requested_num_enabled_values(*, config_path: Path, requested_num_enabled_values):
        del config_path, requested_num_enabled_values
        return [1, 3]

    def _fake_run_single_model_for_num_enabled(**kwargs):
        calls.append(int(kwargs["num_enabled"]))
        return run_model.result_output_path(
            out_dir=tmp_path,
            model_slug="Chronos-2",
            num_enabled=int(kwargs["num_enabled"]),
        )

    monkeypatch.setattr(
        run_model,
        "resolve_requested_num_enabled_values",
        _fake_resolve_requested_num_enabled_values,
    )
    monkeypatch.setattr(
        run_model,
        "run_single_model_for_num_enabled",
        _fake_run_single_model_for_num_enabled,
    )

    out_paths = run_model.run_single_model(
        model_name="Chronos-2",
        out_dir=tmp_path,
        num_enabled_values=[1, 3],
    )

    assert calls == [1, 3]
    assert out_paths == [
        tmp_path / "Chronos-2__num_enabled_1.json",
        tmp_path / "Chronos-2__num_enabled_3.json",
    ]
