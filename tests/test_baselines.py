from __future__ import annotations

import json

import torch

from aionoscope_benchmarks.baselines import (
    PAPER_CRITICAL_BASELINES,
    collect_split_features,
    get_baseline_spec,
    resolve_baseline_names,
)
from aionoscope_benchmarks.run_baseline import run_baselines_for_num_enabled


def _toy_split(*, samples: int = 8, length: int = 128) -> dict[str, torch.Tensor]:
    x = torch.linspace(-1.0, 1.0, samples * length, dtype=torch.float32).reshape(samples, 1, length)
    y_cls = torch.zeros((samples, 14), dtype=torch.float32)
    y_cls[torch.arange(samples), torch.arange(samples) % 14] = 1.0
    y_dense = torch.full((samples, 34), float("nan"), dtype=torch.float32)
    y_dense[:, :4] = torch.arange(samples, dtype=torch.float32).unsqueeze(1) / 10.0
    return {"x": x, "y_cls": y_cls, "y_dense": y_dense}


def test_paper_critical_baseline_group_resolves_without_legacy_floor_aliases() -> None:
    names = resolve_baseline_names(["paper-critical"])

    assert names == list(PAPER_CRITICAL_BASELINES)
    assert resolve_baseline_names(["Majority", "MeanTarget"]) == ["MetricFloor"]


def test_baseline_feature_extractors_are_deterministic_and_finite() -> None:
    split = _toy_split()
    manifest = {"sampling_frequency": 500, "channel_size": 128}
    for name in resolve_baseline_names(["all"]):
        spec = get_baseline_spec(name)
        if not spec.uses_probe:
            continue
        features_a, timings_a = collect_split_features(
            spec=spec,
            split=split,
            manifest=manifest,
            seed=0,
            device=torch.device("cpu"),
            batch_size=4,
        )
        features_b, timings_b = collect_split_features(
            spec=spec,
            split=split,
            manifest=manifest,
            seed=0,
            device=torch.device("cpu"),
            batch_size=4,
        )

        assert features_a.shape[0] == split["x"].shape[0]
        assert features_a.ndim == 2
        assert torch.isfinite(features_a).all()
        assert torch.equal(features_a, features_b)
        assert timings_a["samples"] == split["x"].shape[0]
        assert timings_b["samples"] == split["x"].shape[0]


def test_tiny_baseline_run_writes_schema_compatible_json(tmp_path) -> None:
    probe_config = tmp_path / "probe.yaml"
    probe_config.write_text(
        "\n".join(
            [
                "steps: 2",
                "batch_size: 64",
                "runtime_dataset_batch_size: 64",
                "learning_rate: 1.0e-2",
                "final_learning_rate: 1.0e-2",
                "learning_rate_warmup_steps: 0",
                "weight_decay: 0.0",
                "opt_betas: [0.9, 0.999]",
                "gradient_clip: 1.0",
                "checkpoint_interval: 1",
            ]
        ),
        encoding="utf-8",
    )

    out_paths = run_baselines_for_num_enabled(
        baseline_names=["MetricFloor", "RawDownsample512"],
        channel_size=128,
        num_enabled=1,
        probe_config_path=probe_config,
        out_dir=tmp_path,
        device=torch.device("cpu"),
        feature_batch_size=64,
        train_batches=2,
        val_batches=2,
        validation_seed_values=[0],
    )

    assert len(out_paths) == 2
    for path in out_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["model"]["type"] == "baseline"
        assert payload["model"]["baseline"]["synthetic_layer"] == 0
        assert payload["dataset"]["benchmark_family"] == "aiono_basic_components"
        assert payload["dataset"]["benchmark_version"] == "v2"
        assert payload["results"]["categorical"]
        assert payload["results"]["dense"]
        assert payload["results"]["shared"]["validation_seed_values"] == [0]
