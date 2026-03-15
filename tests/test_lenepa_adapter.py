from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

import pytest
import torch

from aionoscope_benchmarks.adapters import lenepa as lenepa_module
from aionoscope_benchmarks.adapters.lenepa import (
    LeNEPAAionoAdapter,
    LeNEPACauKer2MAdapter,
    LeNEPACauKer2M20KAdapter,
)
from aionoscope_benchmarks.constants import FOUNDATIONAL_MODELS
from aionoscope_benchmarks.model_registry import MODEL_SPECS, all_foundational_model_names


def _write_bundle(
    tmp_path: Path,
    *,
    with_tokenizer: bool,
    config: dict[str, object],
) -> dict[str, Path]:
    config_path = tmp_path / "lenepa_encoder_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    weights_path = tmp_path / "lenepa_encoder.safetensors"
    weights_path.write_bytes(b"")

    tokenizer_block = """
    def _tokenize(self, x):
        b, _, length = x.shape
        return x[:, 0, :4].reshape(b, 2, 2)
    """
    patch_embed_block = """
    class PatchEmbed(nn.Module):
        def forward(self, x):
            b, _, length = x.shape
            return x[:, 0, :4].reshape(b, 2, 2) + 5.0
    """
    lines = [
        "from pathlib import Path",
        "",
        "import torch",
        "from torch import nn",
        "",
        "",
        "class AddConstant(nn.Module):",
        "    def __init__(self, delta):",
        "        super().__init__()",
        "        self.delta = float(delta)",
        "",
        "    def forward(self, x):",
        "        return x + self.delta",
        "",
        "",
        "class FinalNorm(nn.Module):",
        "    def forward(self, x):",
        "        return x + 100.0",
        "",
        "",
    ]
    if not with_tokenizer:
        lines.extend(dedent(patch_embed_block).strip().splitlines())
        lines.extend(["", ""])
    lines.extend(
        [
            "class LeNEPAEncoder(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
        ]
    )
    if not with_tokenizer:
        lines.append("        self.patch_embed = PatchEmbed()")
    lines.extend(
        [
            "        self.blocks = nn.ModuleList([AddConstant(10.0), AddConstant(20.0)])",
            "        self.norm = FinalNorm()",
            "",
        ]
    )
    if with_tokenizer:
        lines.extend(f"    {line}" if line else "" for line in dedent(tokenizer_block).strip().splitlines())
        lines.append("")
    lines.extend(
        [
            "",
            "def load_lenepa_encoder(*, weights_path: Path, device: torch.device):",
            '    assert Path(weights_path).name == "lenepa_encoder.safetensors"',
            "    return LeNEPAEncoder().to(device)",
            "",
        ]
    )
    inference_code = "\n".join(lines)
    inference_path = tmp_path / "inference.py"
    inference_path.write_text(inference_code, encoding="utf-8")
    return {
        "config": config_path,
        "weights": weights_path,
        "inference": inference_path,
    }


@pytest.fixture(autouse=True)
def clear_lenepa_module_cache() -> None:
    to_remove = [name for name in list(sys.modules) if name.startswith("_aionoscope_lenepa_")]
    for name in to_remove:
        sys.modules.pop(name, None)


def test_lenepa_registry_contains_both_checkpoints() -> None:
    assert "LeNEPA-Aiono" in MODEL_SPECS
    assert "LeNEPA-CauKer2M" in MODEL_SPECS
    assert "LeNEPA-CauKer2M-20k" in MODEL_SPECS
    assert "LeNEPA-Aiono" in FOUNDATIONAL_MODELS
    assert "LeNEPA-CauKer2M" in FOUNDATIONAL_MODELS
    assert "LeNEPA-CauKer2M-20k" in FOUNDATIONAL_MODELS
    assert "LeNEPA-Aiono" in all_foundational_model_names()
    assert "LeNEPA-CauKer2M" in all_foundational_model_names()
    assert "LeNEPA-CauKer2M-20k" in all_foundational_model_names()


def test_lenepa_adapter_uses_exported_tokenizer_and_final_norm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(
        tmp_path,
        with_tokenizer=True,
        config={
            "format": "lenepa_encoder",
            "channels": ["I"],
            "channel_size": 5000,
            "sampling_frequency": 500,
            "patch_size": 25,
            "num_patches": 200,
            "dim": 192,
            "depth": 2,
        },
    )

    def fake_download(*, repo_id: str, filename: str) -> str:
        del repo_id
        if filename == "inference.py":
            return str(bundle["inference"])
        if filename == "lenepa_encoder.safetensors":
            return str(bundle["weights"])
        if filename == "lenepa_encoder_config.json":
            return str(bundle["config"])
        raise AssertionError(f"Unexpected filename {filename}")

    monkeypatch.setattr(lenepa_module, "hf_hub_download", fake_download)

    adapter = LeNEPAAionoAdapter()
    x = torch.arange(5000, dtype=torch.float32).reshape(1, 1, 5000)
    reps = adapter.forward_layer_dict(x, layers=(0, 1, 2))

    assert tuple(adapter.available_layers) == (0, 1, 2)
    assert torch.equal(reps[0], torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert torch.equal(reps[1], torch.tensor([[11.0, 12.0]], dtype=torch.float32))
    assert torch.equal(reps[2], torch.tensor([[131.0, 132.0]], dtype=torch.float32))
    metadata = adapter.adapter_metadata()
    assert metadata["published_sampling_frequency"] == 500
    assert metadata["published_channels"] == ["I"]
    assert metadata["patch_size"] == 25
    assert metadata["n_benchmark_layers"] == 3
    assert metadata["benchmark_sequence_length"] == 5000
    assert metadata["input_length_policy"] == "exact"


def test_lenepa_adapter_falls_back_to_patch_embed_when_no_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(
        tmp_path,
        with_tokenizer=False,
        config={
            "format": "lenepa_encoder",
            "channels": ["c0"],
            "channel_size": 5000,
            "sampling_frequency": 1,
            "patch_size": 8,
            "num_patches": 625,
            "dim": 256,
            "depth": 2,
            "nepa_static_tokenizer": "conv_patch_embed",
            "nepa_patch_embed_scalar_stats_mode": "patch_norm",
        },
    )

    def fake_download(*, repo_id: str, filename: str) -> str:
        del repo_id
        if filename == "inference.py":
            return str(bundle["inference"])
        if filename == "lenepa_encoder.safetensors":
            return str(bundle["weights"])
        if filename == "lenepa_encoder_config.json":
            return str(bundle["config"])
        raise AssertionError(f"Unexpected filename {filename}")

    monkeypatch.setattr(lenepa_module, "hf_hub_download", fake_download)

    adapter = LeNEPACauKer2MAdapter()
    x = torch.arange(5000, dtype=torch.float32).reshape(1, 1, 5000)
    reps = adapter.forward_layer_dict(x, layers=(0, 2))

    assert set(reps) == {0, 2}
    assert torch.equal(reps[0], torch.tensor([[6.0, 7.0]], dtype=torch.float32))
    assert torch.equal(reps[2], torch.tensor([[136.0, 137.0]], dtype=torch.float32))
    metadata = adapter.adapter_metadata()
    assert metadata["patch_stats_mode"] == "patch_norm"
    assert metadata["published_sampling_frequency"] == 1
    assert metadata["benchmark_sequence_length"] == 5000


def test_lenepa_cauker2m_20k_adapter_reuses_published_bundle_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(
        tmp_path,
        with_tokenizer=False,
        config={
            "format": "lenepa_encoder",
            "channels": ["c0"],
            "channel_size": 5000,
            "sampling_frequency": 1,
            "patch_size": 8,
            "num_patches": 625,
            "dim": 256,
            "depth": 2,
            "nepa_static_tokenizer": "conv_patch_embed",
            "nepa_patch_embed_scalar_stats_mode": "patch_norm",
        },
    )

    def fake_download(*, repo_id: str, filename: str) -> str:
        del repo_id
        if filename == "inference.py":
            return str(bundle["inference"])
        if filename == "lenepa_encoder.safetensors":
            return str(bundle["weights"])
        if filename == "lenepa_encoder_config.json":
            return str(bundle["config"])
        raise AssertionError(f"Unexpected filename {filename}")

    monkeypatch.setattr(lenepa_module, "hf_hub_download", fake_download)

    adapter = LeNEPACauKer2M20KAdapter()
    x = torch.arange(5000, dtype=torch.float32).reshape(1, 1, 5000)
    reps = adapter.forward_layer_dict(x, layers=(0, 2))

    assert set(reps) == {0, 2}
    assert torch.equal(reps[0], torch.tensor([[6.0, 7.0]], dtype=torch.float32))
    assert torch.equal(reps[2], torch.tensor([[136.0, 137.0]], dtype=torch.float32))
    metadata = adapter.adapter_metadata()
    assert metadata["patch_stats_mode"] == "patch_norm"
    assert metadata["published_sampling_frequency"] == 1
    assert metadata["benchmark_sequence_length"] == 5000
