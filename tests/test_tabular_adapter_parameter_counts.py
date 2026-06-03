import torch

from aionoscope_benchmarks.adapters.tabicl import TabICLAdapter
from aionoscope_benchmarks.adapters.tabpfn import TabPFNAdapter


class _FakeClassifier:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model_ = model
        self.classes_ = torch.tensor([0, 1]).numpy()

    def predict_proba(self, x):
        positive = x.mean(axis=1).clip(0.0, 1.0)
        return torch.stack(
            [
                torch.from_numpy(1.0 - positive),
                torch.from_numpy(positive),
            ],
            dim=1,
        ).numpy()


class _TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)


def test_tabpfn_adapter_reports_single_backbone_parameter_count() -> None:
    adapter = TabPFNAdapter()
    adapter._classifiers = [_FakeClassifier(_TinyBackbone())]

    metadata = adapter.adapter_metadata()

    assert metadata["parameter_count"] == 18
    assert metadata["parameter_count_total"] == 18
    assert metadata["trainable_parameter_count"] == 18
    assert metadata["parameter_count_source"] == "official_tabpfn_classifier_model"
    assert metadata["parameter_count_prefix_by_layer"] == {"0": 18}
    assert metadata["parameter_count_reference"] == (
        "single official TabPFN backbone; not multiplied by one-vs-rest labels or estimator replicas"
    )


def test_tabicl_adapter_reports_single_backbone_parameter_count() -> None:
    adapter = TabICLAdapter()
    adapter._classifiers = [_FakeClassifier(_TinyBackbone())]

    metadata = adapter.adapter_metadata()

    assert metadata["parameter_count"] == 18
    assert metadata["parameter_count_total"] == 18
    assert metadata["trainable_parameter_count"] == 18
    assert metadata["parameter_count_source"] == "official_tabicl_classifier_model"
    assert metadata["parameter_count_prefix_by_layer"] == {"0": 18}
    assert metadata["parameter_count_reference"] == (
        "single official TabICL backbone; not multiplied by one-vs-rest labels or estimator replicas"
    )


def test_tabpfn_manifold_representation_uses_dynamic_inputs() -> None:
    adapter = TabPFNAdapter()
    adapter._class_names = ["a", "b"]
    adapter._classifiers = [_FakeClassifier(_TinyBackbone()), _FakeClassifier(_TinyBackbone())]
    x = torch.full((3, 1, adapter.reduced_feature_length), 0.25)

    representation_fn = adapter.make_representation_fn(layers=(0,), split="manifold_train")
    features = representation_fn(x)[0]

    assert features.shape == (3, 2)
    assert torch.allclose(features, torch.full((3, 2), 0.25))


def test_tabicl_manifold_representation_uses_dynamic_inputs() -> None:
    adapter = TabICLAdapter()
    adapter._class_names = ["a", "b"]
    adapter._classifiers = [_FakeClassifier(_TinyBackbone()), _FakeClassifier(_TinyBackbone())]
    x = torch.full((3, 1, adapter.reduced_feature_length), 0.75)

    representation_fn = adapter.make_representation_fn(layers=(0,), split="manifold_val")
    features = representation_fn(x)[0]

    assert features.shape == (3, 2)
    assert torch.allclose(features, torch.full((3, 2), 0.75))
