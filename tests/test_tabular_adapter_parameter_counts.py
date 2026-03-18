import torch

from aionoscope_benchmarks.adapters.tabicl import TabICLAdapter
from aionoscope_benchmarks.adapters.tabpfn import TabPFNAdapter


class _FakeClassifier:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model_ = model


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
