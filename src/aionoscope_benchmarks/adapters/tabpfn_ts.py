from __future__ import annotations

import copy
import sys

import numpy as np
import torch

from .base import FrozenTimeSeriesAdapter
from .forecast_tables import (
    build_context_dataframe,
    make_cached_representation_fn,
    sample_probe_indices,
    subset_split,
)


class TabPFNTSAdapter(FrozenTimeSeriesAdapter):
    model_name = "TabPFN-TS"
    model_slug = "TabPFN-TS"
    source = "https://github.com/PriorLabs/tabpfn-time-series"
    checkpoint = "tabpfn-v2-regressor-2noar4o2.ckpt"
    import_path = "tabpfn_time_series"
    env_name = "tabular"
    default_encode_batch_size = 4096
    use_bfloat16_amp = False

    max_context_length = 4096
    prediction_length = 1
    n_estimators = 8
    fit_seed = 0
    probe_train_sample_cap = 128
    probe_val_sample_cap = 128

    def __init__(self) -> None:
        super().__init__()
        try:
            from tabpfn_time_series import (
                TABPFN_DEFAULT_CONFIG,
                TABPFN_TS_DEFAULT_FEATURES,
                FeatureTransformer,
                TimeSeriesDataFrame,
            )
            from tabpfn_time_series.data_preparation import generate_test_X
        except ImportError as exc:
            raise ImportError(
                "TabPFN-TS adapter requires the `tabpfn-time-series` package in the `tabular` "
                "environment. Install it in `.venv-tabular` and retry."
            ) from exc

        self._split_feature_cache: dict[str, dict[int, torch.Tensor]] = {}
        self._feature_transformer_class = FeatureTransformer
        self._time_series_dataframe_class = TimeSeriesDataFrame
        self._default_temporal_features = tuple(copy.deepcopy(list(TABPFN_TS_DEFAULT_FEATURES)))
        self._default_tabpfn_config = dict(TABPFN_DEFAULT_CONFIG)
        self._generate_test_X = generate_test_X
        self._sampling_frequency_hz: int | None = None
        self._runtime_device: torch.device = torch.device("cpu")
        self._embedding_dim: int | None = None
        self.benchmark_sequence_length = int(self.max_context_length)
        self.benchmark_sequence_length_source = "tabpfn_ts_pipeline.max_context_length"
        self.probe_train_split: dict[str, torch.Tensor] | None = None
        self.probe_val_split: dict[str, torch.Tensor] | None = None

    @property
    def available_layers(self) -> tuple[int, ...]:
        return (0,)

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["max_context_length"] = int(self.max_context_length)
        payload["prediction_length"] = int(self.prediction_length)
        payload["n_estimators"] = int(self.n_estimators)
        payload["temporal_features"] = [
            type(feature).__name__ for feature in self._default_temporal_features
        ]
        payload["representation_kind"] = "averaged TabPFNRegressor forecast-query embeddings"
        payload["forecast_query_policy"] = "synthetic_next_step"
        payload["tabpfn_model_config"] = dict(self._default_tabpfn_config)
        payload["probe_train_sample_cap"] = int(self.probe_train_sample_cap)
        payload["probe_val_sample_cap"] = int(self.probe_val_sample_cap)
        if self._embedding_dim is not None:
            payload["embedding_dim"] = int(self._embedding_dim)
        if self.probe_train_split is not None:
            payload["probe_train_sample_count"] = int(self.probe_train_split["x"].size(0))
        if self.probe_val_split is not None:
            payload["probe_val_sample_count"] = int(self.probe_val_split["x"].size(0))
        return payload

    def _reduce_inputs(self, x: torch.Tensor) -> np.ndarray:
        self.validate_benchmark_input(x, channels=1)
        return np.ascontiguousarray(
            x[:, 0, :].to(dtype=torch.float32).cpu().numpy(),
            dtype=np.float32,
        )

    def _require_sampling_frequency_hz(self) -> int:
        if self._sampling_frequency_hz is None:
            raise RuntimeError("TabPFN-TS sampling frequency is not prepared yet")
        return int(self._sampling_frequency_hz)

    def _tabular_device(self) -> str:
        return "cuda" if self._runtime_device.type == "cuda" and torch.cuda.is_available() else "cpu"

    def _make_feature_transformer(self):
        return self._feature_transformer_class(copy.deepcopy(list(self._default_temporal_features)))

    def _build_regressor(self, *, seed: int):
        from tabpfn import TabPFNRegressor

        config = dict(self._default_tabpfn_config)
        config.update(
            {
                "n_estimators": int(self.n_estimators),
                "device": self._tabular_device(),
                "n_preprocessing_jobs": 1,
                "random_state": int(seed),
            }
        )
        return TabPFNRegressor(**config)

    def _build_feature_tables(self, waveforms: np.ndarray):
        context_df = build_context_dataframe(
            waveforms,
            sampling_frequency_hz=self._require_sampling_frequency_hz(),
        )
        context_tsdf = self._time_series_dataframe_class.from_data_frame(context_df)
        future_tsdf = self._generate_test_X(
            context_tsdf,
            prediction_length=int(self.prediction_length),
        )
        transformer = self._make_feature_transformer()
        return transformer.transform(context_tsdf, future_tsdf)

    def _encode_waveforms(
        self,
        waveforms: np.ndarray,
        *,
        split_name: str,
    ) -> dict[int, torch.Tensor]:
        train_tsdf, test_tsdf = self._build_feature_tables(waveforms)
        total_series = int(waveforms.shape[0])
        features: np.ndarray | None = None
        log_every = max(1, (total_series + 3) // 4)

        for sample_index in range(total_series):
            if (
                sample_index == 0
                or sample_index + 1 == total_series
                or (sample_index + 1) % log_every == 0
            ):
                print(
                    f"[TabPFN-TS] {split_name} embedding progress: "
                    f"series {sample_index + 1}/{total_series}",
                    file=sys.stderr,
                    flush=True,
                )

            train_item = train_tsdf.loc[sample_index].copy()
            test_item = test_tsdf.loc[sample_index].copy()
            train_x = train_item.drop(columns=["target"])
            train_y = train_item["target"].to_numpy(dtype=np.float32, copy=True)
            test_x = test_item.drop(columns=["target"])

            regressor = self._build_regressor(seed=self.fit_seed + sample_index)
            regressor.fit(train_x, train_y)
            embedding = np.asarray(regressor.get_embeddings(test_x, data_source="test"), dtype=np.float32)
            if embedding.ndim != 3 or embedding.shape[1] != self.prediction_length:
                raise RuntimeError(
                    "TabPFN-TS expected embeddings shaped "
                    f"[n_estimators, prediction_length, dim], got {embedding.shape}"
                )
            embedding_vector = embedding.mean(axis=0)[0]
            if features is None:
                features = np.empty((total_series, embedding_vector.shape[0]), dtype=np.float32)
            features[sample_index] = embedding_vector

        if features is None:
            raise RuntimeError("TabPFN-TS did not produce any embeddings")
        self._embedding_dim = int(features.shape[1])
        return {0: torch.from_numpy(features)}

    def prepare(
        self,
        *,
        manifest: dict[str, object],
        train_split: dict[str, torch.Tensor],
        val_split: dict[str, torch.Tensor],
    ) -> None:
        self._sampling_frequency_hz = int(manifest["sampling_frequency"])
        self._runtime_device = torch.device(train_split["x"].device)
        train_waveforms = self._reduce_inputs(train_split["x"])
        train_probe_indices = sample_probe_indices(
            train_waveforms.shape[0],
            sample_cap=self.probe_train_sample_cap,
            seed=self.fit_seed + 10_000,
        )
        self.probe_train_split = subset_split(train_split, train_probe_indices)
        probe_train_waveforms = train_waveforms[train_probe_indices]
        self._split_feature_cache["train"] = self._encode_waveforms(
            probe_train_waveforms,
            split_name="train",
        )
        self.update_probe_val_split(val_split=val_split)

    def update_probe_val_split(
        self,
        *,
        val_split: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        val_waveforms = self._reduce_inputs(val_split["x"])
        val_probe_indices = sample_probe_indices(
            val_waveforms.shape[0],
            sample_cap=self.probe_val_sample_cap,
            seed=self.fit_seed + 20_000,
        )
        self.probe_val_split = subset_split(val_split, val_probe_indices)
        probe_val_waveforms = val_waveforms[val_probe_indices]
        self._split_feature_cache["val"] = self._encode_waveforms(
            probe_val_waveforms,
            split_name="val",
        )
        return self.probe_val_split

    def make_representation_fn(
        self,
        *,
        layers: tuple[int, ...],
        split: str = "val",
    ):
        requested_layers = tuple(int(layer) for layer in layers)
        if requested_layers != (0,):
            raise ValueError(f"TabPFN-TS only exposes layer 0, got {requested_layers}")
        return make_cached_representation_fn(
            model_name=self.model_name,
            split_feature_cache=self._split_feature_cache,
            layers=requested_layers,
            split=split,
        )

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        raise RuntimeError("TabPFN-TS uses cached split features; call make_representation_fn()")
