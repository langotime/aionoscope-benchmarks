from __future__ import annotations

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


class TabICLForecasterAdapter(FrozenTimeSeriesAdapter):
    model_name = "TabICLForecaster"
    model_slug = "TabICLForecaster"
    source = "https://github.com/soda-inria/tabicl"
    checkpoint = "tabicl-regressor-v2-20260212.ckpt"
    import_path = "tabicl[forecast]"
    env_name = "tabular"
    default_encode_batch_size = 4096
    use_bfloat16_amp = False

    max_context_length = 4096
    prediction_length = 1
    n_estimators = 8
    fit_seed = 0
    batch_size = 8
    probe_train_sample_cap = 128
    probe_val_sample_cap = 128

    def __init__(self) -> None:
        super().__init__()
        try:
            from tabicl import TabICL, TabICLForecaster
        except ImportError as exc:
            raise ImportError(
                "TabICLForecaster adapter requires `tabicl[forecast]` in the `tabular` environment. "
                "Install the forecasting extra in `.venv-tabular` and retry."
            ) from exc

        provisional_model = TabICL()
        self._split_feature_cache: dict[str, dict[int, torch.Tensor]] = {}
        self._tabicl_forecaster_class = TabICLForecaster
        self._provisional_num_layers = int(len(provisional_model.icl_predictor.tf_icl.blocks)) + 1
        self._verified_num_layers: int | None = None
        self._sampling_frequency_hz: int | None = None
        self._runtime_device: torch.device = torch.device("cpu")
        self._representation_dim: int | None = None
        self.benchmark_sequence_length = int(self.max_context_length)
        self.benchmark_sequence_length_source = "tabicl_forecaster.max_context_length"
        self.probe_train_split: dict[str, torch.Tensor] | None = None
        self.probe_val_split: dict[str, torch.Tensor] | None = None

    @property
    def available_layers(self) -> tuple[int, ...]:
        num_layers = self._verified_num_layers or self._provisional_num_layers
        return tuple(range(int(num_layers)))

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["max_context_length"] = int(self.max_context_length)
        payload["prediction_length"] = int(self.prediction_length)
        payload["n_estimators"] = int(self.n_estimators)
        payload["checkpoint_version"] = self.checkpoint
        payload["representation_kind"] = (
            "forecast-query row states from TabICLForecaster preprocessing + TabICL ICL blocks"
        )
        payload["forecast_query_policy"] = "synthetic_next_step"
        payload["layer_layout"] = (
            "layer 0 is the forecast-query row after row interaction; "
            "layers 1..N are ICL block outputs for that same query row"
        )
        payload["probe_train_sample_cap"] = int(self.probe_train_sample_cap)
        payload["probe_val_sample_cap"] = int(self.probe_val_sample_cap)
        if self._representation_dim is not None:
            payload["representation_dim"] = int(self._representation_dim)
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
            raise RuntimeError("TabICLForecaster sampling frequency is not prepared yet")
        return int(self._sampling_frequency_hz)

    def _tabular_device(self) -> str:
        return "cuda" if self._runtime_device.type == "cuda" and torch.cuda.is_available() else "cpu"

    def _make_forecaster(self):
        return self._tabicl_forecaster_class(
            max_context_length=int(self.max_context_length),
            tabicl_config={
                "n_estimators": int(self.n_estimators),
                "checkpoint_version": self.checkpoint,
                "device": self._tabular_device(),
                "batch_size": int(self.batch_size),
                "verbose": False,
            },
        )

    def _build_feature_tables(self, waveforms: np.ndarray):
        from tabicl.forecast.preprocessing import build_horizon
        from tabicl.forecast.ts_dataframe import TimeSeriesDataFrame

        forecaster = self._make_forecaster()
        context_df = build_context_dataframe(
            waveforms,
            sampling_frequency_hz=self._require_sampling_frequency_hz(),
        )
        context_tsdf = TimeSeriesDataFrame.from_data_frame(context_df)
        context_tsdf = forecaster._prepare_context(context_tsdf)
        future_tsdf = build_horizon(
            context_tsdf,
            prediction_length=int(self.prediction_length),
        )
        future_tsdf = forecaster._prepare_future(future_tsdf)
        context_tsdf, future_tsdf = forecaster._align_covariates(context_tsdf, future_tsdf)
        return forecaster.feature_transformer.transform(context_tsdf, future_tsdf)

    def _extract_series_layers(
        self,
        *,
        train_item,
        test_item,
        seed: int,
    ) -> dict[int, np.ndarray]:
        from tabicl import TabICLRegressor

        regressor = TabICLRegressor(
            n_estimators=int(self.n_estimators),
            checkpoint_version=self.checkpoint,
            device=self._tabular_device(),
            batch_size=int(self.batch_size),
            kv_cache=False,
            random_state=int(seed),
            verbose=False,
        )
        train_x = train_item.drop(columns=["target"])
        train_y = train_item["target"].to_numpy(dtype=np.float32, copy=True)
        test_x = test_item.drop(columns=["target"])
        regressor.fit(train_x, train_y)

        test_x_encoded = regressor.X_encoder_.transform(test_x)
        layer_batches: dict[int, list[torch.Tensor]] = {layer: [] for layer in self.available_layers}
        total_blocks: int | None = None

        for norm_method, (x_variants, y_variants) in regressor.ensemble_generator_.transform(
            test_x_encoded,
            mode="both",
        ).items():
            del norm_method
            x_batch = torch.from_numpy(x_variants).float().to(regressor.device_)
            y_batch = torch.from_numpy(y_variants).float().to(regressor.device_)
            train_size = int(y_batch.shape[1])
            raw_model = regressor.model_

            with torch.no_grad():
                row_states = raw_model.row_interactor(
                    raw_model.col_embedder(
                        x_batch,
                        y_train=y_batch,
                        mgr_config=regressor.inference_config_.COL_CONFIG,
                    ),
                    mgr_config=regressor.inference_config_.ROW_CONFIG,
                )
                icl_blocks = raw_model.icl_predictor.tf_icl.blocks
                total_blocks = int(len(icl_blocks))
                layer_batches[0].append(row_states[:, -1].float().cpu())

                hidden_states = raw_model.icl_predictor.prepare_repr_cache(row_states.clone(), y_batch)
                for layer_index, block in enumerate(icl_blocks, start=1):
                    hidden_states = block(
                        q=hidden_states,
                        train_size=train_size,
                        rope=raw_model.icl_predictor.tf_icl.rope,
                    )
                    if (
                        layer_index == total_blocks
                        and raw_model.icl_predictor.norm_first
                    ):
                        query_states = raw_model.icl_predictor.ln(hidden_states)[:, -1]
                    else:
                        query_states = hidden_states[:, -1]
                    layer_batches[layer_index].append(query_states.float().cpu())

        if total_blocks is None:
            raise RuntimeError("TabICLForecaster did not produce any ensemble variants")

        actual_num_layers = int(total_blocks) + 1
        if actual_num_layers != self._provisional_num_layers:
            raise RuntimeError(
                "TabICLForecaster checkpoint layer count differs from the provisional adapter contract: "
                f"expected {self._provisional_num_layers} layers from the published default TabICL "
                f"constructor, got {actual_num_layers} after fitting the regressor checkpoint. "
                "Update the adapter registry and tests to keep layer numbering honest."
            )
        self._verified_num_layers = actual_num_layers

        series_layers: dict[int, np.ndarray] = {}
        for layer, tensors in layer_batches.items():
            if not tensors:
                raise RuntimeError(f"Missing TabICLForecaster representations for layer {layer}")
            stacked = torch.cat(tensors, dim=0)
            averaged = stacked.mean(dim=0).numpy().astype(np.float32, copy=False)
            series_layers[int(layer)] = averaged
        return series_layers

    def _encode_waveforms(
        self,
        waveforms: np.ndarray,
        *,
        split_name: str,
    ) -> dict[int, torch.Tensor]:
        train_tsdf, test_tsdf = self._build_feature_tables(waveforms)
        total_series = int(waveforms.shape[0])
        features_by_layer: dict[int, np.ndarray] = {}
        log_every = max(1, (total_series + 3) // 4)

        for sample_index in range(total_series):
            if (
                sample_index == 0
                or sample_index + 1 == total_series
                or (sample_index + 1) % log_every == 0
            ):
                print(
                    f"[TabICLForecaster] {split_name} embedding progress: "
                    f"series {sample_index + 1}/{total_series}",
                    file=sys.stderr,
                    flush=True,
                )

            layer_vectors = self._extract_series_layers(
                train_item=train_tsdf.loc[sample_index].copy(),
                test_item=test_tsdf.loc[sample_index].copy(),
                seed=self.fit_seed + sample_index,
            )
            if not features_by_layer:
                for layer, vector in layer_vectors.items():
                    features_by_layer[layer] = np.empty(
                        (total_series, vector.shape[0]),
                        dtype=np.float32,
                    )
                self._representation_dim = int(next(iter(layer_vectors.values())).shape[0])
            for layer, vector in layer_vectors.items():
                features_by_layer[layer][sample_index] = vector

        if not features_by_layer:
            raise RuntimeError("TabICLForecaster did not produce any embeddings")
        return {
            int(layer): torch.from_numpy(features)
            for layer, features in sorted(features_by_layer.items())
        }

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
        unknown_layers = [layer for layer in requested_layers if layer not in self.available_layers]
        if unknown_layers:
            raise ValueError(
                f"TabICLForecaster does not expose layers {unknown_layers}; "
                f"available_layers={self.available_layers}"
            )
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
        raise RuntimeError(
            "TabICLForecaster uses cached split features; call make_representation_fn()"
        )
