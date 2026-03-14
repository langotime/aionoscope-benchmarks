from __future__ import annotations

import numpy as np
import sys
import torch

from .base import FrozenTimeSeriesAdapter


class TabICLAdapter(FrozenTimeSeriesAdapter):
    model_name = "TabICL"
    model_slug = "TabICL"
    source = "https://github.com/soda-inria/tabicl"
    checkpoint = "tabicl-classifier-v1-20250208.ckpt"
    import_path = "tabicl"
    env_name = "tabular"
    default_encode_batch_size = 4096
    use_bfloat16_amp = False

    reduced_feature_length = 128
    max_fit_samples_per_label = 2_048
    n_estimators = 1
    predict_batch_size = 512
    fit_seed = 0
    probe_train_sample_cap = 2_048
    probe_val_sample_cap = 2_048

    def __init__(self) -> None:
        super().__init__()
        self._split_feature_cache: dict[str, dict[int, torch.Tensor]] = {}
        self._class_names: list[str] = []
        self._classifiers: list[object] = []
        self.benchmark_sequence_length = int(self.reduced_feature_length)
        self.benchmark_sequence_length_source = "tabular_fallback_feature_length"
        self.probe_train_split: dict[str, torch.Tensor] | None = None
        self.probe_val_split: dict[str, torch.Tensor] | None = None

    @property
    def available_layers(self) -> tuple[int, ...]:
        return (0,)

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["representation_kind"] = "one-vs-rest positive-class probabilities"
        payload["feature_length"] = int(self.reduced_feature_length)
        payload["fit_samples_per_label_cap"] = int(self.max_fit_samples_per_label)
        payload["n_estimators"] = int(self.n_estimators)
        payload["checkpoint_version"] = self.checkpoint
        payload["paper_fallback_note"] = (
            "TabICL is a supervised tabular classifier rather than a frozen layerwise time-series encoder; "
            "this adapter uses 14 binary one-vs-rest classifiers on exact-length tabularized waveforms."
        )
        payload["probe_train_sample_cap"] = int(self.probe_train_sample_cap)
        payload["probe_val_sample_cap"] = int(self.probe_val_sample_cap)
        if self.probe_train_split is not None:
            payload["probe_train_sample_count"] = int(self.probe_train_split["x"].size(0))
        if self.probe_val_split is not None:
            payload["probe_val_sample_count"] = int(self.probe_val_split["x"].size(0))
        return payload

    def _reduce_inputs(self, x: torch.Tensor) -> np.ndarray:
        self.validate_benchmark_input(x, channels=1)
        return np.ascontiguousarray(x[:, 0, :].to(dtype=torch.float32).cpu().numpy(), dtype=np.float32)

    def _sample_fit_indices(self, y_binary: np.ndarray, *, seed: int) -> np.ndarray:
        positives = np.flatnonzero(y_binary == 1)
        negatives = np.flatnonzero(y_binary == 0)
        if positives.size == 0 or negatives.size == 0:
            raise ValueError("TabICL one-vs-rest fit needs both positive and negative samples")

        rng = np.random.default_rng(seed)
        per_class_cap = max(1, self.max_fit_samples_per_label // 2)
        pos_take = min(per_class_cap, positives.size)
        neg_take = min(per_class_cap, negatives.size)
        pos_pick = rng.choice(positives, size=pos_take, replace=False)
        neg_pick = rng.choice(negatives, size=neg_take, replace=False)
        picked = np.concatenate([pos_pick, neg_pick], axis=0)
        rng.shuffle(picked)
        return picked

    def _sample_probe_indices(
        self,
        size: int,
        *,
        sample_cap: int,
        seed: int,
    ) -> np.ndarray:
        if sample_cap >= size:
            return np.arange(size, dtype=np.int64)
        rng = np.random.default_rng(seed)
        indices = rng.choice(size, size=sample_cap, replace=False)
        indices.sort()
        return indices.astype(np.int64, copy=False)

    def _positive_proba(self, classifier, x: np.ndarray) -> np.ndarray:
        positive_index = int(np.flatnonzero(np.asarray(classifier.classes_) == 1)[0])
        probs: list[np.ndarray] = []
        for start in range(0, x.shape[0], self.predict_batch_size):
            stop = min(start + self.predict_batch_size, x.shape[0])
            batch = x[start:stop]
            prob = classifier.predict_proba(batch)[:, positive_index]
            probs.append(np.asarray(prob, dtype=np.float32))
        return np.concatenate(probs, axis=0)

    def prepare(
        self,
        *,
        manifest: dict[str, object],
        train_split: dict[str, torch.Tensor],
        val_split: dict[str, torch.Tensor],
    ) -> None:
        from tabicl import TabICLClassifier

        self._class_names = [str(name) for name in manifest["class_names"]]
        train_x = self._reduce_inputs(train_split["x"])
        train_y = train_split["y_cls"].to(dtype=torch.int64).cpu().numpy()
        train_probe_indices = self._sample_probe_indices(
            train_x.shape[0],
            sample_cap=self.probe_train_sample_cap,
            seed=self.fit_seed + 10_000,
        )
        probe_train_x = train_x[train_probe_indices]

        train_features = np.empty((probe_train_x.shape[0], train_y.shape[1]), dtype=np.float32)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        classifiers: list[object] = []
        total_labels = len(self._class_names)
        log_every = max(1, (total_labels + 3) // 4)

        for class_index, _class_name in enumerate(self._class_names):
            if (
                class_index == 0
                or class_index + 1 == total_labels
                or (class_index + 1) % log_every == 0
            ):
                print(
                    f"[TabICL] adapter prepare progress: label {class_index + 1}/{total_labels}",
                    file=sys.stderr,
                    flush=True,
                )
            y_binary = train_y[:, class_index]
            fit_indices = self._sample_fit_indices(
                y_binary,
                seed=self.fit_seed + class_index,
            )
            classifier = TabICLClassifier(
                n_estimators=self.n_estimators,
                checkpoint_version=self.checkpoint,
                device=device,
                batch_size=8,
                kv_cache=False,
                random_state=self.fit_seed + class_index,
                verbose=False,
            )
            classifier.fit(train_x[fit_indices], y_binary[fit_indices])
            train_features[:, class_index] = self._positive_proba(classifier, probe_train_x)
            classifiers.append(classifier)

        self._classifiers = classifiers
        self._split_feature_cache = {
            "train": {0: torch.from_numpy(train_features)},
        }
        self.probe_train_split = {
            key: value[train_probe_indices] for key, value in train_split.items()
        }
        self.update_probe_val_split(val_split=val_split)

    def update_probe_val_split(
        self,
        *,
        val_split: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self._classifiers:
            raise RuntimeError("TabICL classifiers are not prepared yet")
        val_x = self._reduce_inputs(val_split["x"])
        val_probe_indices = self._sample_probe_indices(
            val_x.shape[0],
            sample_cap=self.probe_val_sample_cap,
            seed=self.fit_seed + 20_000,
        )
        probe_val_x = val_x[val_probe_indices]
        val_features = np.empty((probe_val_x.shape[0], len(self._class_names)), dtype=np.float32)
        for class_index, classifier in enumerate(self._classifiers):
            val_features[:, class_index] = self._positive_proba(classifier, probe_val_x)
        self._split_feature_cache["val"] = {0: torch.from_numpy(val_features)}
        self.probe_val_split = {
            key: value[val_probe_indices] for key, value in val_split.items()
        }
        return self.probe_val_split

    def make_representation_fn(
        self,
        *,
        layers: tuple[int, ...],
        split: str = "val",
    ):
        requested_layers = tuple(int(layer) for layer in layers)
        if requested_layers != (0,):
            raise ValueError(f"TabICL only exposes layer 0, got {requested_layers}")
        split_cache = self._split_feature_cache.get(split)
        if split_cache is None:
            raise RuntimeError("TabICL features are not prepared yet")

        offset = 0
        total_size = int(split_cache[0].size(0))

        def _representation_fn(x: torch.Tensor) -> dict[int, torch.Tensor]:
            nonlocal offset
            batch_size = int(x.size(0))
            start = offset
            stop = start + batch_size
            if stop > total_size:
                raise ValueError(
                    f"TabICL cached features exhausted for split={split}: "
                    f"requested stop={stop} total={total_size}"
                )
            offset = stop
            return {0: split_cache[0][start:stop]}

        return _representation_fn

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        raise RuntimeError("TabICL uses cached split features; call make_representation_fn()")
