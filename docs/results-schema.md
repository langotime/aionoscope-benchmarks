# Results Schema

The canonical artifact for one benchmark run is:

- `results/models/<model-slug>__num_enabled_<k>.json`

`results/dashboard.html` must be able to initialize from those files without any hard-coded manifest.

Official standalone manifold artifacts live under `results/manifolds/`. They are
canonical for the manifold evaluation workflow, but separate from the leaderboard
benchmark result artifacts.
They use `schema_version = "manifold_result_v0"` plus a dedicated static
viewer for inspection, and `results/dashboard.html` must not discover them as
leaderboard inputs. The manifold viewer reads each layer's `plot_data_json` in
the browser and renders ECharts charts; metric computation remains in Python.

Calibration baselines write to `results/models/`. They must set
`model.type = "baseline"` and use synthetic layer `0`; foundational model runs
keep `model.type = "foundational"`.

## Required Top-Level Payload Sections

- `model`
- `dataset`
- `probe_config`
- `runtime`
- `results.categorical`
- `results.dense`
- `results.shared`
- `results.summary`

## Required Identity Fields

- `model.slug`
- `model.family`
- `model.checkpoint_name`
- `model.architecture.backbone`
- `model.training.paradigm`
- `dataset.benchmark_family`
- `dataset.benchmark_version`
- `dataset.num_enabled`

Those fields are the minimum needed for discovery, dashboard grouping, and run identity.

## Probe Learning Rates

`probe_config.learning_rate` and `probe_config.final_learning_rate` are base
optimizer rates. Current benchmark runs also include
`probe_config.learning_rate_scaling`, which records the normalized
feature-dimension scaling policy:

```python
effective_lr = base_lr * min(1.0, (1024.0 / feature_dim) ** 3)
effective_lr = max(effective_lr, 1.0e-3)
```

Each layer payload under `results.categorical` and `results.dense` includes a
`learning_rate` object with the base and effective rates used for that probe
head/layer. Older checked-in artifacts may not contain these fields.

## Corpus Rules

- `results/models/` must contain one coherent benchmark family/version at a time.
- Historical benchmark results belong in git history, not mixed into the active discovery path.
- `results/models/list.txt` is a deployment artifact when a static host needs an explicit manifest. Do not keep it in the dev tree or commit it.

## Maintenance

Benchmark result writes only emit JSON artifacts. If deployment needs `results/models/list.txt`, generate it as part of the website publish step rather than in the development checkout.
