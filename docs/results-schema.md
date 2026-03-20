# Results Schema

The canonical artifact for one benchmark run is:

- `results/models/<model-slug>__num_enabled_<k>.json`

`results/dashboard.html` must be able to initialize from those files without any hard-coded manifest.

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

## Corpus Rules

- `results/models/` must contain one coherent benchmark family/version at a time.
- Historical benchmark results belong in git history, not mixed into the active discovery path.
- `results/models/list.txt` is a deployment artifact when a static host needs an explicit manifest. Do not keep it in the dev tree or commit it.

## Maintenance

Benchmark result writes only emit JSON artifacts. If deployment needs `results/models/list.txt`, generate it as part of the website publish step rather than in the development checkout.
