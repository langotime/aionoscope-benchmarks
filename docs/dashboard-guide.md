# Dashboard Guide

The dashboard is a static site rooted at `results/`.

## Discovery Contract

- site root: `results/`
- dashboard page: `results/dashboard.html`
- result files: `results/models/*.json`

When published with `results/` as the site root, the page first tries `/models/list.txt` and only then falls back to directory listing. `results/models/list.txt` is deployment-only and should not exist in the dev checkout.

`results/dashboard-v2.html` is an optional comparison snapshot for the copied
`results/models-v2/*.json` corpus. It uses the same static dashboard code, but
discovers files through `/models-v2/list.txt` or the `results/models-v2/`
directory listing so `dashboard.html` can keep showing the git-restored
`results/models/` corpus.

## What The Dashboard Is Allowed To Do

- load result JSON files
- derive view-specific summaries from already serialized JSON
- filter or group runs in browser state
- filter `model.type` so baseline artifacts can stay hidden from the model view by default

## What The Dashboard Must Not Do

- run benchmark computation
- regenerate datasets
- train probes
- infer missing source-of-truth metadata that should have been serialized in JSON

## Smoke Test

Run:

```bash
uv run python -m aionoscope_benchmarks.dashboard_smoke
```

The local smoke harness verifies that the dashboard still works from a dev-style `results/` tree without `results/models/list.txt`, relying on the directory-listing fallback.
