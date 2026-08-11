# Dashboard Guide

The dashboard is part of a static site rooted at `results/`.

## Site Map

`results/index.html` is the site index at `https://aionoscope.langotime.ai/`. It
is a small landing page that fans out to the three surfaces, and it owns that
navigation: each surface links back to the index rather than to its siblings.

- `results/index.html` — index
- `results/dashboard-v2.html` — Aionoscope Linear Probe Dashboard v2, the current
  surface (`models-v2/*.json`)
- `results/dashboard.html` — Aionoscope Linear Probe Dashboard, the earlier
  surface (`models/*.json`)
- `results/manifolds.html` — Aionoscope Manifold Viewer

Both dashboard surfaces are named after the instrument and deliberately avoid the
word "benchmark": the index headline contrasts a benchmark with what Aionoscope
shows. The internal vocabulary is unchanged — the repository, the sweeps, and the
`dataset.benchmark_family` / `benchmark_version` result fields still say
benchmark.

Every page loads the shared Langotime Design System from
`https://langotime.ai/design-system/v1/styles.css`. It is never vendored into
this repository, and per-surface CSS must use design-system tokens instead of
re-declaring brand literals.

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
