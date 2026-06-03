# Dashboard Architecture Guide

This folder contains two very different things:

- `dashboard.html`: the only hand-maintained dashboard source file.
- `manifolds.html`: the checked-in Cloudflare Pages shell for the standalone
  manifold viewer; its large generated JSON data lives in Cloudflare R2.
- `models/*.json`: generated benchmark artifacts consumed by the dashboard.

## Critical decisions

### 0. Manifold viewer deployment is split across Pages and R2

`results/manifolds.html` is intentionally checked in so the Git-backed
Cloudflare Pages project deploys the viewer shell. The generated
`results/manifolds/` directory is ignored and must not be committed. It is the
local source for bulk-uploaded R2 objects under the versioned
`manifolds/v.../` prefix. Read `../docs/manifold-r2-pages.md` before changing
manifold hosting, cache rules, CORS, upload paths, or `MANIFOLD_DATA_BASE_URL`.

### 1. Single-file static dashboard

Keep the dashboard as one self-contained HTML document with inline CSS and inline JavaScript in `results/dashboard.html`.

- No framework.
- No bundler.
- No build step.
- No split frontend app structure.

The dashboard should remain easy to open from a plain static file server and easy to audit in one file.

### 2. External libraries stay minimal and explicit

The dashboard currently depends on exactly two frontend libraries loaded from CDNs:

- Apache ECharts for all chart rendering.
- Font Awesome for small UI icons such as sidebar open/close controls.

Do not introduce extra UI, state-management, or charting libraries unless there is a strong reason and the change is explicitly requested.

If dependency versions change:

- pin explicit versions in `dashboard.html`;
- keep Font Awesome icon class compatibility in mind;
- keep the dependency surface small.

### 3. Client-side JSON processing only

The browser reads `results/models/*.json` directly and computes presentation-layer summaries in JavaScript.

- Aggregation for charts happens client-side.
- Filtering and selection happen client-side.
- Layer selection behavior happens client-side.

Do not move benchmark generation, probe execution, or model logic into browser code. Python remains the source of truth for benchmark computation.

### 4. The dashboard is a reader, not a source of truth

`results/models/*.json` is the canonical machine-readable output. The dashboard must adapt to that schema, not define a competing one.

- Do not hand-edit model JSON files as part of normal dashboard work.
- Do not invent browser-only result fields as a substitute for Python-side schema changes.
- If the Python result schema changes, update `results/dashboard.html`, `README.md`, `DOCUMENTATION.md`, and `ARCHITECTURE.md` in the same task.

### 5. Static-server delivery is intentional

The dashboard is meant to be served by a simple static HTTP server.

- Keep browser fetches relative to `results/`.
- Preserve the current model discovery model: try directory listing first, then fall back to the built-in manifest.
- Do not require a custom backend or API server for normal dashboard use unless explicitly requested.

### 6. Apache ECharts is the charting layer

Use Apache ECharts for all existing dashboard visualizations.

- Keep radar and line-chart logic in ECharts option builders.
- Prefer extending the current chart-building helpers over adding parallel rendering systems.
- Do not replace ECharts with another charting library without an explicit request.

### 7. Font Awesome is only for lightweight icons

Font Awesome is used for small interface affordances, not for broader UI architecture.

- Keep icon usage simple and local.
- Prefer stable icon classes already supported by the pinned Font Awesome version.
- If legacy `fa-angle-double-*` names are used, keep the matching v4 shim aligned with the pinned Font Awesome version.

### 8. Keep browser-side behavior presentation-focused

Browser code may:

- choose visible models;
- choose the active best-layer selector;
- derive chart-ready series from stored JSON metrics;
- manage transient UI state such as sidebar collapse.

Browser code should not:

- change benchmark semantics;
- silently reinterpret missing data;
- hide schema problems with broad fallback behavior;
- duplicate Python-side benchmark contracts in a divergent way.

### 9. Validation expectations

After changing the dashboard:

- verify the HTML still parses;
- run a lightweight smoke check that the page can still initialize from `results/models/*.json`;
- if chart behavior or JSON consumption changed, verify the dashboard still reads real result files correctly.

## Practical editing guidance

- Treat `results/dashboard.html` as the canonical implementation.
- Treat `results/models/*.json` as generated artifacts.
- Prefer small, direct changes over frontend rewrites.
- Preserve clear error messages when CDN assets or JSON files fail to load.
