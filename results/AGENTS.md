# Dashboard Architecture Guide

This folder contains several different things:

- `dashboard.html`: the checked-in dashboard shell for `models/*.json`.
- `dashboard-v2.html`: the checked-in dashboard shell for `models-v2/*.json`.
  It intentionally mirrors `dashboard.html`; when changing shared dashboard UI,
  discovery, charting, or filtering behavior, update both files unless the task
  explicitly calls for a one-version divergence.
- `manifolds.html`: the checked-in Cloudflare Pages shell for the standalone
  manifold viewer; its generated `manifest.json` and large JSON data live in
  Cloudflare R2.
- `_redirects`: Cloudflare Pages redirects for legacy article URLs that moved to
  `https://blog.langotime.ai/`. Do not restore checked-in `about-*.html` article
  pages here.
- `models/*.json` and `models-v2/*.json`: generated benchmark artifacts consumed
  by the dashboard shells.
- `assets/`: shared styling for the dashboard and manifold viewer — the vendored
  Langotime Design System (`assets/langotime/`), dashboard/viewer stylesheets
  (`assets/css/`), and the ECharts theme (`assets/js/`). The deploy unit is now
  "HTML + `_redirects` + `assets/`": this directory must be deployed/uploaded
  alongside the pages so the relative `assets/...` links resolve.

## Critical decisions

### 0. Manifold viewer deployment is split across Pages and R2

`results/manifolds.html` is intentionally checked in so the Git-backed
Cloudflare Pages project deploys the viewer shell. It should be edited as a
normal static shell; data updates should regenerate `results/manifolds/manifest.json`
instead of regenerating this HTML. The generated `results/manifolds/` directory
is ignored and must not be committed. It is the local source for bulk-uploaded
R2 objects under the versioned
`manifolds/v.../` prefix. Read `../docs/manifold-r2-pages.md` before changing
manifold hosting, cache rules, CORS, upload paths, or `MANIFOLD_DATA_BASE_URL`.
When uploading R2 data, `results/manifolds/manifest.json` must be uploaded with
the manifest cache lifetime documented there, while the large per-target and
plot JSON payloads use the documented browser/shared-cache TTLs. Do not restore
the old one-year `immutable` policy for these JSON objects.

### 1. Static, framework-free pages with a shared design system

Every page in `results/` is plain static HTML with inline JavaScript and
**external** CSS. Styling lives under `results/assets/`, not in inline `<style>`
blocks (this supersedes the original inline-CSS rule):

- `assets/langotime/` — the vendored Langotime Design System (`styles.css` +
  `tokens/`): the single source of brand tokens (color, type, spacing, motifs)
  and webfonts. Treat as read-only vendored input.
- `assets/css/` — per-surface stylesheets. `dashboard.css` is shared by
  `dashboard.html` and `dashboard-v2.html`. `manifolds.css` is for the viewer.
  Article CSS moved to `blog-langotime-ai`; do not restore `about-*.css` here.
- `assets/js/chart-theme.js` — the `langotime` ECharts theme; pages init charts
  with `echarts.init(el, "langotime", …)`.

Still holds: no framework, no bundler, no build step, no split frontend app, and
JavaScript stays inline per page — each page is still easy to open from a plain
static file server. Link order per page: the design system first, then the page's
own stylesheet.

When restyling, work through tokens, not literals. Chart colors are not CSS:
ECharts series colors live in each page's inline `option` objects. Defaults come
from the `langotime` theme in `chart-theme.js`; any explicit per-series color
must be a Langotime palette value.

### 2. External libraries stay minimal and explicit

The dashboard shells currently depend on exactly two frontend libraries loaded from CDNs:

- Apache ECharts for all chart rendering.
- Font Awesome for small UI icons such as sidebar open/close controls.

Do not introduce extra UI, state-management, or charting libraries unless there is a strong reason and the change is explicitly requested.

If dependency versions change:

- pin explicit versions in both `dashboard.html` and `dashboard-v2.html`;
- keep Font Awesome icon class compatibility in mind;
- keep the dependency surface small.

### 3. Client-side JSON processing only

The browser reads `results/models/*.json` or `results/models-v2/*.json` directly
and computes presentation-layer summaries in JavaScript.

- Aggregation for charts happens client-side.
- Filtering and selection happen client-side.
- Layer selection behavior happens client-side.

Do not move benchmark generation, probe execution, or model logic into browser code. Python remains the source of truth for benchmark computation.

### 4. The dashboard is a reader, not a source of truth

`results/models/*.json` and `results/models-v2/*.json` are the canonical
machine-readable outputs. The dashboard shells must adapt to that schema, not
define a competing one.

- Do not hand-edit model JSON files as part of normal dashboard work.
- Do not invent browser-only result fields as a substitute for Python-side schema changes.
- If the Python result schema changes, update `results/dashboard.html`,
  `results/dashboard-v2.html`, `README.md`, `DOCUMENTATION.md`, and
  `ARCHITECTURE.md` in the same task.

### 5. Static-server delivery is intentional

The dashboard shells are meant to be served by a simple static HTTP server.

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
- run a lightweight smoke check that the page can still initialize from the
  relevant generated JSON directory;
- if chart behavior or JSON consumption changed, verify the dashboard still reads real result files correctly.

## Practical editing guidance

- Treat `results/dashboard.html` and `results/dashboard-v2.html` as paired
  dashboard shells; keep them synchronized unless a task explicitly splits
  their behavior.
- Treat `results/models/*.json` and `results/models-v2/*.json` as generated artifacts.
- Prefer small, direct changes over frontend rewrites.
- Preserve clear error messages when CDN assets or JSON files fail to load.
