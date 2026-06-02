from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
from typing import Any


def _relpath(path: str | None, *, base: Path) -> str | None:
    if not path:
        return None
    raw = Path(path)
    if not raw.is_absolute():
        raw = (Path.cwd() / raw).resolve()
    try:
        return os.path.relpath(raw, base)
    except ValueError:
        return str(raw)


def collect_viewer_records(*, artifact_root: Path, viewer_path: Path) -> list[dict[str, Any]]:
    records = []
    base = viewer_path.parent.resolve()
    for metrics_path in sorted(artifact_root.glob("**/metrics.json")):
        # Layout is <run>/<model>/<target>/metrics.json, so parents[2] is the
        # run directory. The root viewer aggregates several runs; carrying the
        # run name lets the page keep one run's records separate from another's.
        run_name = metrics_path.parents[2].name if len(metrics_path.parents) >= 3 else ""
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        model = payload.get("model", {})
        target = payload.get("target", {})
        train_slice_manifest = payload.get("train_slice_manifest", {})
        sweep = train_slice_manifest.get("sweep", {}) if isinstance(train_slice_manifest, dict) else {}
        summary = payload.get("summary", {})
        by_layer = payload.get("by_layer", {})
        visualizations = payload.get("visualizations", {})
        for layer_key, metrics in by_layer.items():
            viz = visualizations.get(str(layer_key), {})
            target_name = str(target.get("target_name", ""))
            target_label = target_name
            if isinstance(sweep, dict):
                range_policy = sweep.get("range_policy")
                grid_mode = sweep.get("grid_mode")
                if (
                    isinstance(range_policy, str)
                    and range_policy.startswith("wide_abs_")
                    and isinstance(grid_mode, str)
                ):
                    target_label = f"{target_name} [{range_policy}, {grid_mode}]"
            records.append(
                {
                    "run": run_name,
                    "model": str(model.get("name", model.get("slug", ""))),
                    "model_slug": str(model.get("slug", "")),
                    "target": target_label,
                    "target_name": target_name,
                    "sweep": sweep if isinstance(sweep, dict) else {},
                    "geometry": str(target.get("geometry", "")),
                    "layer": str(layer_key),
                    "metrics": metrics,
                    "summary": summary,
                    "paths": {
                        key: _relpath(value, base=base)
                        for key, value in viz.items()
                        if key == "plot_data_json"
                        if isinstance(value, str)
                    },
                    "metrics_json": _relpath(str(metrics_path), base=base),
                }
            )
    return records


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _json_for_script(payload: Any) -> str:
    return json.dumps(_json_safe(payload), separators=(",", ":"), allow_nan=False).replace(
        "</",
        "<\\/",
    )


def build_viewer(*, artifact_root: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = collect_viewer_records(artifact_root=artifact_root, viewer_path=out_path)
    records_json = _json_for_script(records)
    title = "Aionoscope Manifold Calibration Viewer"
    html_text = _VIEWER_TEMPLATE.replace("__TITLE__", html.escape(title)).replace(
        "__RECORDS_JSON__",
        records_json,
    )
    out_path.write_text(html_text, encoding="utf-8")


_VIEWER_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/echarts-gl@2/dist/echarts-gl.min.js" onerror="window.__glLoadFailed=true"></script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f8;
      --panel: #ffffff;
      --panel-border: #d8dee8;
      --text: #152033;
      --muted: #5c6a82;
      --accent: #0f766e;
      --accent-soft: #dff5f2;
      --blue: #2563eb;
      --red: #dc2626;
      --grid: #e2e8f0;
      font-family: "IBM Plex Sans", "Segoe UI", system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; }
    body {
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 28%),
        linear-gradient(180deg, #eef4f6 0%, #f5f6f8 22%, #f5f6f8 100%);
    }
    a { color: #155ca6; text-decoration: none; }
    header {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
      padding: 20px 24px;
      border-bottom: 1px solid var(--panel-border);
      background: rgba(255, 255, 255, 0.92);
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(10px);
    }
    h1 { font-size: 20px; line-height: 1.15; margin: 0 0 5px; font-weight: 720; }
    .header-copy { margin: 0; color: var(--muted); font-size: 13px; max-width: 820px; }
    main { padding: 18px 24px 36px; }
    .controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    label { display: grid; gap: 5px; font-size: 12px; color: var(--muted); font-weight: 650; }
    select {
      min-width: 0;
      padding: 8px 10px;
      border: 1px solid #c8d0d9;
      background: #fff;
      border-radius: 7px;
      color: var(--text);
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(300px, 390px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }
    .summary-panel { padding: 15px; position: sticky; top: 96px; }
    .record-title { font-size: 15px; line-height: 1.25; margin: 0 0 12px; }
    .record-meta {
      display: grid;
      gap: 7px;
      padding: 10px 0 12px;
      border-top: 1px solid #edf1f5;
      border-bottom: 1px solid #edf1f5;
      color: var(--muted);
      font-size: 12px;
    }
    .meta-row { display: flex; justify-content: space-between; gap: 12px; }
    .meta-row span:last-child { color: var(--text); text-align: right; }
    .metrics {
      display: grid;
      gap: 8px;
      margin-top: 13px;
    }
    .metric-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 7px 0;
      border-bottom: 1px solid #edf1f5;
      font-size: 13px;
    }
    .metric-row:last-child { border-bottom: 0; }
    .metric-label {
      display: inline-flex;
      min-width: 0;
      gap: 6px;
      align-items: center;
      color: #334155;
    }
    .metric-name {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .metric-info {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 16px;
      height: 16px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #0f766e;
      font-size: 11px;
      font-weight: 750;
      flex: 0 0 auto;
    }
    .metric-value {
      font-variant-numeric: tabular-nums;
      color: #0f172a;
      font-weight: 650;
      text-align: right;
    }
    .links { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; font-size: 12px; }
    .layer-metrics {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 14px;
    }
    .layer-metric {
      border: 1px solid #edf1f5;
      border-radius: 7px;
      padding: 8px 8px 4px;
      background: #fbfdff;
      min-width: 0;
    }
    .layer-metric-head { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; font-size: 12px; }
    .layer-metric-head .metric-name { font-weight: 650; color: #334155; }
    .layer-metric-value {
      margin-left: auto;
      font-variant-numeric: tabular-nums;
      font-weight: 650;
      color: #0f172a;
    }
    .chart-mini { width: 100%; height: 168px; }
    .metric-info { cursor: help; }
    #tip {
      position: fixed;
      z-index: 60;
      max-width: 320px;
      padding: 10px 12px;
      background: #0f172a;
      color: #f1f5f9;
      font-size: 12px;
      line-height: 1.5;
      border-radius: 8px;
      box-shadow: 0 14px 34px rgba(15, 23, 42, 0.34);
      white-space: pre-line;
      pointer-events: none;
      opacity: 0;
      transform: translateY(2px);
      transition: opacity 0.09s ease, transform 0.09s ease;
    }
    #tip.visible { opacity: 1; transform: translateY(0); }
    .charts { display: grid; gap: 16px; min-width: 0; }
    .chart-card { padding: 14px 14px 10px; overflow: hidden; }
    .chart-header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px 18px;
      align-items: start;
      margin-bottom: 8px;
    }
    .chart-header h3 { margin: 0; font-size: 15px; line-height: 1.2; }
    .chart-copy {
      grid-column: 1 / -1;
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .chart-note {
      margin: 0;
      color: #64748b;
      font-size: 12px;
      text-align: right;
    }
    .chart-head-right { display: flex; align-items: center; gap: 10px; justify-content: flex-end; }
    .seg-toggle { display: inline-flex; border: 1px solid #c8d0d9; border-radius: 7px; overflow: hidden; flex: 0 0 auto; }
    .seg-toggle button {
      border: 0;
      background: #fff;
      color: #334155;
      font: inherit;
      font-size: 12px;
      font-weight: 650;
      padding: 4px 11px;
      cursor: pointer;
    }
    .seg-toggle button + button { border-left: 1px solid #c8d0d9; }
    .seg-toggle button.active { background: var(--accent); color: #fff; }
    .collapsible > summary {
      cursor: pointer;
      list-style: none;
      user-select: none;
    }
    .collapsible > summary::-webkit-details-marker { display: none; }
    .collapsible > summary h3 { display: inline-flex; align-items: center; gap: 9px; }
    .chev {
      flex: 0 0 auto;
      width: 0;
      height: 0;
      border-left: 5px solid transparent;
      border-right: 5px solid transparent;
      border-top: 6px solid #475569;
      transform: rotate(-90deg);
      transition: transform 0.15s ease;
    }
    .collapsible[open] > summary .chev { transform: rotate(0deg); }
    .collapsible:not([open]) > summary { margin-bottom: 0; }
    .chart {
      width: 100%;
      height: 430px;
      border-top: 1px solid #edf1f5;
    }
    #heatmap-chart { height: 410px; }
    .status, .empty {
      padding: 24px;
      color: var(--muted);
      font-size: 13px;
    }
    .warning {
      color: #9f1239;
      background: #fff1f1;
      border: 1px solid #f1b5b5;
      border-radius: 8px;
    }
    @media (max-width: 1040px) {
      header { position: static; }
      .controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .layout { grid-template-columns: 1fr; }
      .summary-panel { position: static; }
    }
    @media (max-width: 620px) {
      main, header { padding-left: 14px; padding-right: 14px; }
      .controls { grid-template-columns: 1fr; }
      .chart { height: 360px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>__TITLE__</h1>
      <p class="header-copy">Static reader for controlled manifold-slice artifacts. Metrics are computed in Python over all stored grid points; the browser only renders the selected record.</p>
    </div>
  </header>
  <main>
    <div class="controls">
      <label>Run<select id="run"></select></label>
      <label>Model<select id="model"></select></label>
      <label>Geometry<select id="geometry"></select></label>
      <label>Target<select id="target"></select></label>
      <label>Layer<select id="layer"></select></label>
    </div>
    <div id="content"></div>
  </main>
  <script>
    const records = __RECORDS_JSON__;
    const selects = {
      run: document.getElementById("run"),
      model: document.getElementById("model"),
      target: document.getElementById("target"),
      geometry: document.getElementById("geometry"),
      layer: document.getElementById("layer"),
    };
    const chartInstances = {};
    let renderToken = 0;
    let centroidMode = "3d";
    let centroidCache = null;
    let metricsCollapsed = false;

    function glReady() {
      return !!window.echarts && !window.__glLoadFailed;
    }

    const metricSpecs = {
      spearman_latent_vs_linear: {
        label: "Spearman: latent vs linear",
        help: "Rank correlation between true target-space pairwise distances and direct Euclidean distances between layer centroids. Higher is better; 1 means direct embedding distances preserve the target-distance ordering.",
      },
      spearman_latent_vs_geodesic: {
        label: "Spearman: latent vs geodesic",
        help: "Rank correlation between true target-space distances and shortest-path distances on the kNN graph of centroids. High values mean the representation has a recoverable manifold even if it is curved.",
      },
      pearson_latent_vs_linear: {
        label: "Pearson: latent vs linear",
        help: "Linear (Pearson) correlation between true target-space pairwise distances and direct Euclidean distances between centroids. Like the Spearman version but sensitive to magnitude, not only rank order.",
      },
      pearson_latent_vs_geodesic: {
        label: "Pearson: latent vs geodesic",
        help: "Linear (Pearson) correlation between true target-space distances and shortest-path graph distances between centroids. Sensitive to distance magnitude agreement, not just ordering.",
      },
      geodesic_gain: {
        label: "Geodesic gain",
        help: "Difference between geodesic Spearman and direct-linear Spearman. Positive values mean graph distances recover the target geometry better than straight-line embedding distances.",
      },
      stress_scaled: {
        label: "Scaled stress",
        help: "Scale-normalized embedding stress: residual mismatch between target-space distances and representation distances after optimal global scaling. Lower is better; 0 means distances match perfectly up to scale.",
      },
      knn_recall_at_1: {
        label: "kNN recall @ 1",
        help: "Average overlap between each point's single nearest neighbor in target space and in representation space. The strictest local-neighborhood check.",
      },
      knn_recall_at_3: {
        label: "kNN recall @ 3",
        help: "Average overlap between each point's 3 nearest neighbors in target space and in representation space.",
      },
      knn_recall_at_5: {
        label: "kNN recall @ 5",
        help: "Average overlap between each point's 5 nearest neighbors in target space and in representation space. This measures local neighborhood preservation.",
      },
      trustworthiness: {
        label: "Trustworthiness",
        help: "Penalizes representation neighbors that are not true target-space neighbors (false neighbors intruding into the local neighborhood). 1 is best.",
      },
      continuity: {
        label: "Continuity",
        help: "Penalizes true target-space neighbors that are not representation neighbors (true neighbors pushed away). 1 is best; complements trustworthiness.",
      },
      monotone_order_score: {
        label: "Monotone order score",
        help: "Interval-only: how monotonically the centroid ordering along the manifold follows the latent target coordinate. 1 means the path advances in latent order without backtracking.",
      },
      endpoint_separation: {
        label: "Endpoint separation",
        help: "Interval-only metric: distance between first and last centroids divided by the median adjacent-step distance. Large values mean the two ends did not collapse together.",
      },
      foldover_rate: {
        label: "Foldover rate",
        help: "Fraction of non-adjacent grid-point pairs that are closer than a typical adjacent step. Near 0 means few distant target values collapse together.",
      },
      circular_order_score: {
        label: "Circular order score",
        help: "Circle-only: agreement between the angular ordering of centroids around the recovered loop and the true phase ordering. Near 1 means the cycle is traversed in the correct order.",
      },
      cycle_closure_ratio: {
        label: "Cycle closure ratio",
        help: "Circle-only: how the first-to-last centroid gap compares to typical adjacent steps. Values near 1 mean the loop returns close to its starting point.",
      },
      cycle_closure_error: {
        label: "Cycle closure error",
        help: "Circle-only metric: how close the first and last centroids are relative to a typical adjacent step. Near 0 means the manifold closes cleanly.",
      },
      cycle_neighbor_wrap_score: {
        label: "Cycle neighbor wrap score",
        help: "Circle-only: fraction of endpoints whose nearest neighbors correctly wrap across the 0/2 pi seam. High values mean wraparound neighborhoods are preserved.",
      },
      projection_r2: {
        label: "Projection R2",
        help: "Validation coordinate recovery after projecting validation embeddings onto the train centroid polyline. Higher is better for interval targets; circle targets report circular error instead.",
      },
      projection_pearson: {
        label: "Projection Pearson",
        help: "Interval-only: Pearson correlation between true and recovered validation coordinates after projecting validation embeddings onto the train centroid path.",
      },
      projection_mae: {
        label: "Projection MAE",
        help: "Mean absolute error of target-coordinate recovery after projecting validation embeddings onto the train centroid path. Interpret this in the target's latent coordinate units.",
      },
      projection_rmse: {
        label: "Projection RMSE",
        help: "Root-mean-square error of validation target-coordinate recovery after projecting onto the train centroid path, in latent coordinate units. More sensitive to large misses than MAE.",
      },
      projection_circular_mae: {
        label: "Projection circular MAE",
        help: "Circle-only analogue of projection MAE: mean absolute angular error of recovered phase after projecting validation embeddings onto the centroid loop.",
      },
      mean_fiber_ratio: {
        label: "Mean fiber ratio",
        help: "Within-target variance divided by between-target signal. It only applies when there are multiple samples per grid point, such as nuisance variation or repeated samples.",
      },
      median_fiber_ratio: {
        label: "Median fiber ratio",
        help: "Median within-target spread divided by between-target signal across grid points. Robust version of the mean fiber ratio; only meaningful with multiple samples per grid point.",
      },
      max_fiber_ratio: {
        label: "Max fiber ratio",
        help: "Worst-case within-target spread relative to between-target signal across grid points. Flags the single thickest fiber.",
      },
      between_to_within_snr: {
        label: "Between/within SNR",
        help: "Signal-to-noise ratio of between-target separation versus within-target spread. Higher means grid points are cleanly separated relative to nuisance jitter.",
      },
    };
    // Bookkeeping/config fields that are numeric but not quality metrics.
    const metricBlocklist = new Set([
      "selected_geodesic_k",
      "usable_grid_points",
      "min_grid_point_count",
      "median_grid_point_count",
    ]);

    function specFor(key) {
      if (metricSpecs[key]) return metricSpecs[key];
      const label = key.replace(/_/g, " ").replace(/\\b\\w/g, (c) => c.toUpperCase());
      return { label, help: "Raw scalar metric emitted by the manifold evaluation. No curated description is available for this key yet." };
    }

    // Show every metric that has a finite value somewhere in the corpus:
    // curated keys first (in their defined order), then any uncurated extras.
    function discoverMetricKeys() {
      const finiteKeys = new Set();
      const nonScalar = new Set();
      for (const record of records) {
        const metrics = record.metrics || {};
        for (const [key, value] of Object.entries(metrics)) {
          if (value === null || value === undefined) continue;
          if (typeof value === "number") {
            if (Number.isFinite(value)) finiteKeys.add(key);
          } else {
            nonScalar.add(key);
          }
        }
      }
      const curated = Object.keys(metricSpecs).filter(
        (key) => finiteKeys.has(key) && !metricBlocklist.has(key),
      );
      const curatedSet = new Set(curated);
      const extras = [...finiteKeys]
        .filter((key) => !curatedSet.has(key) && !metricBlocklist.has(key) && !nonScalar.has(key) && !metricSpecs[key])
        .sort((a, b) => a.localeCompare(b));
      return [...curated, ...extras];
    }
    const metricOrder = discoverMetricKeys();

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function unique(values) {
      return [...new Set(values.filter(Boolean))].sort((a, b) => String(a).localeCompare(String(b)));
    }

    function sortLayers(values) {
      return [...new Set(values.filter(Boolean))].sort((a, b) => {
        const an = Number(a);
        const bn = Number(b);
        if (Number.isFinite(an) && Number.isFinite(bn)) return an - bn;
        return String(a).localeCompare(String(b));
      });
    }

    function fill(select, values, current, options = {}) {
      select.innerHTML = "";
      if (options.includeAll) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = options.allLabel || "(all)";
        select.appendChild(option);
      }
      for (const value of values) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value || "(missing)";
        select.appendChild(option);
      }
      if (current && values.includes(current)) {
        select.value = current;
      }
    }

    function filtered(except) {
      return records.filter((r) =>
        (except === "run" || !selects.run.value || r.run === selects.run.value) &&
        (except === "model" || !selects.model.value || r.model === selects.model.value) &&
        (except === "target" || !selects.target.value || r.target === selects.target.value) &&
        (except === "geometry" || !selects.geometry.value || r.geometry === selects.geometry.value) &&
        (except === "layer" || !selects.layer.value || r.layer === selects.layer.value)
      );
    }

    // Coarse-to-fine hierarchy. Each selector is narrowed only by the
    // selectors above it, never below: Run stays a full list regardless of the
    // model/target/layer in view, so sparse runs never drop out of the picker.
    const selectOrder = ["run", "model", "geometry", "target", "layer"];

    function refreshOptions() {
      const current = Object.fromEntries(Object.entries(selects).map(([k, s]) => [k, s.value]));
      selectOrder.forEach((key, idx) => {
        const pool = records.filter((r) =>
          selectOrder.slice(0, idx).every((k) => !selects[k].value || r[k] === selects[k].value),
        );
        const values = key === "layer"
          ? sortLayers(pool.map((r) => r[key]))
          : unique(pool.map((r) => r[key]));
        const options = key === "geometry"
          ? { includeAll: true, allLabel: "all geometries" }
          : {};
        fill(selects[key], values, current[key], options);
      });
    }

    function fmt(value) {
      if (value === null || value === undefined) return "n/a";
      if (typeof value === "number") {
        if (!Number.isFinite(value)) return "n/a";
        const abs = Math.abs(value);
        if ((abs >= 10000 || (abs > 0 && abs < 0.001))) return value.toExponential(3);
        return value.toFixed(4);
      }
      if (typeof value === "object") return JSON.stringify(value);
      return String(value);
    }

    function metricInterpretation(key, value) {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return "Current value is not available for this geometry or artifact.";
      }
      const number = Number(value);
      if (key.startsWith("spearman") || key.startsWith("pearson")) {
        if (number >= 0.9) return "Current value indicates strong distance-order preservation.";
        if (number >= 0.6) return "Current value indicates moderate distance-order preservation.";
        if (number >= 0.3) return "Current value indicates weak but visible distance-order preservation.";
        return "Current value indicates little distance-order preservation.";
      }
      if (key === "geodesic_gain") {
        if (number > 0.1) return "Graph distances are materially better than direct Euclidean distances here.";
        if (number >= -0.05) return "Graph distances are similar to direct Euclidean distances here.";
        return "Graph distances are worse than direct Euclidean distances here.";
      }
      if (key.startsWith("knn_recall")) {
        return `${fmt(number * 100)}% of true local neighbors are recovered on average.`;
      }
      if (key === "trustworthiness" || key === "continuity") {
        if (number >= 0.95) return "Current value indicates excellent neighborhood preservation.";
        if (number >= 0.8) return "Current value indicates good neighborhood preservation.";
        return "Current value indicates weak neighborhood preservation.";
      }
      if (key === "projection_r2") {
        if (number >= 0.95) return "Validation coordinates are almost perfectly recovered by the centroid path.";
        if (number >= 0.7) return "Validation coordinates are substantially recoverable by the centroid path.";
        return "Validation coordinate recovery is weak for this layer.";
      }
      if (key === "foldover_rate") {
        return number < 0.01 ? "Very few distant target values collapse together." : "Some distant target values collapse unusually close together.";
      }
      if (key === "cycle_closure_error") {
        return number < 0.1 ? "The circular manifold closes cleanly." : "The circular manifold closure is visibly imperfect.";
      }
      return `Current value: ${fmt(number)}.`;
    }

    function metricTooltip(key, value) {
      const spec = specFor(key);
      return `${spec.label}\\n\\n${spec.help}\\n\\n${metricInterpretation(key, value)}`;
    }

    function renderLayerMetricPanels(record) {
      return metricOrder.map((key) => {
        const spec = specFor(key);
        const value = record.metrics ? record.metrics[key] : undefined;
        const tooltip = escapeHtml(metricTooltip(key, value));
        const valueTip = escapeHtml(`Value at selected layer ${record.layer}: ${fmt(value)}`);
        return `
          <div class="layer-metric">
            <div class="layer-metric-head">
              <span class="metric-name" data-tip="${tooltip}">${escapeHtml(spec.label)}</span>
              <span class="metric-info" tabindex="0" role="button" aria-label="${escapeHtml(spec.label)} explanation" data-tip="${tooltip}">?</span>
              <span class="layer-metric-value" data-tip="${valueTip}">${escapeHtml(fmt(value))}</span>
            </div>
            <div class="chart-mini" id="metric-chart-${key}"></div>
          </div>`;
      }).join("");
    }

    function renderShell(record) {
      const sweep = record.sweep || {};
      const sweepParts = [];
      if (sweep.range_policy) sweepParts.push(sweep.range_policy);
      if (sweep.grid_mode) sweepParts.push(`grid=${sweep.grid_mode}`);
      if (sweep.physical_low !== undefined && sweep.physical_high !== undefined) {
        sweepParts.push(`range=[${fmt(sweep.physical_low)}, ${fmt(sweep.physical_high)}]`);
      }
      const sweepText = sweepParts.length ? sweepParts.join(", ") : "n/a";
      const metricsLink = record.metrics_json ? `<a href="${escapeHtml(record.metrics_json)}">metrics JSON</a>` : "";
      const plotDataLink = record.paths && record.paths.plot_data_json ? `<a href="${escapeHtml(record.paths.plot_data_json)}">plot data JSON</a>` : "";
      return `
        <div class="layout">
          <section class="panel summary-panel">
            <h2 class="record-title">${escapeHtml(record.model)} / ${escapeHtml(record.target)} / layer ${escapeHtml(record.layer)}</h2>
            <div class="record-meta">
              <div class="meta-row"><span>Geometry</span><span>${escapeHtml(record.geometry || "n/a")}</span></div>
              <div class="meta-row"><span>Sweep</span><span>${escapeHtml(sweepText)}</span></div>
              <div class="meta-row"><span>Selected geodesic k</span><span>${escapeHtml(fmt(record.metrics && record.metrics.selected_geodesic_k))}</span></div>
            </div>
            <div class="links">${metricsLink}${plotDataLink}</div>
          </section>
          <section class="charts">
            <section class="panel chart-card">
              <details class="collapsible"${metricsCollapsed ? "" : " open"}>
                <summary class="chart-header">
                  <h3><span class="chev"></span>Metrics across layers</h3>
                  <p class="chart-note">${escapeHtml(record.model)} / ${escapeHtml(record.target)}</p>
                  <p class="chart-copy">Each panel plots one manifold metric against layer depth, computed in Python over all stored grid points. The dashed line marks the selected layer; the number on each panel is that layer's value. Panels auto-scale independently.</p>
                </summary>
                <div class="layer-metrics">${renderLayerMetricPanels(record)}</div>
              </details>
            </section>
            <section class="panel chart-card">
              <div class="chart-header">
                <h3>Centroid path</h3>
                <div class="chart-head-right">
                  <div class="seg-toggle" role="group" aria-label="centroid projection dimensions">
                    <button type="button" data-centroid-mode="2d">2D</button>
                    <button type="button" data-centroid-mode="3d">3D</button>
                  </div>
                  <p class="chart-note" id="centroid-note"></p>
                </div>
                <p class="chart-copy">Centroids are ordered by the controlled target value and projected onto their top principal components with browser-side PCA. Axes are PCA components of the layer centroids; color is the latent target coordinate; the line is the ordered path (closed for circle targets). In 3D, drag to rotate and scroll to zoom.</p>
              </div>
              <div class="chart" id="centroid-chart"></div>
            </section>
            <section class="panel chart-card">
              <div class="chart-header">
                <h3>Distance scatter</h3>
                <p class="chart-note" id="scatter-note"></p>
                <p class="chart-copy">Each point is a pair of grid points. X is true target-space distance; Y is representation distance. Blue uses direct Euclidean distance between centroids, red uses shortest-path distance on the kNN graph.</p>
              </div>
              <div class="chart" id="scatter-chart"></div>
            </section>
            <section class="panel chart-card">
              <div class="chart-header">
                <h3>Distance heatmap</h3>
                <p class="chart-note" id="heatmap-note"></p>
                <p class="chart-copy">Pairwise distance matrices for the same grid subset: latent target distance, direct representation distance, and graph/geodesic distance. Good interval geometry has a dark diagonal and smoothly increasing values away from it.</p>
              </div>
              <div class="chart" id="heatmap-chart"></div>
            </section>
          </section>
        </div>`;
    }

    function getChart(id) {
      const element = document.getElementById(id);
      if (!element || !window.echarts) return null;
      const existing = chartInstances[id];
      // Re-initialize whenever the cached instance is bound to a detached
      // container. render() replaces #content innerHTML on every selector
      // change, so a non-disposed instance can still point at a stale DOM
      // node; reusing it would draw onto an invisible, detached canvas.
      if (!existing || existing.isDisposed() || existing.getDom() !== element) {
        if (existing && !existing.isDisposed()) existing.dispose();
        chartInstances[id] = echarts.init(element, null, { renderer: "canvas" });
      }
      return chartInstances[id];
    }

    function disposeCharts() {
      for (const id of Object.keys(chartInstances)) {
        const chart = chartInstances[id];
        if (chart && !chart.isDisposed()) chart.dispose();
        delete chartInstances[id];
      }
    }

    function setChartStatus(id, message, className = "status") {
      if (chartInstances[id] && !chartInstances[id].isDisposed()) {
        chartInstances[id].dispose();
      }
      chartInstances[id] = null;
      const element = document.getElementById(id);
      if (element) element.innerHTML = `<div class="${className}">${escapeHtml(message)}</div>`;
    }

    function finiteNumber(value) {
      return typeof value === "number" && Number.isFinite(value) ? value : Number.NaN;
    }

    function numericMatrix(matrix) {
      if (!Array.isArray(matrix)) return [];
      return matrix.map((row) => Array.isArray(row) ? row.map((value) => finiteNumber(value)) : []);
    }

    function dot(a, b) {
      let total = 0;
      for (let i = 0; i < a.length; i += 1) total += a[i] * b[i];
      return total;
    }

    function normalize(vector) {
      const norm = Math.sqrt(dot(vector, vector));
      if (!Number.isFinite(norm) || norm <= 0) return vector.map(() => 0);
      return vector.map((value) => value / norm);
    }

    function centeredRows(rows) {
      const n = rows.length;
      const d = rows[0] ? rows[0].length : 0;
      const means = new Array(d).fill(0);
      for (const row of rows) {
        for (let j = 0; j < d; j += 1) means[j] += Number.isFinite(row[j]) ? row[j] : 0;
      }
      for (let j = 0; j < d; j += 1) means[j] /= Math.max(n, 1);
      return rows.map((row) => means.map((mean, j) => (Number.isFinite(row[j]) ? row[j] : 0) - mean));
    }

    function multiplyCovariance(rows, vector) {
      const d = vector.length;
      const out = new Array(d).fill(0);
      for (const row of rows) {
        const score = dot(row, vector);
        for (let j = 0; j < d; j += 1) out[j] += score * row[j];
      }
      return out.map((value) => value / Math.max(rows.length - 1, 1));
    }

    function principalComponent(rows, bases, seed) {
      const d = rows[0] ? rows[0].length : 0;
      let vector = normalize(new Array(d).fill(0).map((_, idx) => Math.sin((idx + 1) * (seed + 1)) + 0.01 * (idx + 1)));
      for (let iter = 0; iter < 45; iter += 1) {
        vector = multiplyCovariance(rows, vector);
        for (const base of bases) {
          const projection = dot(vector, base);
          vector = vector.map((value, idx) => value - projection * base[idx]);
        }
        vector = normalize(vector);
      }
      return vector;
    }

    function project2d(points) {
      const rows = numericMatrix(points).filter((row) => row.length > 0);
      if (!rows.length) return [];
      const d = rows[0].length;
      if (d === 1) return rows.map((row) => [row[0], 0]);
      const centered = centeredRows(rows);
      const pc1 = principalComponent(centered, [], 1);
      const pc2 = principalComponent(centered, [pc1], 2);
      return centered.map((row) => [dot(row, pc1), dot(row, pc2)]);
    }

    function project3d(points) {
      const rows = numericMatrix(points).filter((row) => row.length > 0);
      if (!rows.length) return [];
      const d = rows[0].length;
      const centered = centeredRows(rows);
      const pc1 = principalComponent(centered, [], 1);
      const pc2 = d >= 2 ? principalComponent(centered, [pc1], 2) : null;
      const pc3 = d >= 3 ? principalComponent(centered, [pc1, pc2], 3) : null;
      return centered.map((row) => [
        dot(row, pc1),
        pc2 ? dot(row, pc2) : 0,
        pc3 ? dot(row, pc3) : 0,
      ]);
    }

    function finiteValues(values) {
      return values.filter((value) => Number.isFinite(value));
    }

    function upperTriangularPairs(latent, representation) {
      const out = [];
      const n = Math.min(latent.length, representation.length);
      for (let row = 0; row < n; row += 1) {
        const latentRow = latent[row] || [];
        const repRow = representation[row] || [];
        for (let col = row + 1; col < n; col += 1) {
          const x = Number(latentRow[col]);
          const y = Number(repRow[col]);
          if (Number.isFinite(x) && Number.isFinite(y)) out.push([x, y]);
        }
      }
      return out;
    }

    function matrixMinMax(matrices) {
      let low = Infinity;
      let high = -Infinity;
      for (const matrix of matrices) {
        for (const row of matrix) {
          for (const value of row) {
            const number = finiteNumber(value);
            if (Number.isFinite(number)) {
              low = Math.min(low, number);
              high = Math.max(high, number);
            }
          }
        }
      }
      if (!Number.isFinite(low) || !Number.isFinite(high)) return [0, 1];
      if (low === high) {
        low -= 0.5;
        high += 0.5;
      }
      return [low, high];
    }

    function heatmapData(matrix) {
      const data = [];
      for (let row = 0; row < matrix.length; row += 1) {
        for (let col = 0; col < matrix[row].length; col += 1) {
          const value = finiteNumber(matrix[row][col]);
          data.push([col, row, Number.isFinite(value) ? value : null]);
        }
      }
      return data;
    }

    function updateCentroidToggle() {
      for (const button of document.querySelectorAll("[data-centroid-mode]")) {
        button.classList.toggle("active", button.getAttribute("data-centroid-mode") === centroidMode);
      }
    }

    function renderCentroidChart(record, plotData) {
      centroidCache = { record, plotData };
      drawCentroid();
    }

    function drawCentroid() {
      if (!centroidCache) return;
      const { record, plotData } = centroidCache;
      const centroids = plotData.path_centroids || plotData.centroids || [];
      const coords = plotData.path_centroid_coordinates || plotData.centroid_coordinates || [];
      const counts = plotData.path_centroid_counts || plotData.centroid_counts || [];
      // Mode switches alternate a 2D cartesian grid and a 3D WebGL scene on the
      // same container, so start each draw from a fresh chart instance.
      const existing = chartInstances["centroid-chart"];
      if (existing && !existing.isDisposed()) existing.dispose();
      chartInstances["centroid-chart"] = null;
      const element = document.getElementById("centroid-chart");
      if (element) element.innerHTML = "";
      if (!window.echarts) {
        setChartStatus("centroid-chart", "Apache ECharts did not load. Check network access to the CDN and refresh.", "status warning");
        return;
      }
      if (!centroids.length || !coords.length) {
        setChartStatus("centroid-chart", "No centroid path data available for this layer.");
        return;
      }
      if (centroidMode === "3d") {
        if (!glReady()) {
          centroidMode = "2d";
          updateCentroidToggle();
        } else {
          renderCentroid3D(record, centroids, coords, counts);
          return;
        }
      }
      renderCentroid2D(record, centroids, coords, counts);
    }

    function renderCentroid2D(record, centroids, coords, counts) {
      const chart = getChart("centroid-chart");
      if (!chart) return;
      const points2d = project2d(centroids);
      const data = points2d.map((point, idx) => [point[0], point[1], Number(coords[idx]), idx, counts[idx] || 0]);
      const lineData = record.geometry === "circle" && data.length > 2 ? [...data, data[0]] : data;
      const coordValues = finiteValues(data.map((item) => item[2]));
      const low = coordValues.length ? Math.min(...coordValues) : 0;
      const high = coordValues.length ? Math.max(...coordValues) : 1;
      chart.setOption({
        animation: false,
        grid: { left: 64, right: 28, top: 42, bottom: 60 },
        tooltip: {
          trigger: "item",
          formatter: (params) => {
            const value = params.value || [];
            return [
              `<strong>Grid point ${value[3]}</strong>`,
              `latent coordinate: ${fmt(value[2])}`,
              `centroid count: ${fmt(value[4])}`,
              `PCA-1: ${fmt(value[0])}`,
              `PCA-2: ${fmt(value[1])}`,
            ].join("<br>");
          },
        },
        visualMap: {
          type: "continuous",
          min: low,
          max: high,
          dimension: 2,
          orient: "horizontal",
          left: "center",
          bottom: 8,
          text: ["high latent", "low latent"],
          inRange: { color: ["#2563eb", "#14b8a6", "#f59e0b", "#dc2626"] },
        },
        xAxis: {
          type: "value",
          name: "centroid PCA component 1",
          nameLocation: "middle",
          nameGap: 34,
          axisLine: { lineStyle: { color: "#94a3b8" } },
          splitLine: { lineStyle: { color: "#edf2f7" } },
        },
        yAxis: {
          type: "value",
          name: "centroid PCA component 2",
          nameLocation: "middle",
          nameGap: 46,
          axisLine: { lineStyle: { color: "#94a3b8" } },
          splitLine: { lineStyle: { color: "#edf2f7" } },
        },
        series: [
          {
            name: "ordered centroid path",
            type: "line",
            data: lineData,
            showSymbol: false,
            lineStyle: { width: 2.2, color: "#0f766e" },
            emphasis: { disabled: true },
          },
          {
            name: "centroids",
            type: "scatter",
            data,
            symbolSize: 7,
          },
        ],
      }, true);
    }

    function renderCentroid3D(record, centroids, coords, counts) {
      const chart = getChart("centroid-chart");
      if (!chart) return;
      const points3d = project3d(centroids);
      const data = points3d.map((point, idx) => [point[0], point[1], point[2], Number(coords[idx]), idx, counts[idx] || 0]);
      const lineData = (record.geometry === "circle" && data.length > 2 ? [...data, data[0]] : data)
        .map((item) => [item[0], item[1], item[2]]);
      const coordValues = finiteValues(data.map((item) => item[3]));
      const low = coordValues.length ? Math.min(...coordValues) : 0;
      const high = coordValues.length ? Math.max(...coordValues) : 1;
      try {
        chart.setOption({
          animation: false,
          tooltip: {
            formatter: (params) => {
              const value = params.value || [];
              return [
                `<strong>Grid point ${value[4]}</strong>`,
                `latent coordinate: ${fmt(value[3])}`,
                `centroid count: ${fmt(value[5])}`,
                `PCA-1: ${fmt(value[0])}`,
                `PCA-2: ${fmt(value[1])}`,
                `PCA-3: ${fmt(value[2])}`,
              ].join("<br>");
            },
          },
          visualMap: {
            type: "continuous",
            min: low,
            max: high,
            dimension: 3,
            orient: "horizontal",
            left: "center",
            bottom: 4,
            text: ["high latent", "low latent"],
            inRange: { color: ["#2563eb", "#14b8a6", "#f59e0b", "#dc2626"] },
          },
          xAxis3D: { type: "value", name: "PCA 1", axisLine: { lineStyle: { color: "#94a3b8" } } },
          yAxis3D: { type: "value", name: "PCA 2", axisLine: { lineStyle: { color: "#94a3b8" } } },
          zAxis3D: { type: "value", name: "PCA 3", axisLine: { lineStyle: { color: "#94a3b8" } } },
          grid3D: {
            boxWidth: 100,
            boxDepth: 100,
            boxHeight: 78,
            top: -10,
            axisLine: { lineStyle: { color: "#94a3b8" } },
            splitLine: { lineStyle: { color: "#e2e8f0" } },
            axisPointer: { lineStyle: { color: "#64748b" } },
            viewControl: { autoRotate: false, distance: 200, rotateSensitivity: 1.4 },
          },
          series: [
            {
              name: "ordered centroid path",
              type: "line3D",
              data: lineData,
              lineStyle: { width: 3.2, color: "#0f766e", opacity: 0.85 },
            },
            {
              name: "centroids",
              type: "scatter3D",
              data,
              symbolSize: 9,
              itemStyle: { opacity: 0.95 },
              emphasis: { itemStyle: { borderColor: "#0f172a", borderWidth: 1 } },
            },
          ],
        }, true);
      } catch (error) {
        // echarts-gl unavailable or incompatible: degrade to the 2D projection.
        centroidMode = "2d";
        updateCentroidToggle();
        renderCentroid2D(record, centroids, coords, counts);
      }
    }

    function renderScatterChart(plotData, metrics) {
      const chart = getChart("scatter-chart");
      if (!chart) {
        setChartStatus("scatter-chart", "Apache ECharts did not load. Check network access to the CDN and refresh.", "status warning");
        return;
      }
      const latent = numericMatrix(plotData.latent_distance);
      const linear = numericMatrix(plotData.linear_distance);
      const geodesic = plotData.geodesic_distance ? numericMatrix(plotData.geodesic_distance) : [];
      const linearData = upperTriangularPairs(latent, linear);
      const graphData = geodesic.length ? upperTriangularPairs(latent, geodesic) : [];
      chart.setOption({
        animation: false,
        grid: { left: 72, right: 28, top: 42, bottom: 62 },
        tooltip: {
          trigger: "item",
          formatter: (params) => {
            const value = params.value || [];
            return [
              `<strong>${escapeHtml(params.seriesName)}</strong>`,
              `target-space distance: ${fmt(value[0])}`,
              `representation distance: ${fmt(value[1])}`,
            ].join("<br>");
          },
        },
        legend: { top: 8, right: 18 },
        xAxis: {
          type: "value",
          name: "target-space pairwise distance",
          nameLocation: "middle",
          nameGap: 36,
          splitLine: { lineStyle: { color: "#edf2f7" } },
        },
        yAxis: {
          type: "value",
          name: "representation pairwise distance",
          nameLocation: "middle",
          nameGap: 52,
          splitLine: { lineStyle: { color: "#edf2f7" } },
        },
        series: [
          {
            name: `direct linear distance; rho=${fmt(metrics.spearman_latent_vs_linear)}`,
            type: "scatter",
            data: linearData,
            symbolSize: 3,
            large: linearData.length > 2000,
            itemStyle: { color: "#2563eb", opacity: 0.45 },
          },
          {
            name: `graph geodesic distance; rho=${fmt(metrics.spearman_latent_vs_geodesic)}`,
            type: "scatter",
            data: graphData,
            symbolSize: 3,
            large: graphData.length > 2000,
            itemStyle: { color: "#dc2626", opacity: 0.45 },
          },
        ],
      }, true);
    }

    function renderHeatmapChart(plotData) {
      const chart = getChart("heatmap-chart");
      if (!chart) {
        setChartStatus("heatmap-chart", "Apache ECharts did not load. Check network access to the CDN and refresh.", "status warning");
        return;
      }
      const matrices = [
        { name: "latent", matrix: numericMatrix(plotData.latent_distance) },
        { name: "linear", matrix: numericMatrix(plotData.linear_distance) },
      ];
      if (plotData.geodesic_distance) matrices.push({ name: "graph", matrix: numericMatrix(plotData.geodesic_distance) });
      const [low, high] = matrixMinMax(matrices.map((item) => item.matrix));
      const count = matrices.length;
      const gap = 4;
      const width = (86 - gap * (count - 1)) / count;
      const grids = matrices.map((_, idx) => ({
        left: `${7 + idx * (width + gap)}%`,
        top: 54,
        width: `${width}%`,
        height: 255,
      }));
      const axes = matrices.map((item, idx) => {
        const n = item.matrix.length;
        return {
          xAxis: {
            type: "category",
            gridIndex: idx,
            name: `${item.name} col index`,
            nameLocation: "middle",
            nameGap: 28,
            data: Array.from({ length: n }, (_, i) => i),
            axisLabel: { hideOverlap: true },
          },
          yAxis: {
            type: "category",
            gridIndex: idx,
            name: "row index",
            nameLocation: "middle",
            nameGap: 40,
            inverse: true,
            data: Array.from({ length: n }, (_, i) => i),
            axisLabel: { hideOverlap: true },
          },
        };
      });
      chart.setOption({
        animation: false,
        tooltip: {
          position: "top",
          formatter: (params) => {
            const value = params.value || [];
            return [
              `<strong>${escapeHtml(params.seriesName)}</strong>`,
              `row: ${value[1]}`,
              `col: ${value[0]}`,
              `distance: ${fmt(value[2])}`,
            ].join("<br>");
          },
        },
        visualMap: {
          min: low,
          max: high,
          calculable: true,
          orient: "horizontal",
          left: "center",
          bottom: 0,
          inRange: { color: ["#f8fafc", "#bfdbfe", "#14b8a6", "#f59e0b", "#dc2626"] },
        },
        grid: grids,
        xAxis: axes.map((item) => item.xAxis),
        yAxis: axes.map((item) => item.yAxis),
        series: matrices.map((item, idx) => ({
          name: item.name,
          type: "heatmap",
          xAxisIndex: idx,
          yAxisIndex: idx,
          data: heatmapData(item.matrix),
          progressive: 8000,
          emphasis: { itemStyle: { borderColor: "#0f172a", borderWidth: 1 } },
        })),
        graphic: matrices.map((item, idx) => ({
          type: "text",
          left: `${7 + idx * (width + gap)}%`,
          top: 22,
          style: {
            text: item.name,
            fill: "#334155",
            font: "600 13px IBM Plex Sans, Segoe UI, sans-serif",
          },
        })),
      }, true);
    }

    function layerRecords(record) {
      const seen = new Set();
      return records
        .filter((r) => r.run === record.run && r.model === record.model && r.target === record.target)
        .filter((r) => {
          if (seen.has(r.layer)) return false;
          seen.add(r.layer);
          return true;
        })
        .map((r) => ({ layer: r.layer, layerNum: Number(r.layer), metrics: r.metrics || {} }))
        .sort((a, b) => {
          const af = Number.isFinite(a.layerNum);
          const bf = Number.isFinite(b.layerNum);
          if (af && bf) return a.layerNum - b.layerNum;
          if (af) return -1;
          if (bf) return 1;
          return String(a.layer).localeCompare(String(b.layer));
        });
    }

    function renderLayerMetricsCharts(record) {
      const rows = layerRecords(record);
      const numeric = rows.length > 0 && rows.every((row) => Number.isFinite(row.layerNum));
      const selectedLayer = Number(record.layer);
      for (const key of metricOrder) {
        const id = `metric-chart-${key}`;
        const points = rows.map((row, idx) => {
          const value = finiteNumber(row.metrics[key]);
          return [numeric ? row.layerNum : idx, Number.isFinite(value) ? value : null];
        });
        if (!points.some((point) => point[1] !== null)) {
          setChartStatus(id, "Not available for this target.");
          continue;
        }
        const chart = getChart(id);
        if (!chart) {
          setChartStatus(id, "Apache ECharts did not load. Check network access to the CDN and refresh.", "status warning");
          continue;
        }
        const markLine = numeric && Number.isFinite(selectedLayer)
          ? {
              silent: true,
              symbol: "none",
              label: { show: false },
              lineStyle: { color: "#dc2626", type: "dashed", width: 1.2 },
              data: [{ xAxis: selectedLayer }],
            }
          : undefined;
        chart.setOption({
          animation: false,
          grid: { left: 56, right: 12, top: 12, bottom: 30 },
          tooltip: {
            trigger: "axis",
            formatter: (params) => {
              const point = Array.isArray(params) && params.length ? params[0] : null;
              if (!point) return "";
              const value = point.value || [];
              return `layer ${fmt(value[0])}<br>${escapeHtml(specFor(key).label)}: ${fmt(value[1])}`;
            },
          },
          xAxis: {
            type: numeric ? "value" : "category",
            name: "layer",
            nameLocation: "middle",
            nameGap: 19,
            minInterval: 1,
            data: numeric ? undefined : rows.map((row) => row.layer),
            axisLine: { lineStyle: { color: "#94a3b8" } },
            splitLine: { show: false },
          },
          yAxis: {
            type: "value",
            scale: true,
            axisLine: { lineStyle: { color: "#94a3b8" } },
            splitLine: { lineStyle: { color: "#edf2f7" } },
            axisLabel: { fontSize: 10, formatter: (value) => fmt(value) },
          },
          series: [
            {
              type: "line",
              data: points,
              showSymbol: true,
              symbolSize: 4,
              connectNulls: false,
              lineStyle: { width: 1.8, color: "#0f766e" },
              itemStyle: { color: "#0f766e" },
              markLine,
            },
          ],
        }, true);
      }
    }

    function updateNotes(plotData) {
      const source = Number(plotData.source_grid_points || 0);
      const plot = Number(plotData.plot_grid_points || 0);
      const path = Number(plotData.path_grid_points || plot || 0);
      const pathNote = path && source && path < source
        ? `path shows ${path} stored points; source grid has ${source}`
        : `path shows ${path || "n/a"} grid points`;
      const distanceNote = source && plot && plot < source
        ? `distance charts show ${plot} of ${source}; metrics use all ${source}`
        : `distance charts show ${plot || "n/a"} grid points`;
      const centroidNote = document.getElementById("centroid-note");
      const scatterNote = document.getElementById("scatter-note");
      const heatmapNote = document.getElementById("heatmap-note");
      if (centroidNote) centroidNote.textContent = pathNote;
      if (scatterNote) scatterNote.textContent = distanceNote;
      if (heatmapNote) heatmapNote.textContent = distanceNote;
    }

    async function renderPlotData(record, token) {
      if (!window.echarts) {
        for (const id of ["centroid-chart", "scatter-chart", "heatmap-chart"]) {
          setChartStatus(id, "Apache ECharts did not load. Check network access to the CDN and refresh.", "status warning");
        }
        return;
      }
      const path = record.paths && record.paths.plot_data_json;
      if (!path) {
        for (const id of ["centroid-chart", "scatter-chart", "heatmap-chart"]) {
          setChartStatus(id, "No plot data JSON was recorded for this layer.");
        }
        return;
      }
      try {
        const response = await fetch(path);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const plotData = await response.json();
        if (token !== renderToken) return;
        updateNotes(plotData);
        renderCentroidChart(record, plotData);
        renderScatterChart(plotData, record.metrics || {});
        renderHeatmapChart(plotData);
      } catch (error) {
        const message = `Could not load plot data JSON (${path}). Serve this run directory over a static HTTP server if the browser blocks local file fetches. ${error}`;
        for (const id of ["centroid-chart", "scatter-chart", "heatmap-chart"]) {
          setChartStatus(id, message, "status warning");
        }
      }
    }

    function render() {
      refreshOptions();
      disposeCharts();
      const content = document.getElementById("content");
      if (!records.length) {
        content.innerHTML = '<div class="empty panel">No calibration metrics found.</div>';
        return;
      }
      const matches = filtered();
      if (!matches.length) {
        content.innerHTML = '<div class="empty panel">No records match the current filters.</div>';
        return;
      }
      const record = matches[0];
      content.innerHTML = renderShell(record);
      renderLayerMetricsCharts(record);
      updateCentroidToggle();
      const token = ++renderToken;
      renderPlotData(record, token);
    }

    const tip = document.createElement("div");
    tip.id = "tip";
    document.body.appendChild(tip);
    let tipTarget = null;

    function positionTip() {
      if (!tipTarget) return;
      const rect = tipTarget.getBoundingClientRect();
      const margin = 8;
      const tw = tip.offsetWidth;
      const th = tip.offsetHeight;
      let left = rect.left;
      let top = rect.bottom + margin;
      if (top + th > window.innerHeight - 4) top = rect.top - th - margin;
      if (left + tw > window.innerWidth - 8) left = window.innerWidth - tw - 8;
      if (left < 8) left = 8;
      tip.style.left = `${Math.round(left)}px`;
      tip.style.top = `${Math.round(Math.max(top, 4))}px`;
    }

    function showTip(target) {
      const text = target.getAttribute("data-tip");
      if (!text) return;
      tipTarget = target;
      tip.textContent = text;
      tip.classList.add("visible");
      positionTip();
    }

    function hideTip() {
      tipTarget = null;
      tip.classList.remove("visible");
    }

    document.addEventListener("pointerover", (event) => {
      const target = event.target.closest ? event.target.closest("[data-tip]") : null;
      if (target && target !== tipTarget) showTip(target);
    });
    document.addEventListener("pointerout", (event) => {
      const target = event.target.closest ? event.target.closest("[data-tip]") : null;
      if (target && target === tipTarget && (!event.relatedTarget || !target.contains(event.relatedTarget))) {
        hideTip();
      }
    });
    document.addEventListener("focusin", (event) => {
      const target = event.target.closest ? event.target.closest("[data-tip]") : null;
      if (target) showTip(target);
    });
    document.addEventListener("focusout", hideTip);
    window.addEventListener("scroll", hideTip, true);

    document.addEventListener("click", (event) => {
      const button = event.target.closest ? event.target.closest("[data-centroid-mode]") : null;
      if (!button) return;
      const mode = button.getAttribute("data-centroid-mode");
      if (mode !== "2d" && mode !== "3d") return;
      centroidMode = mode;
      updateCentroidToggle();
      drawCentroid();
    });

    // Charts hidden inside a collapsed <details> can be resized to zero by the
    // window resize handler; re-fit them when the section is expanded.
    document.addEventListener("toggle", (event) => {
      const target = event.target;
      if (!target || !target.matches || !target.matches("details.collapsible")) return;
      // Remember the collapse state so it survives shell re-renders on selection.
      metricsCollapsed = !target.open;
      if (!target.open) return;
      for (const chart of Object.values(chartInstances)) {
        if (chart && !chart.isDisposed()) chart.resize();
      }
    }, true);

    for (const select of Object.values(selects)) select.addEventListener("change", render);
    window.addEventListener("resize", () => {
      for (const chart of Object.values(chartInstances)) {
        if (chart && !chart.isDisposed()) chart.resize();
      }
    });

    // With a single run there is nothing to switch between, so hide the picker.
    if (unique(records.map((r) => r.run)).length <= 1) {
      const runLabel = selects.run.closest("label");
      if (runLabel) runLabel.style.display = "none";
    }

    refreshOptions();
    render();
  </script>
</body>
</html>
"""
