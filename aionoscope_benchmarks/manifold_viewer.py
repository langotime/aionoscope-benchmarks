from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


MANIFOLD_VIEWER_MANIFEST = "manifest.json"
MANIFOLD_VIEWER_MANIFEST_SCHEMA = "manifold_viewer_manifest_v1"


def _format_checkpoint_step(step: int) -> str:
    if step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def _artifact_relpath(path: str | None, *, artifact_root: Path, context_path: Path | None = None) -> str | None:
    if not path:
        return None
    root = artifact_root.resolve()
    raw = Path(path)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw.resolve())
    else:
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((root / raw).resolve())
        if context_path is not None:
            candidates.append((context_path.parent / raw).resolve())

    for candidate in candidates:
        try:
            relative = os.path.relpath(candidate, root)
        except ValueError:
            continue
        if not relative.startswith(".."):
            return relative.replace(os.sep, "/")
    # Stored visualization paths can be absolute into a *different* root than the
    # current artifact_root (e.g. the run was produced in another worktree, then
    # the tree was moved). The on-disk layout is stable
    # (<root>/<model>/<target>/<artifact-tail>), so rebuild the relative path from
    # the real metrics path under root plus the tail after the <model>/<target>
    # segment of the stored path.
    if context_path is not None:
        rel_dir = os.path.relpath(context_path.parent, root)
        if not rel_dir.startswith(".."):
            rel_dir = rel_dir.replace(os.sep, "/")
            marker = f"/{rel_dir}/"
            raw_posix = str(raw).replace(os.sep, "/")
            index = raw_posix.rfind(marker)
            if index != -1:
                tail = raw_posix[index + len(marker):]
                return f"{rel_dir}/{tail}" if tail else rel_dir
    return str(raw).replace(os.sep, "/")


def _run_name_for_metrics_path(
    *,
    artifact_root: Path,
    metrics_path: Path,
    model_slug: str,
) -> str:
    try:
        relative_parts = metrics_path.resolve().relative_to(artifact_root.resolve()).parts
    except ValueError:
        relative_parts = metrics_path.parts
    if relative_parts and str(relative_parts[0]) == str(model_slug):
        return artifact_root.name
    if len(relative_parts) >= 4:
        return str(relative_parts[0])
    return artifact_root.name


def _signal_spec(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Compact, reproducible description of the input waveform for one controlled slice.

    The browser regenerates the signal from the per-grid swept value
    (``centroid_coordinates`` in the plot data) plus these fixed constants, so no
    raw waveform is stored. Returns ``None`` when the metrics payload does not name a
    component (e.g. minimal synthetic fixtures)."""
    target = payload.get("target", {}) or {}
    component = str(target.get("component", ""))
    if not component:
        return None
    train_slice_manifest = payload.get("train_slice_manifest", {}) or {}
    fixed_all = train_slice_manifest.get("fixed_values", {}) or {}
    dataset_manifest = train_slice_manifest.get("dataset_manifest", {}) or {}
    adapter = (payload.get("model", {}) or {}).get("adapter", {}) or {}
    seq_len = adapter.get("benchmark_sequence_length")
    if seq_len is None:
        seq_len = dataset_manifest.get("channel_size")
    fixed = fixed_all.get(component)
    return {
        "component": component,
        "param": str(target.get("parameter", "")),
        "coordinate_name": str(target.get("coordinate_name", "")),
        "geometry": str(target.get("geometry", "")),
        "seq_len": seq_len,
        "fs": dataset_manifest.get("sampling_frequency"),
        "fixed": fixed if isinstance(fixed, dict) else {},
    }


def collect_viewer_manifest_records(*, artifact_root: Path) -> list[dict[str, Any]]:
    records = []
    for metrics_path in sorted(artifact_root.glob("**/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        model = payload.get("model", {})
        model_slug = str(model.get("slug", ""))
        # Layout is usually <model>/<target>/metrics.json under one run root,
        # but an aggregate root may contain <run>/<model>/<target>/metrics.json.
        # Checkpoint sweeps add <model>/<checkpoint>/<target>/metrics.json,
        # which is still part of the aggregate root rather than a separate run.
        run_name = _run_name_for_metrics_path(
            artifact_root=artifact_root,
            metrics_path=metrics_path,
            model_slug=model_slug,
        )
        target = payload.get("target", {})
        train_slice_manifest = payload.get("train_slice_manifest", {})
        sweep = train_slice_manifest.get("sweep", {}) if isinstance(train_slice_manifest, dict) else {}
        by_layer = payload.get("by_layer", {})
        visualizations = payload.get("visualizations", {})
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
        layers = []
        for layer_key in by_layer:
            viz = visualizations.get(str(layer_key), {})
            layers.append(
                {
                    "layer": str(layer_key),
                    "paths": {
                        key: _artifact_relpath(value, artifact_root=artifact_root, context_path=metrics_path)
                        for key, value in viz.items()
                        if key in {"plot_data_json", "distance_data_json"}
                        if isinstance(value, str)
                    },
                }
            )
        record = {
            "run": run_name,
            "model": str(model.get("name", model.get("slug", ""))),
            "model_slug": model_slug,
            "target": target_label,
            "target_name": target_name,
            "sweep": sweep if isinstance(sweep, dict) else {},
            "geometry": str(target.get("geometry", "")),
            "metrics_json": _artifact_relpath(str(metrics_path), artifact_root=artifact_root),
            "layers": layers,
        }
        adapter = model.get("adapter", {}) if isinstance(model, dict) else {}
        if not isinstance(adapter, dict):
            adapter = {}
        checkpoint_index = model.get("checkpoint_index", adapter.get("checkpoint_index"))
        checkpoint_step = model.get("checkpoint_step", adapter.get("checkpoint_step"))
        if checkpoint_index is not None or checkpoint_step is not None:
            record["checkpoint_index"] = checkpoint_index
            record["checkpoint_step"] = checkpoint_step
            record["checkpoint_path"] = model.get("checkpoint_path", adapter.get("checkpoint_path"))
            checkpoint_label = adapter.get("checkpoint_label")
            if not checkpoint_label:
                if checkpoint_index is not None and checkpoint_step is not None:
                    checkpoint_label = (
                        f"#{int(checkpoint_index):03d} / "
                        f"{_format_checkpoint_step(int(checkpoint_step))}"
                    )
                elif checkpoint_step is not None:
                    checkpoint_label = _format_checkpoint_step(int(checkpoint_step))
                else:
                    checkpoint_label = f"#{int(checkpoint_index):03d}"
            record["checkpoint_label"] = checkpoint_label
        signal_spec = _signal_spec(payload)
        if signal_spec is not None:
            record["signal_spec"] = signal_spec
        records.append(record)
    return records


def collect_viewer_manifest(*, artifact_root: Path) -> dict[str, Any]:
    records = collect_viewer_manifest_records(artifact_root=artifact_root)
    return {
        "schema_version": MANIFOLD_VIEWER_MANIFEST_SCHEMA,
        "records": records,
    }


def collect_viewer_records(*, artifact_root: Path, viewer_path: Path | None = None) -> list[dict[str, Any]]:
    del viewer_path
    flat_records = []
    for target_record in collect_viewer_manifest_records(artifact_root=artifact_root):
        layers = target_record.get("layers", [])
        for layer in layers:
            record = {key: value for key, value in target_record.items() if key != "layers"}
            record["layer"] = str(layer.get("layer", ""))
            record["paths"] = layer.get("paths", {})
            flat_records.append(record)
    return flat_records


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_manifest(*, artifact_root: Path) -> Path:
    manifest_path = artifact_root / MANIFOLD_VIEWER_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = collect_viewer_manifest(artifact_root=artifact_root)
    manifest_path.write_text(
        json.dumps(_json_safe(manifest_payload), separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path


def build_viewer_manifest(*, artifact_root: Path) -> Path:
    artifact_root.mkdir(parents=True, exist_ok=True)
    return _write_manifest(artifact_root=artifact_root)


def build_viewer(*, artifact_root: Path, out_path: Path) -> None:
    build_viewer_manifest(artifact_root=artifact_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = "Aionoscope Manifold Viewer"
    html_text = _VIEWER_TEMPLATE.replace("__TITLE__", title)
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
    .about-link {
      flex: none;
      align-self: center;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 9px 16px;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      font-size: 13px;
      font-weight: 650;
      white-space: nowrap;
      box-shadow: 0 6px 16px rgba(15, 118, 110, 0.25);
      transition: background 0.15s ease, transform 0.15s ease;
    }
    .about-link:hover { background: #0c5f58; transform: translateY(-1px); }
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
    .pair-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; min-width: 0; }
    .pair-row > section { min-width: 0; margin: 0; }
    .pair-row > section:nth-child(1) { order: 2; }  /* Centroid path -> right */
    .pair-row > section:nth-child(2) { order: 1; }  /* Input signal -> left */
    @media (max-width: 980px) { .pair-row { grid-template-columns: 1fr; } }
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
    .distance-section { margin-top: 12px; }
    .distance-section + .distance-section {
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid #edf1f5;
    }
    .distance-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin: 0 0 6px;
    }
    .distance-header h4 {
      margin: 0;
      color: #334155;
      font-size: 13px;
      line-height: 1.2;
    }
    .comparison {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin: 0 0 16px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid var(--panel-border);
      border-radius: 8px;
    }
    .comparison-label { font-size: 12px; font-weight: 720; color: var(--muted); }
    .comparison button {
      font: inherit;
      font-size: 12px;
      font-weight: 650;
      padding: 5px 11px;
      border: 1px solid #c8d0d9;
      border-radius: 7px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
    }
    .comparison button.ghost { background: #fff; color: #334155; }
    .comparison button[disabled] { opacity: 0.45; cursor: not-allowed; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; flex: 1 1 auto; min-width: 0; }
    .chips-empty { color: var(--muted); font-size: 12px; }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 3px 6px 3px 9px;
      border: 1px solid var(--panel-border);
      border-radius: 999px;
      background: #fff;
      font-size: 12px;
    }
    .chip-dot { width: 10px; height: 10px; border-radius: 999px; background: var(--chip, #0f766e); flex: 0 0 auto; }
    .chip-label { color: #334155; }
    .chip-x { border: 0; background: transparent; color: #64748b; cursor: pointer; font-size: 15px; line-height: 1; padding: 0 2px; }
    .chip-x:hover { color: var(--red); }
    .comparison-hint { color: var(--muted); font-size: 11px; }
    .sbs-grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
    .sbs-grid.heatmap-grid { grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); }
    .sbs-cell { border: 1px solid #edf1f5; border-radius: 8px; padding: 8px 8px 4px; background: #fbfdff; min-width: 0; }
    .sbs-title { display: flex; align-items: center; gap: 7px; font-size: 12px; font-weight: 650; color: #334155; margin-bottom: 4px; }
    .sbs-chart { height: 330px; border-top: 0; }
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
    .signal-controls { display: flex; align-items: center; gap: 14px; margin: 12px 4px 2px; }
    .signal-controls .signal-play {
      flex: none; width: 34px; height: 34px; border-radius: 50%; border: 0; cursor: pointer;
      background: #0f766e; color: #fff; font-size: 12px; line-height: 1;
      display: inline-flex; align-items: center; justify-content: center;
    }
    .signal-controls .signal-play:hover { background: #0c5f58; }
    .signal-controls input[type="range"] { flex: 1; accent-color: #0f766e; height: 4px; cursor: pointer; }
    .signal-controls .readout {
      flex: none; min-width: 188px; text-align: right; font-variant-numeric: tabular-nums;
      font-size: 13px; font-weight: 600; color: #0f172a; font-family: ui-monospace, "SF Mono", Menlo, monospace;
    }
    @media (max-width: 720px) { .signal-controls .readout { min-width: 120px; font-size: 11px; } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>__TITLE__</h1>
      <p class="header-copy">Static reader for controlled manifold-slice artifacts. Metrics are computed in Python over all stored grid points; the browser only renders the selected record.</p>
    </div>
    <a class="about-link" href="about-manifolds.html">About manifolds &rarr;</a>
  </header>
  <main>
    <div class="controls">
      <label>Run<select id="run"></select></label>
      <label>Model<select id="model"></select></label>
      <label id="checkpoint-control">Checkpoint<select id="checkpoint"></select></label>
      <label>Geometry<select id="geometry"></select></label>
      <label>Target<select id="target"></select></label>
      <label>Layer<select id="layer"></select></label>
    </div>
    <div id="comparison" class="comparison"></div>
    <div id="content"></div>
  </main>
  <script>
    const MANIFEST_PATH = "manifest.json";
    const selects = {
      run: document.getElementById("run"),
      model: document.getElementById("model"),
      checkpoint: document.getElementById("checkpoint"),
      target: document.getElementById("target"),
      geometry: document.getElementById("geometry"),
      layer: document.getElementById("layer"),
    };
    const chartInstances = {};
    let records = [];
    let metricOrder = [];
    let renderToken = 0;
    let centroidMode = "3d";
    let sideCache = null;
    let signalIndex = 0;
    let signalPlaying = false;
    let signalTimer = null;
    let signalRaf = null;
    let metricsCollapsed = false;
    let distanceBlockOpen = false;
    const REMOTE_MANIFOLD_DATA_BASE_URL = "https://manifolds-data.aionoscope.langotime.ai/manifolds/v20260603T142443Z/";
    function defaultManifoldDataBaseUrl() {
      const host = window.location.hostname;
      if (
        window.location.protocol === "file:" ||
        host === "localhost" ||
        host === "127.0.0.1" ||
        host === "::1" ||
        host === "[::1]"
      ) {
        return window.location.pathname.endsWith("manifolds.html") ? "manifolds/" : "";
      }
      return REMOTE_MANIFOLD_DATA_BASE_URL;
    }
    const MANIFOLD_DATA_BASE_URL = window.MANIFOLD_DATA_BASE_URL ?? defaultManifoldDataBaseUrl();

    const PALETTE = ["#0f766e", "#2563eb", "#d97706", "#db2777"];
    const MAX_ITEMS = 4;
    const selection = [];
    const targetMetricsCache = new Map();
    const plotDataCache = new Map();
    const distanceDataCache = new Map();

    function glReady() {
      return !!window.echarts && !window.__glLoadFailed;
    }

    function recordKey(r) {
      return `${r.run}|${r.model}|${r.checkpoint || ""}|${r.target}|${r.layer}`;
    }
    function targetDataKey(r) {
      return `${r.run}|${r.metrics_json || ""}`;
    }
    function itemColor(idx) {
      return PALETTE[idx % PALETTE.length];
    }
    function modelDisplay(record) {
      return record && record.checkpoint ? `${record.model} / ${record.checkpoint}` : String(record && record.model || "");
    }
    function trackLabel(track) {
      return `${track.model_label || track.model} / ${track.target}`;
    }
    function activeCandidate() {
      const matches = filtered();
      return matches.length ? matches[0] : null;
    }
    function effectiveItems() {
      if (selection.length) return selection.slice();
      const candidate = activeCandidate();
      return candidate ? [candidate] : [];
    }
    function tracksOf(items) {
      const seen = new Map();
      items.forEach((it, idx) => {
        const tkey = `${it.run}|${it.model}|${it.checkpoint || ""}|${it.target}`;
        if (!seen.has(tkey)) {
          seen.set(tkey, {
            tkey,
            color: itemColor(idx),
            record: it,
            model: it.model,
            model_label: modelDisplay(it),
            checkpoint: it.checkpoint || "",
            target: it.target,
            layers: [],
          });
        }
        seen.get(tkey).layers.push(it.layer);
      });
      return [...seen.values()];
    }

    function normalizeManifoldDataPath(path) {
      const raw = String(path);
      const embeddedRoot = "/results/manifolds/";
      const embeddedIndex = raw.lastIndexOf(embeddedRoot);
      if (embeddedIndex !== -1) return raw.slice(embeddedIndex + embeddedRoot.length);
      const relativeRoot = "results/manifolds/";
      if (raw.startsWith(relativeRoot)) return raw.slice(relativeRoot.length);
      return raw;
    }

    function resolveDataUrl(path) {
      if (!path) return path;
      if (/^https?:\\/\\//.test(path)) return path;
      const dataPath = normalizeManifoldDataPath(path);
      if (!MANIFOLD_DATA_BASE_URL) return dataPath;
      const base = String(MANIFOLD_DATA_BASE_URL);
      const normalizedBase = base.endsWith("/") ? base : `${base}/`;
      return new URL(dataPath, new URL(normalizedBase, window.location.href)).toString();
    }

    function fetchJson(path) {
      const url = resolveDataUrl(path);
      return fetch(url).then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status} while loading ${url}`);
        return response.json();
      });
    }

    function buildRecordsFromManifest(manifest) {
      const targetRecords = Array.isArray(manifest && manifest.records) ? manifest.records : [];
      return targetRecords.flatMap((targetRecord) => {
        const layers = Array.isArray(targetRecord.layers) ? targetRecord.layers : [];
        return layers.map((layerRecord) => ({
          run: String(targetRecord.run || ""),
          model: String(targetRecord.model || targetRecord.model_slug || ""),
          model_slug: String(targetRecord.model_slug || ""),
          checkpoint: String(targetRecord.checkpoint_label || ""),
          checkpoint_index: targetRecord.checkpoint_index === undefined ? null : targetRecord.checkpoint_index,
          checkpoint_step: targetRecord.checkpoint_step === undefined ? null : targetRecord.checkpoint_step,
          target: String(targetRecord.target || targetRecord.target_name || ""),
          target_name: String(targetRecord.target_name || ""),
          sweep: targetRecord.sweep && typeof targetRecord.sweep === "object" ? targetRecord.sweep : {},
          geometry: String(targetRecord.geometry || ""),
          signal_spec: targetRecord.signal_spec && typeof targetRecord.signal_spec === "object" ? targetRecord.signal_spec : null,
          metrics_json: String(targetRecord.metrics_json || ""),
          layer: String(layerRecord.layer === undefined || layerRecord.layer === null ? "" : layerRecord.layer),
          paths: layerRecord.paths && typeof layerRecord.paths === "object" ? layerRecord.paths : {},
          metrics: null,
          summary: null,
          metricsLoaded: false,
          metricsError: null,
        }));
      });
    }

    function sameTargetRecords(record) {
      const key = targetDataKey(record);
      return records.filter((candidate) => targetDataKey(candidate) === key);
    }

    function applyTargetMetrics(record, payload) {
      const byLayer = payload && payload.by_layer && typeof payload.by_layer === "object" ? payload.by_layer : {};
      const visualizations = payload && payload.visualizations && typeof payload.visualizations === "object" ? payload.visualizations : {};
      const summary = payload && payload.summary && typeof payload.summary === "object" ? payload.summary : {};
      const signalSpec = signalSpecFromPayload(payload);
      sameTargetRecords(record).forEach((targetRecord) => {
        if (signalSpec && !targetRecord.signal_spec) targetRecord.signal_spec = signalSpec;
        const layerKey = String(targetRecord.layer);
        targetRecord.metrics = byLayer[layerKey] && typeof byLayer[layerKey] === "object" ? byLayer[layerKey] : {};
        targetRecord.summary = summary;
        const viz = visualizations[layerKey];
        if (viz && typeof viz === "object") {
          targetRecord.paths = {
            ...(targetRecord.paths || {}),
            ...(viz.plot_data_json && !targetRecord.paths.plot_data_json ? { plot_data_json: viz.plot_data_json } : {}),
            ...(viz.distance_data_json && !targetRecord.paths.distance_data_json ? { distance_data_json: viz.distance_data_json } : {}),
          };
        }
        targetRecord.metricsLoaded = true;
        targetRecord.metricsError = null;
      });
    }

    function markTargetMetricsError(record, error) {
      sameTargetRecords(record).forEach((targetRecord) => {
        targetRecord.metricsLoaded = false;
        targetRecord.metricsError = String(error);
      });
    }

    function loadTargetMetrics(record) {
      if (!record || !record.metrics_json) return Promise.resolve(null);
      if (record.metricsLoaded) return Promise.resolve(targetMetricsCache.get(targetDataKey(record)) || null);
      if (record.metricsError) return Promise.resolve({ __error: record.metricsError });
      const key = targetDataKey(record);
      if (!targetMetricsCache.has(key)) {
        const promise = fetchJson(record.metrics_json)
          .then((payload) => {
            applyTargetMetrics(record, payload);
            return payload;
          })
          .catch((error) => {
            markTargetMetricsError(record, error);
            return { __error: String(error) };
          });
        targetMetricsCache.set(key, promise);
      }
      return targetMetricsCache.get(key);
    }

    function missingMetricTargets(items) {
      const seen = new Set();
      return items.filter((item) => {
        if (!item || item.metricsLoaded || item.metricsError) return false;
        const key = targetDataKey(item);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
    }

    function metricErrors(items) {
      return items.map((item) => item && item.metricsError).filter(Boolean);
    }

    function renderComparison() {
      const host = document.getElementById("comparison");
      if (!host) return;
      const candidate = activeCandidate();
      const candKey = candidate ? recordKey(candidate) : null;
      const inSel = candKey && selection.some((it) => recordKey(it) === candKey);
      const full = selection.length >= MAX_ITEMS;
      const addDisabled = !candidate || inSel || full;
      const addTitle = !candidate
        ? "Nothing selected"
        : inSel ? "Already in comparison" : full ? `At most ${MAX_ITEMS} items` : "Add the current selection";
      const chips = selection.map((it, idx) => `
        <span class="chip">
          <span class="chip-dot" style="--chip:${itemColor(idx)}"></span>
          <span class="chip-label">${escapeHtml(modelDisplay(it))} / ${escapeHtml(it.target)} &middot; L${escapeHtml(it.layer)}</span>
          <button type="button" class="chip-x" data-remove="${escapeHtml(recordKey(it))}" aria-label="remove">&times;</button>
        </span>`).join("");
      host.innerHTML = `
        <span class="comparison-label">Comparison</span>
        <button type="button" id="add-comparison"${addDisabled ? " disabled" : ""} title="${escapeHtml(addTitle)}">+ Add current</button>
        <div class="chips">${chips || '<span class="chips-empty">empty &mdash; showing the current selection</span>'}</div>
        ${selection.length ? '<button type="button" id="clear-comparison" class="ghost">Clear</button>' : ""}
        <span class="comparison-hint">up to ${MAX_ITEMS}; overlaid on metrics, side-by-side on plots</span>`;
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

    function checkpointValues(pool) {
      const rows = pool
        .filter((r) => r.checkpoint)
        .slice()
        .sort((a, b) => {
          const ai = Number(a.checkpoint_index);
          const bi = Number(b.checkpoint_index);
          if (Number.isFinite(ai) && Number.isFinite(bi)) return ai - bi;
          const as = Number(a.checkpoint_step);
          const bs = Number(b.checkpoint_step);
          if (Number.isFinite(as) && Number.isFinite(bs)) return as - bs;
          return String(a.checkpoint).localeCompare(String(b.checkpoint));
        });
      const seen = new Set();
      const values = [];
      for (const row of rows) {
        if (seen.has(row.checkpoint)) continue;
        seen.add(row.checkpoint);
        values.push(row.checkpoint);
      }
      return values;
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
        (except === "checkpoint" || !selects.checkpoint.value || r.checkpoint === selects.checkpoint.value) &&
        (except === "target" || !selects.target.value || r.target === selects.target.value) &&
        (except === "geometry" || !selects.geometry.value || r.geometry === selects.geometry.value) &&
        (except === "layer" || !selects.layer.value || r.layer === selects.layer.value)
      );
    }

    // Coarse-to-fine hierarchy. Each selector is narrowed only by the
    // selectors above it, never below: Run stays a full list regardless of the
    // model/target/layer in view, so sparse runs never drop out of the picker.
    const selectOrder = ["run", "model", "checkpoint", "geometry", "target", "layer"];

    function refreshOptions() {
      const current = Object.fromEntries(Object.entries(selects).map(([k, s]) => [k, s.value]));
      selectOrder.forEach((key, idx) => {
        const pool = records.filter((r) =>
          selectOrder.slice(0, idx).every((k) => !selects[k].value || r[k] === selects[k].value),
        );
        const values = key === "checkpoint"
          ? checkpointValues(pool)
          : key === "layer"
            ? sortLayers(pool.map((r) => r[key]))
            : unique(pool.map((r) => r[key]));
        const options = key === "geometry"
          ? { includeAll: true, allLabel: "all geometries" }
          : {};
        fill(selects[key], values, current[key], options);
      });
      updateCheckpointPickerVisibility();
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

    function sbsCells(prefix, items) {
      return items.map((it, i) => `
        <div class="sbs-cell">
          <div class="sbs-title"><span class="chip-dot" style="--chip:${itemColor(i)}"></span>${escapeHtml(modelDisplay(it))} / ${escapeHtml(it.target)} &middot; L${escapeHtml(it.layer)}</div>
          <div class="chart sbs-chart" id="${prefix}-${i}"></div>
        </div>`).join("");
    }

    function renderShell(candidate, items) {
      const sweep = candidate.sweep || {};
      const sweepParts = [];
      if (sweep.range_policy) sweepParts.push(sweep.range_policy);
      if (sweep.grid_mode) sweepParts.push(`grid=${sweep.grid_mode}`);
      if (sweep.physical_low !== undefined && sweep.physical_high !== undefined) {
        sweepParts.push(`range=[${fmt(sweep.physical_low)}, ${fmt(sweep.physical_high)}]`);
      }
      const sweepText = sweepParts.length ? sweepParts.join(", ") : "n/a";
      const metricsLink = candidate.metrics_json ? `<a href="${escapeHtml(resolveDataUrl(candidate.metrics_json))}">metrics JSON</a>` : "";
      const plotDataLink = candidate.paths && candidate.paths.plot_data_json ? `<a href="${escapeHtml(resolveDataUrl(candidate.paths.plot_data_json))}">plot data JSON</a>` : "";
      const distanceDataLink = candidate.paths && candidate.paths.distance_data_json ? `<a href="${escapeHtml(resolveDataUrl(candidate.paths.distance_data_json))}">distance data JSON</a>` : "";
      const trackCount = tracksOf(items).length;
      const overlayNote = `${escapeHtml(modelDisplay(candidate))} / ${escapeHtml(candidate.target)}${items.length > 1 ? ` &middot; ${trackCount} track${trackCount === 1 ? "" : "s"}` : ""}`;
      return `
        <div class="layout">
          <section class="panel summary-panel">
            <h2 class="record-title">${escapeHtml(modelDisplay(candidate))} / ${escapeHtml(candidate.target)} / layer ${escapeHtml(candidate.layer)}</h2>
            <div class="record-meta">
              ${candidate.checkpoint ? `<div class="meta-row"><span>Checkpoint</span><span>${escapeHtml(candidate.checkpoint)}</span></div>` : ""}
              <div class="meta-row"><span>Geometry</span><span>${escapeHtml(candidate.geometry || "n/a")}</span></div>
              <div class="meta-row"><span>Sweep</span><span>${escapeHtml(sweepText)}</span></div>
              <div class="meta-row"><span>Selected geodesic k</span><span>${escapeHtml(fmt(candidate.metrics && candidate.metrics.selected_geodesic_k))}</span></div>
            </div>
            <div class="links">${metricsLink}${plotDataLink}${distanceDataLink}</div>
          </section>
          <section class="charts">
            <section class="panel chart-card">
              <details class="collapsible"${metricsCollapsed ? "" : " open"}>
                <summary class="chart-header">
                  <h3><span class="chev"></span>Metrics across layers</h3>
                  <p class="chart-note">${overlayNote}</p>
                  <p class="chart-copy">Each panel plots one manifold metric against layer depth, computed in Python over all stored grid points. With several items selected, one coloured curve is drawn per (model, target) track; markers sit on each track's selected layers. The headline number is the active selection's value. Panels auto-scale independently.</p>
                </summary>
                <div class="layer-metrics">${renderLayerMetricPanels(candidate)}</div>
              </details>
            </section>
            <div class="pair-row">
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
                <p class="chart-copy">Centroids ordered by the controlled target value, projected with browser-side PCA. Each panel auto-fits its own axes (PCA scale differs across models). Same-target panels are Procrustes-aligned (rotation/reflection only) to the first selection so orientations are comparable; panels that cannot be put in correspondence are badged "independent PCA". Color is the latent target coordinate; in 3D, drag to rotate and scroll to zoom.</p>
              </div>
              <div class="sbs-grid" id="centroid-grid">${sbsCells("centroid-chart", items)}</div>
            </section>
            <section class="panel chart-card">
              <div class="chart-header">
                <h3>Input signal</h3>
                <p class="chart-note" id="signal-note"></p>
                <p class="chart-copy">The exact waveform fed to the model for the selected grid point, generated in your browser from the controlled-slice spec (aiono sine / spike / gaussian / trend); no raw signal is stored. Drag the slider or press play to sweep the controlled factor: each input updates and a dark marker rides the matching centroid path to the right. The same grid index is shown across panels.</p>
              </div>
              <div class="sbs-grid" id="signal-grid">${sbsCells("signal-chart", items)}</div>
              <div class="signal-controls">
                <button type="button" class="signal-play" id="signal-play" aria-label="play or pause">▶</button>
                <input type="range" id="signal-slider" min="0" max="0" value="0" aria-label="sweep position">
                <span class="readout" id="signal-readout"></span>
              </div>
            </section>
            </div>
            <section class="panel chart-card">
              <details id="distance-details" class="collapsible"${distanceBlockOpen ? " open" : ""}>
                <summary class="chart-header">
                  <h3><span class="chev"></span>Distance scatter / heatmap</h3>
                  <p class="chart-note" id="distance-note"></p>
                  <p class="chart-copy">Pairwise distance matrices are stored separately and loaded only when this block is opened. Scatter uses target-space distance on X and representation distance on Y; heatmaps show latent / direct / geodesic matrices with per-panel color scales.</p>
                </summary>
                <div class="distance-section">
                  <div class="distance-header">
                    <h4>Distance scatter</h4>
                    <p class="chart-note" id="scatter-note"></p>
                  </div>
                  <div class="sbs-grid" id="scatter-grid">${sbsCells("scatter-chart", items)}</div>
                </div>
                <div class="distance-section">
                  <div class="distance-header">
                    <h4>Distance heatmap</h4>
                    <p class="chart-note" id="heatmap-note"></p>
                  </div>
                  <div class="sbs-grid heatmap-grid" id="heatmap-grid">${sbsCells("heatmap-chart", items)}</div>
                </div>
              </details>
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

    // --- small linear algebra for shared-frame (Procrustes) centroid alignment ---
    function jacobiEigSym(input, n) {
      const S = input.map((row) => row.slice());
      const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
      for (let sweep = 0; sweep < 60; sweep += 1) {
        let off = 0;
        for (let p = 0; p < n; p += 1) for (let q = p + 1; q < n; q += 1) off += S[p][q] * S[p][q];
        if (off < 1e-22) break;
        for (let p = 0; p < n; p += 1) {
          for (let q = p + 1; q < n; q += 1) {
            if (Math.abs(S[p][q]) < 1e-18) continue;
            const phi = 0.5 * Math.atan2(2 * S[p][q], S[q][q] - S[p][p]);
            const c = Math.cos(phi);
            const s = Math.sin(phi);
            for (let i = 0; i < n; i += 1) {
              const sip = S[i][p];
              const siq = S[i][q];
              S[i][p] = c * sip - s * siq;
              S[i][q] = s * sip + c * siq;
            }
            for (let i = 0; i < n; i += 1) {
              const spi = S[p][i];
              const sqi = S[q][i];
              S[p][i] = c * spi - s * sqi;
              S[q][i] = s * spi + c * sqi;
            }
            for (let i = 0; i < n; i += 1) {
              const vip = V[i][p];
              const viq = V[i][q];
              V[i][p] = c * vip - s * viq;
              V[i][q] = s * vip + c * viq;
            }
          }
        }
      }
      return { values: Array.from({ length: n }, (_, i) => S[i][i]), vectors: V };
    }

    function procrustesRotation(A, B) {
      // k x k rotation/reflection R minimizing ||A R - B|| over matched centered rows
      const k = A[0] ? A[0].length : 0;
      if (!k || A.length < 2) return null;
      const M = Array.from({ length: k }, () => new Array(k).fill(0));
      for (let p = 0; p < A.length; p += 1) {
        for (let i = 0; i < k; i += 1) for (let j = 0; j < k; j += 1) M[i][j] += A[p][i] * B[p][j];
      }
      const MtM = Array.from({ length: k }, () => new Array(k).fill(0));
      for (let i = 0; i < k; i += 1) for (let j = 0; j < k; j += 1) {
        let acc = 0;
        for (let t = 0; t < k; t += 1) acc += M[t][i] * M[t][j];
        MtM[i][j] = acc;
      }
      const { values, vectors: V } = jacobiEigSym(MtM, k);
      const sigma = values.map((v) => Math.sqrt(Math.max(v, 0)));
      const U = Array.from({ length: k }, () => new Array(k).fill(0));
      for (let c = 0; c < k; c += 1) {
        const s = sigma[c] > 1e-9 ? sigma[c] : 1e-9;
        for (let r = 0; r < k; r += 1) {
          let acc = 0;
          for (let t = 0; t < k; t += 1) acc += M[r][t] * V[t][c];
          U[r][c] = acc / s;
        }
      }
      const R = Array.from({ length: k }, () => new Array(k).fill(0));
      for (let i = 0; i < k; i += 1) for (let j = 0; j < k; j += 1) {
        let acc = 0;
        for (let c = 0; c < k; c += 1) acc += U[i][c] * V[j][c];
        R[i][j] = acc;
      }
      return R;
    }

    function meanRows(rows, k) {
      const m = new Array(k).fill(0);
      for (const r of rows) for (let i = 0; i < k; i += 1) m[i] += r[i];
      return m.map((v) => v / Math.max(rows.length, 1));
    }

    function applyRotation(points, R, aMean, bMean) {
      const k = R.length;
      return points.map((p) => {
        const out = new Array(k).fill(0);
        for (let j = 0; j < k; j += 1) {
          let acc = 0;
          for (let i = 0; i < k; i += 1) acc += (p[i] - aMean[i]) * R[i][j];
          out[j] = acc + bMean[j];
        }
        return out;
      });
    }

    function centroidEmbedding(plotData, k) {
      const centroids = plotData && (plotData.path_centroids || plotData.centroids);
      if (!centroids || !centroids.length) return null;
      const emb = k === 3 ? project3d(centroids) : project2d(centroids);
      const coords = (plotData.path_centroid_coordinates || plotData.centroid_coordinates || []).map(Number);
      const counts = plotData.path_centroid_counts || plotData.centroid_counts || [];
      return { emb: emb.map((p) => p.slice()), coords, counts };
    }

    function alignEmbeddings(items, plotDatas, k) {
      const out = items.map((it, i) => {
        const pd = plotDatas[i];
        if (!pd || pd.__error) return null;
        const e = centroidEmbedding(pd, k);
        if (!e) return null;
        const gridMode = it.sweep && it.sweep.grid_mode ? String(it.sweep.grid_mode) : "";
        return { item: it, emb: e.emb, coords: e.coords, counts: e.counts, gridMode, aligned: false };
      });
      // Single primary reference = first available item. Only same-target items
      // can correspond, so cross-target panels stay in their own PCA frame.
      const ref = out.find((o) => o);
      if (!ref) return out;
      const refIndex = new Map();
      ref.coords.forEach((c, idx) => { if (Number.isFinite(c)) refIndex.set(c.toFixed(6), idx); });
      out.forEach((o) => {
        if (!o) return;
        if (o === ref) { o.aligned = true; return; }
        if (o.item.target_name !== ref.item.target_name) { o.aligned = false; return; }
        let A = [];
        let B = [];
        o.coords.forEach((c, idx) => {
          const ri = Number.isFinite(c) ? refIndex.get(c.toFixed(6)) : undefined;
          if (ri !== undefined) { A.push(o.emb[idx].slice()); B.push(ref.emb[ri].slice()); }
        });
        // Positional fallback only when the controlled grid is the same shape and
        // binning (equal length, same grid mode); never mix linear vs signed_log.
        if (A.length < 2 && o.emb.length === ref.emb.length && o.emb.length >= 2 && o.gridMode === ref.gridMode) {
          A = o.emb.map((p) => p.slice());
          B = ref.emb.map((p) => p.slice());
        }
        if (A.length < 2) { o.aligned = false; return; }
        const aMean = meanRows(A, k);
        const bMean = meanRows(B, k);
        const Ac = A.map((r) => r.map((v, j) => v - aMean[j]));
        const Bc = B.map((r) => r.map((v, j) => v - bMean[j]));
        const R = procrustesRotation(Ac, Bc);
        if (!R) { o.aligned = false; return; }
        o.emb = applyRotation(o.emb, R, aMean, bMean);
        o.aligned = true;
      });
      return out;
    }

    function renderCentroidPanels(items, plotDatas) {
      // Mode switches alternate a 2D cartesian grid and a 3D WebGL scene on the
      // same containers, so start each centroid panel from a fresh instance.
      items.forEach((_, i) => {
        const cid = `centroid-chart-${i}`;
        const existing = chartInstances[cid];
        if (existing && !existing.isDisposed()) existing.dispose();
        chartInstances[cid] = null;
        const element = document.getElementById(cid);
        if (element) element.innerHTML = "";
      });
      if (centroidMode === "3d" && !glReady()) { centroidMode = "2d"; updateCentroidToggle(); }
      const use3d = centroidMode === "3d" && glReady();
      const k = use3d ? 3 : 2;
      const aligned = alignEmbeddings(items, plotDatas, k);
      if (sideCache) { sideCache.aligned = aligned; sideCache.use3d = use3d; }
      // Axes auto-fit per panel: PCA scale differs radically between models, so a
      // shared range would squash most panels.
      const multi = items.length > 1;
      items.forEach((it, i) => {
        const id = `centroid-chart-${i}`;
        const pd = plotDatas[i];
        if (pd && pd.__error) { setChartStatus(id, `Could not load plot data. Serve this directory over a static HTTP server. ${pd.__error}`, "status warning"); return; }
        const o = aligned[i];
        if (!o) { setChartStatus(id, "No centroid path data available for this layer."); return; }
        const badge = multi && !o.aligned ? "independent PCA" : "";
        if (use3d) renderCentroid3DPanel(id, it, o, badge);
        else renderCentroid2DPanel(id, it, o, badge);
      });
      applyCentroidCursors(signalIndex);
    }

    function renderCentroid2DPanel(id, item, o, badge) {
      const chart = getChart(id);
      if (!chart) return;
      const data = o.emb.map((p, idx) => [p[0], p[1], Number(o.coords[idx]), idx, o.counts[idx] || 0]);
      const lineData = item.geometry === "circle" && data.length > 2 ? [...data, data[0]] : data;
      const cv = finiteValues(data.map((d) => d[2]));
      const low = cv.length ? Math.min(...cv) : 0;
      const high = cv.length ? Math.max(...cv) : 1;
      chart.setOption({
        animation: false,
        grid: { left: 52, right: 16, top: 28, bottom: 52 },
        tooltip: {
          trigger: "item",
          formatter: (params) => {
            const v = params.value || [];
            return [`<strong>grid ${v[3]}</strong>`, `latent coordinate: ${fmt(v[2])}`, `count: ${fmt(v[4])}`].join("<br>");
          },
        },
        visualMap: {
          type: "continuous", min: low, max: high, dimension: 2,
          orient: "horizontal", left: "center", bottom: 4,
          text: ["high", "low"], inRange: { color: ["#2563eb", "#14b8a6", "#f59e0b", "#dc2626"] },
        },
        xAxis: { type: "value", scale: true, name: "PCA 1", nameLocation: "middle", nameGap: 22, axisLine: { lineStyle: { color: "#94a3b8" } }, splitLine: { lineStyle: { color: "#edf2f7" } } },
        yAxis: { type: "value", scale: true, name: "PCA 2", nameLocation: "middle", nameGap: 36, axisLine: { lineStyle: { color: "#94a3b8" } }, splitLine: { lineStyle: { color: "#edf2f7" } } },
        series: [
          { type: "line", data: lineData, showSymbol: false, lineStyle: { width: 2, color: "#0f766e" }, emphasis: { disabled: true } },
          { type: "scatter", data, symbolSize: 6 },
        ],
        graphic: badge ? [{ id: "badge", type: "text", right: 8, top: 4, style: { text: badge, fill: "#9a6700", font: "600 11px IBM Plex Sans, sans-serif" } }] : [],
      }, true);
    }

    function renderCentroid3DPanel(id, item, o, badge) {
      const chart = getChart(id);
      if (!chart) return;
      const data = o.emb.map((p, idx) => [p[0], p[1], p[2], Number(o.coords[idx]), idx, o.counts[idx] || 0]);
      const lineData = (item.geometry === "circle" && data.length > 2 ? [...data, data[0]] : data).map((d) => [d[0], d[1], d[2]]);
      const cv = finiteValues(data.map((d) => d[3]));
      const low = cv.length ? Math.min(...cv) : 0;
      const high = cv.length ? Math.max(...cv) : 1;
      try {
        chart.setOption({
          animation: false,
          tooltip: {
            formatter: (params) => {
              const v = params.value || [];
              return [`<strong>grid ${v[4]}</strong>`, `latent coordinate: ${fmt(v[3])}`, `count: ${fmt(v[5])}`].join("<br>");
            },
          },
          visualMap: {
            type: "continuous", min: low, max: high, dimension: 3, seriesIndex: 1,
            orient: "horizontal", left: "center", bottom: 2,
            text: ["high", "low"], inRange: { color: ["#2563eb", "#14b8a6", "#f59e0b", "#dc2626"] },
          },
          xAxis3D: { type: "value", name: "PCA 1" },
          yAxis3D: { type: "value", name: "PCA 2" },
          zAxis3D: { type: "value", name: "PCA 3" },
          grid3D: {
            boxWidth: 90, boxDepth: 90, boxHeight: 68, top: -20,
            axisLine: { lineStyle: { color: "#94a3b8" } },
            splitLine: { lineStyle: { color: "#e2e8f0" } },
            viewControl: { autoRotate: false, distance: 210 },
          },
          series: [
            { type: "line3D", data: lineData, lineStyle: { width: 3, color: "#0f766e", opacity: 0.85 } },
            { type: "scatter3D", data, symbolSize: 8, itemStyle: { opacity: 0.95 } },
          ],
          graphic: badge ? [{ id: "badge", type: "text", right: 8, top: 4, style: { text: badge, fill: "#9a6700", font: "600 11px IBM Plex Sans, sans-serif" } }] : [],
        }, true);
      } catch (error) {
        centroidMode = "2d";
        updateCentroidToggle();
        renderCentroid2DPanel(id, item, o, badge);
      }
    }

    function drawCentroids() {
      if (sideCache) renderCentroidPanels(sideCache.items, sideCache.plotDatas);
    }

    // ---------- input-signal scrubber (waveform generated in-browser from the controlled-slice spec) ----------
    function signalNum(value, fallback) {
      const n = Number(value);
      return Number.isFinite(n) ? n : fallback;
    }

    function signalSpecFromPayload(payload) {
      if (!payload || payload.__error) return null;
      const target = payload.target || {};
      const component = String(target.component || "");
      if (!component) return null;
      const tsm = payload.train_slice_manifest || {};
      const fixedAll = tsm.fixed_values || {};
      const dm = tsm.dataset_manifest || {};
      const adapter = (payload.model || {}).adapter || {};
      let seqLen = adapter.benchmark_sequence_length;
      if (seqLen === undefined || seqLen === null) seqLen = dm.channel_size;
      const fixed = fixedAll[component];
      return {
        component,
        param: String(target.parameter || ""),
        coordinate_name: String(target.coordinate_name || ""),
        geometry: String(target.geometry || ""),
        seq_len: signalNum(seqLen, 512),
        fs: signalNum(dm.sampling_frequency, 500),
        fixed: fixed && typeof fixed === "object" ? fixed : {},
      };
    }

    function signalPhysical(spec, coord) {
      // centroid_coordinates hold the manifold coordinate; for frequency that is log-Hz.
      if (spec.coordinate_name === "log_frequency_hz") return Math.exp(coord);
      return coord;
    }

    function genSignal(spec, coord) {
      const L = Math.max(2, Math.round(signalNum(spec.seq_len, 512)));
      const fs = signalNum(spec.fs, 500);
      const fx = spec.fixed || {};
      const y = new Array(L);
      const c = spec.component;
      if (c === "sine") {
        const pv = signalPhysical(spec, coord);
        const A = spec.param === "amplitude" ? pv : signalNum(fx.amplitude, 1);
        const f = spec.param === "frequency_hz" ? pv : signalNum(fx.frequency_hz, 1);
        const ph = spec.param === "phase" ? pv : signalNum(fx.phase, 0);
        const off = signalNum(fx.offset, 0);
        for (let i = 0; i < L; i += 1) y[i] = A * Math.sin(2 * Math.PI * f * (i / fs) + ph) + off;
      } else if (c === "spike") {
        const amp = signalNum(fx.amplitude, 1);
        const t0 = Math.round(coord * L);
        for (let i = 0; i < L; i += 1) y[i] = i === t0 ? amp : 0;
      } else if (c === "gaussian") {
        const amp = signalNum(fx.amplitude, 1);
        const sigma = Math.max(1e-6, signalNum(fx.sigma_sec, 0.02) * fs);
        const ctr = coord * L;
        for (let i = 0; i < L; i += 1) { const d = (i - ctr) / sigma; y[i] = amp * Math.exp(-0.5 * d * d); }
      } else if (c === "linear_trend") {
        const b = signalNum(fx.intercept, 0);
        for (let i = 0; i < L; i += 1) y[i] = coord * (i / (L - 1) - 0.5) + b;
      } else {
        return null;
      }
      return y;
    }

    function signalBounds(spec, sweep) {
      const c = spec.component;
      const fx = spec.fixed || {};
      if (c === "sine") {
        const A = spec.param === "amplitude"
          ? Math.abs(signalNum(sweep && sweep.physical_high, signalNum(fx.amplitude, 1)))
          : Math.abs(signalNum(fx.amplitude, 1));
        const off = signalNum(fx.offset, 0);
        const b = A * 1.12 + Math.abs(off) + 1e-9;
        return [off - b, off + b];
      }
      if (c === "spike" || c === "gaussian") {
        const amp = Math.abs(signalNum(fx.amplitude, 1)) || 1;
        return [-0.12 * amp, 1.15 * amp];
      }
      if (c === "linear_trend") {
        const m = Math.max(Math.abs(signalNum(sweep && sweep.physical_low, 0)), Math.abs(signalNum(sweep && sweep.physical_high, 1))) * 0.5 * 1.05 + 1e-9;
        return [-m, m];
      }
      return null;
    }

    function signalReadout(spec, coord) {
      const c = spec.component;
      const p = spec.param;
      if (c === "sine" && p === "phase") return `phase = ${fmt(coord)} rad`;
      if (c === "sine" && p === "frequency_hz") return `f = ${fmt(Math.exp(coord))} Hz`;
      if (c === "sine" && p === "amplitude") return `amplitude = ${fmt(coord)}`;
      if (c === "spike") return `spike @ t = ${fmt((coord * spec.seq_len) / spec.fs)} s`;
      if (c === "gaussian") return `centre @ t = ${fmt((coord * spec.seq_len) / spec.fs)} s`;
      if (c === "linear_trend") return `slope = ${fmt(coord)}`;
      return fmt(coord);
    }

    function coordsOf(plotData) {
      if (!plotData || plotData.__error) return [];
      const raw = plotData.centroid_coordinates || plotData.path_centroid_coordinates || [];
      return raw.map(Number);
    }

    function renderSignalPanels(items, plotDatas, idx) {
      items.forEach((item, i) => {
        const id = `signal-chart-${i}`;
        const pd = plotDatas[i];
        if (pd && pd.__error) { setChartStatus(id, "Plot data unavailable for this layer.", "status warning"); return; }
        const spec = item.signal_spec;
        const coords = coordsOf(pd);
        if (!spec || !spec.component) { setChartStatus(id, "Input-signal preview is not available for this target."); return; }
        if (!coords.length) { setChartStatus(id, "No grid coordinates in this artifact."); return; }
        const j = Math.max(0, Math.min(coords.length - 1, idx | 0));
        const y = genSignal(spec, coords[j]);
        if (!y) { setChartStatus(id, `Input-signal preview is not available for component "${escapeHtml(spec.component)}".`); return; }
        const chart = getChart(id);
        if (!chart) return;
        const L = y.length;
        const fs = signalNum(spec.fs, 500);
        const data = new Array(L);
        for (let k = 0; k < L; k += 1) data[k] = [k / fs, y[k]];
        const bounds = signalBounds(spec, item.sweep);
        const series = {
          type: "line", data, showSymbol: false,
          smooth: spec.component === "sine" || spec.component === "gaussian",
          sampling: "lttb", lineStyle: { width: 1.8, color: "#0f766e" },
        };
        if (spec.component === "gaussian" || spec.component === "spike") series.areaStyle = { color: "rgba(15,118,110,0.12)" };
        chart.setOption({
          animation: false,
          grid: { left: 52, right: 12, top: 12, bottom: 34 },
          tooltip: { trigger: "axis", appendToBody: true, valueFormatter: (v) => fmt(v) },
          xAxis: { type: "value", min: 0, max: L / fs, name: "time (s)", nameLocation: "middle", nameGap: 21, axisLabel: { fontSize: 9 }, axisLine: { lineStyle: { color: "#94a3b8" } }, splitLine: { lineStyle: { color: "#edf2f7" } } },
          yAxis: bounds
            ? { type: "value", min: bounds[0], max: bounds[1], axisLabel: { fontSize: 9 }, axisLine: { lineStyle: { color: "#94a3b8" } }, splitLine: { lineStyle: { color: "#edf2f7" } } }
            : { type: "value", scale: true, axisLabel: { fontSize: 9 }, splitLine: { lineStyle: { color: "#edf2f7" } } },
          series: [series],
        }, true);
      });
    }

    function applyCentroidCursors(idx) {
      if (!sideCache || !Array.isArray(sideCache.aligned)) return;
      const use3d = sideCache.use3d;
      sideCache.aligned.forEach((o, i) => {
        const id = `centroid-chart-${i}`;
        const chart = chartInstances[id];
        if (!chart || chart.isDisposed() || !o || !Array.isArray(o.emb) || !o.emb.length) return;
        const j = Math.max(0, Math.min(o.emb.length - 1, idx | 0));
        const pt = o.emb[j];
        if (!pt) return;
        if (use3d) {
          chart.setOption({ series: [{ id: "cursor3d", type: "scatter3D", data: [[pt[0], pt[1], pt[2]]], symbolSize: 15, itemStyle: { color: "#0f172a" } }] });
        } else {
          const px = chart.convertToPixel({ gridIndex: 0 }, [pt[0], pt[1]]);
          if (px) chart.setOption({ graphic: [{ type: "circle", id: "cursor", z: 100, position: px, shape: { r: 6.5 }, style: { fill: "#0f172a", stroke: "#fff", lineWidth: 2.5 } }] });
        }
      });
    }

    function updateSignalReadout(items, plotDatas, idx) {
      const el = document.getElementById("signal-readout");
      if (!el) return;
      for (let i = 0; i < items.length; i += 1) {
        const spec = items[i].signal_spec;
        const coords = coordsOf(plotDatas[i]);
        if (spec && spec.component && coords.length) {
          const j = Math.max(0, Math.min(coords.length - 1, idx | 0));
          el.textContent = `grid ${j} · ${signalReadout(spec, Number(coords[j]))}`;
          return;
        }
      }
      el.textContent = `grid ${idx | 0}`;
    }

    function stopSignalPlayback() {
      signalPlaying = false;
      if (signalTimer) { clearTimeout(signalTimer); signalTimer = null; }
      if (signalRaf) { cancelAnimationFrame(signalRaf); signalRaf = null; }
    }

    function setupSignalPanel(items, plotDatas) {
      const slider = document.getElementById("signal-slider");
      const play = document.getElementById("signal-play");
      if (!slider) return;
      stopSignalPlayback();
      let N = 0;
      plotDatas.forEach((pd) => { const c = coordsOf(pd); if (c.length > N) N = c.length; });
      if (N < 1) N = 1;
      signalIndex = Math.max(0, Math.min(N - 1, signalIndex | 0));
      slider.min = 0;
      slider.max = N - 1;
      slider.value = signalIndex;
      const geometry = (activeCandidate() || items[0] || {}).geometry || "interval";
      const step = Math.max(1, Math.floor(N / 256));
      let dir = 1;
      const apply = (v) => {
        signalIndex = Math.max(0, Math.min(N - 1, v | 0));
        renderSignalPanels(items, plotDatas, signalIndex);
        applyCentroidCursors(signalIndex);
        updateSignalReadout(items, plotDatas, signalIndex);
      };
      apply(signalIndex);
      slider.oninput = () => {
        const v = +slider.value;
        if (signalRaf) cancelAnimationFrame(signalRaf);
        signalRaf = requestAnimationFrame(() => apply(v));
      };
      const tick = () => {
        if (!signalPlaying) return;
        let v = signalIndex + dir * step;
        if (geometry === "circle") { if (v >= N) v = 0; }
        else if (v >= N - 1) { v = N - 1; dir = -1; }
        else if (v <= 0) { v = 0; dir = 1; }
        slider.value = v;
        apply(v);
        signalTimer = setTimeout(() => requestAnimationFrame(tick), 33);
      };
      if (play) {
        play.textContent = "▶";
        play.onclick = () => {
          signalPlaying = !signalPlaying;
          play.textContent = signalPlaying ? "⏸" : "▶";
          if (signalPlaying) { if (geometry !== "circle" && signalIndex >= N - 1) { signalIndex = 0; dir = 1; } tick(); }
          else stopSignalPlayback();
        };
      }
    }

    function renderScatterPanels(items, plotDatas) {
      const prepared = items.map((it, i) => {
        const pd = plotDatas[i];
        if (!pd || pd.__error) return null;
        const latent = numericMatrix(pd.latent_distance);
        const linearData = upperTriangularPairs(latent, numericMatrix(pd.linear_distance));
        const graphData = pd.geodesic_distance ? upperTriangularPairs(latent, numericMatrix(pd.geodesic_distance)) : [];
        return { linearData, graphData, metrics: it.metrics || {} };
      });
      items.forEach((it, i) => {
        const id = `scatter-chart-${i}`;
        const pd = plotDatas[i];
        if (pd && pd.__error) { setChartStatus(id, `Could not load plot data. ${pd.__error}`, "status warning"); return; }
        const pr = prepared[i];
        if (!pr) { setChartStatus(id, "No distance data available for this layer."); return; }
        const chart = getChart(id);
        if (!chart) return;
        chart.setOption({
          animation: false,
          grid: { left: 58, right: 16, top: 30, bottom: 46 },
          legend: { top: 2, right: 6, textStyle: { fontSize: 10 } },
          tooltip: {
            trigger: "item",
            formatter: (params) => {
              const v = params.value || [];
              return [`<strong>${escapeHtml(params.seriesName)}</strong>`, `target-space distance: ${fmt(v[0])}`, `representation distance: ${fmt(v[1])}`].join("<br>");
            },
          },
          xAxis: { type: "value", scale: true, name: "target-space distance", nameLocation: "middle", nameGap: 26, splitLine: { lineStyle: { color: "#edf2f7" } } },
          yAxis: { type: "value", scale: true, name: "repr distance", nameLocation: "middle", nameGap: 40, splitLine: { lineStyle: { color: "#edf2f7" } } },
          series: [
            { name: `linear rho=${fmt(pr.metrics.spearman_latent_vs_linear)}`, type: "scatter", data: pr.linearData, symbolSize: 3, large: pr.linearData.length > 2000, itemStyle: { color: "#2563eb", opacity: 0.45 } },
            { name: `geodesic rho=${fmt(pr.metrics.spearman_latent_vs_geodesic)}`, type: "scatter", data: pr.graphData, symbolSize: 3, large: pr.graphData.length > 2000, itemStyle: { color: "#dc2626", opacity: 0.45 } },
          ],
        }, true);
      });
    }

    function renderHeatmapPanels(items, plotDatas) {
      const prep = items.map((it, i) => {
        const pd = plotDatas[i];
        if (!pd || pd.__error) return null;
        const matrices = [
          { name: "latent", matrix: numericMatrix(pd.latent_distance) },
          { name: "linear", matrix: numericMatrix(pd.linear_distance) },
        ];
        if (pd.geodesic_distance) matrices.push({ name: "graph", matrix: numericMatrix(pd.geodesic_distance) });
        return matrices;
      });
      items.forEach((it, i) => {
        const id = `heatmap-chart-${i}`;
        const pd = plotDatas[i];
        if (pd && pd.__error) { setChartStatus(id, `Could not load plot data. ${pd.__error}`, "status warning"); return; }
        const matrices = prep[i];
        if (!matrices) { setChartStatus(id, "No distance data available for this layer."); return; }
        // Per-panel colour scale: distance magnitudes differ across models.
        const [low, high] = matrixMinMax(matrices.map((m) => m.matrix));
        const chart = getChart(id);
        if (!chart) return;
        const count = matrices.length;
        const gap = 4;
        const width = (86 - gap * (count - 1)) / count;
        const grids = matrices.map((_, idx) => ({ left: `${7 + idx * (width + gap)}%`, top: 40, width: `${width}%`, height: 195 }));
        const axes = matrices.map((m, idx) => {
          const n = m.matrix.length;
          return {
            xAxis: { type: "category", gridIndex: idx, data: Array.from({ length: n }, (_, j) => j), axisLabel: { hideOverlap: true, fontSize: 9 } },
            yAxis: { type: "category", gridIndex: idx, inverse: true, data: Array.from({ length: n }, (_, j) => j), axisLabel: { hideOverlap: true, fontSize: 9 } },
          };
        });
        chart.setOption({
          animation: false,
          tooltip: {
            position: "top",
            formatter: (params) => {
              const v = params.value || [];
              return [`<strong>${escapeHtml(params.seriesName)}</strong>`, `row ${v[1]} col ${v[0]}`, `distance ${fmt(v[2])}`].join("<br>");
            },
          },
          visualMap: { min: low, max: high, calculable: true, orient: "horizontal", left: "center", bottom: 0, itemHeight: 60, inRange: { color: ["#f8fafc", "#bfdbfe", "#14b8a6", "#f59e0b", "#dc2626"] } },
          grid: grids,
          xAxis: axes.map((a) => a.xAxis),
          yAxis: axes.map((a) => a.yAxis),
          series: matrices.map((m, idx) => ({ name: m.name, type: "heatmap", xAxisIndex: idx, yAxisIndex: idx, data: heatmapData(m.matrix), progressive: 8000 })),
          graphic: matrices.map((m, idx) => ({ type: "text", left: `${7 + idx * (width + gap)}%`, top: 20, style: { text: m.name, fill: "#334155", font: "600 11px IBM Plex Sans, sans-serif" } })),
        }, true);
      });
    }

    function layerRecords(record) {
      const seen = new Set();
      return records
        .filter((r) => (
          r.run === record.run &&
          r.model === record.model &&
          (r.checkpoint || "") === (record.checkpoint || "") &&
          r.target === record.target
        ))
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

    function renderLayerMetricsCharts(items) {
      const trackList = tracksOf(items);
      const trackRows = trackList.map((t) => ({ track: t, rows: layerRecords(t.record) }));
      const numeric = trackRows.every((tr) => tr.rows.every((r) => Number.isFinite(r.layerNum)));
      const categories = numeric
        ? null
        : [...new Set(trackRows.flatMap((tr) => tr.rows.map((r) => r.layer)))].sort((a, b) => String(a).localeCompare(String(b)));
      const xOf = (row) => (numeric ? row.layerNum : categories.indexOf(row.layer));
      for (const key of metricOrder) {
        const id = `metric-chart-${key}`;
        let anyFinite = false;
        const series = trackRows.map(({ track, rows }) => {
          const points = rows.map((row) => {
            const v = finiteNumber(row.metrics[key]);
            return [xOf(row), Number.isFinite(v) ? v : null];
          });
          if (points.some((p) => p[1] !== null)) anyFinite = true;
          const markPts = track.layers.map((L) => {
            const row = rows.find((r) => String(r.layer) === String(L));
            if (!row) return null;
            const v = finiteNumber(row.metrics[key]);
            return Number.isFinite(v) ? { coord: [xOf(row), v] } : null;
          }).filter(Boolean);
          return {
            name: trackLabel(track),
            type: "line",
            data: points,
            color: track.color,
            showSymbol: true,
            symbolSize: 3,
            connectNulls: false,
            lineStyle: { width: 1.8, color: track.color },
            itemStyle: { color: track.color },
            markPoint: markPts.length ? { symbol: "circle", symbolSize: 9, data: markPts, label: { show: false }, itemStyle: { color: track.color, borderColor: "#0f172a", borderWidth: 1 } } : undefined,
          };
        });
        if (!anyFinite) { setChartStatus(id, "Not available for this target."); continue; }
        const chart = getChart(id);
        if (!chart) {
          setChartStatus(id, "Apache ECharts did not load. Check network access to the CDN and refresh.", "status warning");
          continue;
        }
        chart.setOption({
          animation: false,
          grid: { left: 56, right: 12, top: 12, bottom: 30 },
          tooltip: {
            trigger: "axis",
            appendToBody: true,
            formatter: (params) => {
              if (!Array.isArray(params) || !params.length) return "";
              const head = `layer ${fmt(params[0].value[0])}`;
              const lines = params
                .filter((p) => p.value && p.value[1] !== null && p.value[1] !== undefined)
                .map((p) => `<span style="color:${p.color}">●</span> ${escapeHtml(p.seriesName)}: ${fmt(p.value[1])}`);
              return [head, ...lines].join("<br>");
            },
          },
          xAxis: {
            type: numeric ? "value" : "category",
            name: "layer",
            nameLocation: "middle",
            nameGap: 19,
            minInterval: 1,
            data: numeric ? undefined : categories,
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
          series,
        }, true);
      }
    }

    function updateNotes(items) {
      const n = items.length;
      const note = `${n} panel${n === 1 ? "" : "s"}`;
      for (const nid of ["centroid-note", "scatter-note", "heatmap-note"]) {
        const el = document.getElementById(nid);
        if (el) el.textContent = note;
      }
      const distanceNote = document.getElementById("distance-note");
      if (distanceNote) distanceNote.textContent = distanceBlockOpen ? note : `${note}; loads on open`;
    }

    function loadPlotData(path) {
      if (!path) return Promise.resolve(null);
      if (plotDataCache.has(path)) return plotDataCache.get(path);
      const promise = fetchJson(path)
        .catch((error) => ({ __error: String(error) }));
      plotDataCache.set(path, promise);
      return promise;
    }

    function loadDistanceData(path) {
      if (!path) return Promise.resolve(null);
      if (distanceDataCache.has(path)) return distanceDataCache.get(path);
      const promise = fetchJson(path)
        .catch((error) => ({ __error: String(error) }));
      distanceDataCache.set(path, promise);
      return promise;
    }

    function hasInlineDistanceData(plotData) {
      return !!(
        plotData &&
        !plotData.__error &&
        plotData.latent_distance &&
        plotData.linear_distance
      );
    }

    function distanceDataPath(item, plotData) {
      if (item && item.paths && item.paths.distance_data_json) return item.paths.distance_data_json;
      if (
        item &&
        item.paths &&
        typeof item.paths.plot_data_json === "string" &&
        item.paths.plot_data_json.endsWith("_plot_data.json")
      ) {
        const suffix = "_plot_data.json";
        return item.paths.plot_data_json.slice(0, -suffix.length) + "_distance_data.json";
      }
      if (plotData && typeof plotData.distance_data_json === "string") return plotData.distance_data_json;
      return null;
    }

    function loadDistanceDataForItem(item, plotData) {
      if (plotData && plotData.__error) return Promise.resolve(plotData);
      if (hasInlineDistanceData(plotData)) return Promise.resolve(plotData);
      const path = distanceDataPath(item, plotData);
      if (!path) return Promise.resolve({ __error: "No distance data JSON path in this artifact." });
      return loadDistanceData(path);
    }

    function setDistanceStatus(items, message, className = "status") {
      items.forEach((_, i) => {
        setChartStatus(`scatter-chart-${i}`, message, className);
        setChartStatus(`heatmap-chart-${i}`, message, className);
      });
    }

    async function renderDistancePanels(items, plotDatas, token) {
      const details = document.getElementById("distance-details");
      if (!details || !details.open) return;
      setDistanceStatus(items, "Loading distance matrices...");
      const distanceDatas = await Promise.all(
        items.map((it, i) => loadDistanceDataForItem(it, plotDatas[i])),
      );
      if (token !== renderToken) return;
      if (!details.open) return;
      if (sideCache) sideCache.distanceDatas = distanceDatas;
      renderScatterPanels(items, distanceDatas);
      renderHeatmapPanels(items, distanceDatas);
      updateNotes(items);
    }

    function attachDistanceToggle(token) {
      const details = document.getElementById("distance-details");
      if (!details) return;
      details.addEventListener("toggle", () => {
        distanceBlockOpen = details.open;
        updateNotes(sideCache ? sideCache.items : effectiveItems());
        if (details.open && sideCache) {
          renderDistancePanels(sideCache.items, sideCache.plotDatas, token);
        } else if (!details.open && sideCache) {
          sideCache.items.forEach((_, i) => {
            for (const prefix of ["scatter-chart", "heatmap-chart"]) {
              const id = `${prefix}-${i}`;
              const chart = chartInstances[id];
              if (chart && !chart.isDisposed()) chart.dispose();
              delete chartInstances[id];
            }
          });
        }
      });
    }

    async function renderSideBySide(items, token) {
      if (!window.echarts) {
        items.forEach((_, i) => {
          for (const pre of ["centroid-chart", "scatter-chart", "heatmap-chart"]) {
            setChartStatus(`${pre}-${i}`, "Apache ECharts did not load. Check network access to the CDN and refresh.", "status warning");
          }
        });
        return;
      }
      const plotDatas = await Promise.all(items.map((it) => loadPlotData(it.paths && it.paths.plot_data_json)));
      if (token !== renderToken) return;
      sideCache = { items, plotDatas, distanceDatas: null, aligned: null, use3d: false };
      updateNotes(items);
      renderCentroidPanels(items, plotDatas);
      setupSignalPanel(items, plotDatas);
      if (distanceBlockOpen) renderDistancePanels(items, plotDatas, token);
    }

    function render() {
      const token = ++renderToken;
      refreshOptions();
      renderComparison();
      disposeCharts();
      stopSignalPlayback();
      signalIndex = 0;
      const content = document.getElementById("content");
      if (!records.length) {
        content.innerHTML = '<div class="empty panel">No manifold metrics found.</div>';
        return;
      }
      const candidate = activeCandidate();
      const items = effectiveItems();
      if (!items.length) {
        content.innerHTML = '<div class="empty panel">No records match the current filters.</div>';
        return;
      }
      const loadItems = [candidate, ...items].filter(Boolean);
      const pending = missingMetricTargets(loadItems);
      if (pending.length) {
        const label = pending.length === 1 ? "target" : "targets";
        content.innerHTML = `<div class="empty panel">Loading manifold metrics for ${pending.length} ${label}...</div>`;
        Promise.all(pending.map(loadTargetMetrics)).then(() => {
          if (token === renderToken) render();
        });
        return;
      }
      const errors = metricErrors(loadItems);
      if (errors.length) {
        content.innerHTML = `<div class="empty panel warning">Could not load manifold metrics. ${escapeHtml(errors[0])}</div>`;
        return;
      }
      metricOrder = discoverMetricKeys();
      content.innerHTML = renderShell(candidate || items[0], items);
      renderLayerMetricsCharts(items);
      updateCentroidToggle();
      attachDistanceToggle(token);
      renderSideBySide(items, token);
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
      if (!event.target.closest) return;
      const modeButton = event.target.closest("[data-centroid-mode]");
      if (modeButton) {
        const mode = modeButton.getAttribute("data-centroid-mode");
        if (mode === "2d" || mode === "3d") {
          centroidMode = mode;
          updateCentroidToggle();
          drawCentroids();
        }
        return;
      }
      if (event.target.closest("#add-comparison")) {
        const candidate = activeCandidate();
        if (candidate && selection.length < MAX_ITEMS && !selection.some((it) => recordKey(it) === recordKey(candidate))) {
          selection.push(candidate);
          render();
        }
        return;
      }
      if (event.target.closest("#clear-comparison")) {
        selection.length = 0;
        render();
        return;
      }
      const removeButton = event.target.closest("[data-remove]");
      if (removeButton) {
        const key = removeButton.getAttribute("data-remove");
        const idx = selection.findIndex((it) => recordKey(it) === key);
        if (idx >= 0) { selection.splice(idx, 1); render(); }
      }
    });

    // Charts hidden inside a collapsed <details> can be resized to zero by the
    // window resize handler; re-fit them when the section is expanded.
    document.addEventListener("toggle", (event) => {
      const target = event.target;
      if (!target || !target.matches || !target.matches("details.collapsible")) return;
      if (target.id === "distance-details") return;
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

    function updateRunPickerVisibility() {
      const runLabel = selects.run.closest("label");
      if (!runLabel) return;
      runLabel.style.display = unique(records.map((r) => r.run)).length <= 1 ? "none" : "";
    }

    function updateCheckpointPickerVisibility() {
      const checkpointLabel = selects.checkpoint.closest("label");
      if (!checkpointLabel) return;
      const pool = records.filter((r) =>
        (!selects.run.value || r.run === selects.run.value) &&
        (!selects.model.value || r.model === selects.model.value)
      );
      const visible = checkpointValues(pool).length > 0;
      checkpointLabel.style.display = visible ? "" : "none";
      if (!visible) selects.checkpoint.value = "";
    }

    function setControlsDisabled(disabled) {
      for (const select of Object.values(selects)) select.disabled = disabled;
    }

    function bootstrap() {
      const content = document.getElementById("content");
      if (content) content.innerHTML = '<div class="empty panel">Loading manifold index...</div>';
      setControlsDisabled(true);
      fetchJson(MANIFEST_PATH)
        .then((manifest) => {
          if (manifest && manifest.schema_version && manifest.schema_version !== "manifold_viewer_manifest_v1") {
            throw new Error(`Unsupported manifest schema: ${manifest.schema_version}`);
          }
          records = buildRecordsFromManifest(manifest);
          setControlsDisabled(false);
          updateRunPickerVisibility();
          refreshOptions();
          render();
        })
        .catch((error) => {
          setControlsDisabled(true);
          if (content) {
            content.innerHTML = `<div class="empty panel warning">Could not load manifold manifest. ${escapeHtml(error)}</div>`;
          }
        });
    }

    bootstrap();
  </script>
</body>
</html>
"""
