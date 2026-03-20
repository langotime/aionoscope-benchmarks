import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = REPO_ROOT / "results" / "dashboard.html"
DOCUMENTATION_PATH = REPO_ROOT / "DOCUMENTATION.md"


def test_dashboard_tries_list_manifest_before_directory_listing_without_builtin_fallback() -> None:
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert 'const MODELS_LIST_PATH = "/models/list.txt";' in html
    assert 'id="color-mode-selector"' in html
    assert 'id="color-legend"' in html
    assert 'id="color-legend-list"' in html
    assert "Palette legend" in html
    assert '<option value="family" selected>By model family</option>' in html
    assert '<option value="architecture_backbone">By architecture type</option>' in html
    assert '<option value="training">By training paradigm</option>' in html
    assert 'transformer_causal: "Transformer (causal)"' in html
    assert 'transformer_full_attention: "Transformer (full attention)"' in html
    assert 'transformer_moe_causal: "Transformer + MoE (causal)"' in html
    assert 'linear_rnn: "Linear RNN"' in html
    assert "const FALLBACK_RESULT_FILES" not in html
    assert "fallback manifest" not in html
    assert html.index("fetch(MODELS_LIST_PATH") < html.index("fetch(MODELS_DIR")
    assert "Could not discover model JSON files via ${MODELS_LIST_PATH} or directory listing" in html


def test_dashboard_uses_composite_run_identity_and_enabled_component_filter() -> None:
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert 'id="num-enabled-filter-list"' in html
    assert "const DEFAULT_NUM_ENABLED_FILTER_VALUE = 2;" in html
    assert "selectedRunKeys: new Set()" in html
    assert "numEnabledFilterValues: new Set()" in html
    assert "function resultNumEnabled(result)" in html
    assert "function runKey(result)" in html
    assert 'result && result.model ? String(result.model.slug || canonicalModelName(result)) : "unknown_model"' in html
    assert 'resultBenchmarkFamily(result),' in html
    assert 'resultBenchmarkVersion(result),' in html
    assert '`num_enabled_${resultNumEnabled(result) === null ? "unknown" : resultNumEnabled(result)}`' in html
    assert "function renderNumEnabledFilter()" in html
    assert "function resultMatchesNumEnabledFilter(result)" in html
    assert "state.numEnabledFilterValues = availableValues.has(DEFAULT_NUM_ENABLED_FILTER_VALUE)" in html
    assert "? new Set([DEFAULT_NUM_ENABLED_FILTER_VALUE])" in html
    assert "state.numEnabledFilterValues = new Set(Array.from(availableValues));" not in html
    assert "state.selectedRunKeys = new Set(visibleResults().map((result) => runKey(result)));" in html
    assert "return visibleResults().filter((result) => state.selectedRunKeys.has(runKey(result)));" in html


def test_dashboard_run_labels_hide_benchmark_version_but_keep_num_enabled_suffix() -> None:
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert "function runLabel(result)" in html
    assert 'return `${canonicalModelName(result)} · num_enabled=${numEnabled === null ? "?" : numEnabled}`;' in html
    assert 'return `${canonicalModelName(result)} · ${resultBenchmarkVersion(result)} · num_enabled=${numEnabled === null ? "?" : numEnabled}`;' not in html


def test_dashboard_num_enabled_filter_applies_before_selector_and_plots() -> None:
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert "function visibleResults()" in html
    assert "function syncSelectedRunsToVisibleResults()" in html
    assert "visibleResults().forEach((result) => {" in html
    assert "return visibleResults().find((result) => runKey(result) === String(runKeyValue)) || null;" in html
    assert "setHoveredModels(runKeysForGroup(entry.groupValue, mode));" in html
    assert "dataIndexByModel: new Map(bubbleRows.map((row, index) => [row.runKey, index]))" in html
    assert "seriesIndexByModel.set(runKey(result), series.length - 1);" in html
    assert "No runs match the current num_enabled filter. Re-enable at least one value." in html


def test_dashboard_json_loader_uses_bounded_concurrency_timeout_and_progress() -> None:
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert "const RESULT_FETCH_TIMEOUT_MS = 30000;" in html
    assert "const RESULT_FETCH_CONCURRENCY = 2;" in html
    assert "const RESULT_FETCH_MAX_ATTEMPTS = 3;" in html
    assert "const RESULT_FETCH_RETRY_DELAY_MS = 500;" in html
    assert "function setLoadingProgress({ isLoading, totalFiles = 0, completedFiles = 0, loadedFiles = 0 })" in html
    assert '`Loading... ${state.loadingLoadedFiles}/${state.loadingTotalFiles}`' in html
    assert 'const fragments = [`Loading JSON files... ${state.loadingLoadedFiles}/${state.loadingTotalFiles}`];' in html
    assert "function wait(delayMs)" in html
    assert "const controller = new AbortController();" in html
    assert "controller.abort();" in html
    assert 'normalizedError = new Error(`timed out after ${Math.round(timeoutMs / 1000)}s`);' in html
    assert "normalizedError.isRetriable = true;" in html
    assert 'throw new Error(`${errorMessage(normalizedError)} after ${attempt} attempts`);' in html
    assert "const workerCount = Math.min(RESULT_FETCH_CONCURRENCY, files.length);" in html
    assert "await Promise.all(Array.from({ length: workerCount }, () => worker()));" in html
    assert "const settled = new Array(files.length);" in html
    assert "const response = await fetch(path, { cache: \"no-store\", signal: controller.signal });" in html


def test_dashboard_sidebar_disclosures_keep_only_model_selector_open_by_default() -> None:
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert 'class="sidebar-disclosure-summary"' in html
    assert '<span class="sidebar-disclosure-title">Best layer selector</span>' in html
    assert '<span class="sidebar-disclosure-title">Color palette</span>' in html
    assert '<span class="sidebar-disclosure-title">Bubble chart</span>' in html
    assert '<span class="sidebar-disclosure-title">Enabled components</span>' in html
    assert '<span class="sidebar-disclosure-title">Model selector</span>' in html
    assert len(re.findall(r'<details class="sidebar-disclosure" open>', html)) == 1
    assert re.search(
        r'<details class="sidebar-disclosure" open>\s*<summary class="sidebar-disclosure-summary">\s*<span class="sidebar-disclosure-title">Model selector</span>',
        html,
    )


def test_documentation_does_not_keep_cloudflare_pages_setup_notes() -> None:
    documentation = DOCUMENTATION_PATH.read_text(encoding="utf-8")

    assert "## Cloudflare Pages Deployment" not in documentation
    assert "SKIP_DEPENDENCY_INSTALL=true" not in documentation
