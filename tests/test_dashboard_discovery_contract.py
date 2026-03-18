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
    assert 'task_finetune: "Task fine-tune"' not in html
    assert '<option value="parameters" selected>Model parameters</option>' in html
    assert 'id="bubble-parameter-scope-selector"' in html
    assert '<option value="best_layer">Through plotted best layer</option>' in html
    assert "const FALLBACK_RESULT_FILES" not in html
    assert "fallback manifest" not in html
    assert html.index("fetch(MODELS_LIST_PATH") < html.index("fetch(MODELS_DIR")
    assert "Could not discover model JSON files via ${MODELS_LIST_PATH} or directory listing" in html


def test_dashboard_bubble_warning_reports_model_specific_missing_fields() -> None:
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert "function summarizeBubbleMissingDetails(skippedDetails)" in html
    assert "function normalizeLoadedResult(payload)" in html
    assert "function bubbleSelectionContext()" in html
    assert "function colorLegendEntries(mode = getColorMode())" in html
    assert "function modelNamesForColorGroup(groupValue, mode = getColorMode())" in html
    assert "function setHoveredModels(modelNames)" in html
    assert "function syncChartHoverState()" in html
    assert 'chart.dispatchAction({ type: "highlight", seriesIndex: seriesIndices });' in html
    assert "dataIndex: dataIndices" in html
    assert 'label.addEventListener("pointerenter", () => {' in html
    assert 'item.addEventListener("pointerenter", () => {' in html
    assert "function renderColorLegend()" in html
    assert "function colorGroupLabel(result, mode = getColorMode())" in html
    assert "color = ${colorModeLabel(getColorMode())}" in html
    assert "parameter scope = through the furthest plotted best layer" in html
    assert 'Bubble chart skipped ${skippedCount} selected model${skippedCount === 1 ? "" : "s"}: ${detailsText}.' in html
    assert "Selected models are missing required fields: ${detailsText}." in html


def test_dashboard_sidebar_disclosures_keep_only_model_selector_open_by_default() -> None:
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert 'class="sidebar-disclosure-summary"' in html
    assert '<span class="sidebar-disclosure-title">Best layer selector</span>' in html
    assert '<span class="sidebar-disclosure-title">Color palette</span>' in html
    assert '<span class="sidebar-disclosure-title">Bubble chart</span>' in html
    assert '<span class="sidebar-disclosure-title">Filter models</span>' in html
    assert '<span class="sidebar-disclosure-title">Model selector</span>' in html
    assert len(re.findall(r'<details class="sidebar-disclosure" open>', html)) == 1
    assert re.search(
        r'<details class="sidebar-disclosure" open>\s*<summary class="sidebar-disclosure-summary">\s*<span class="sidebar-disclosure-title">Model selector</span>',
        html,
    )


def test_dashboard_model_filters_support_search_and_color_classes() -> None:
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert 'id="model-search"' in html
    assert "Class filters follow the active Color" in html
    assert '<span class="color-legend-title">Filter classes</span>' in html
    assert 'id="model-filter-mode"' in html
    assert 'id="model-filter-class-list"' in html
    assert 'id="model-filter-clear"' in html
    assert "modelFilterGroupValues: new Set()" in html
    assert "function clearModelFilterGroups()" in html
    assert "function modelMatchesPickerFilters(result, mode = getColorMode())" in html
    assert "function renderModelFilterGroups()" in html
    assert "No models match the current name / class filters." in html
    assert 'document.getElementById("color-mode-selector").addEventListener("change", handleColorModeChange);' in html
    assert 'document.getElementById("model-filter-clear").addEventListener("click", () => {' in html


def test_documentation_does_not_keep_cloudflare_pages_setup_notes() -> None:
    documentation = DOCUMENTATION_PATH.read_text(encoding="utf-8")

    assert "## Cloudflare Pages Deployment" not in documentation
    assert "SKIP_DEPENDENCY_INSTALL=true" not in documentation
