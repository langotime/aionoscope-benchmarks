from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = REPO_ROOT / "results" / "dashboard.html"
DOCUMENTATION_PATH = REPO_ROOT / "DOCUMENTATION.md"


def test_dashboard_tries_list_manifest_before_directory_listing_without_builtin_fallback() -> None:
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert 'const MODELS_LIST_PATH = "/models/list.txt";' in html
    assert "const FALLBACK_RESULT_FILES" not in html
    assert "fallback manifest" not in html
    assert html.index("fetch(MODELS_LIST_PATH") < html.index("fetch(MODELS_DIR")
    assert "Could not discover model JSON files via ${MODELS_LIST_PATH} or directory listing" in html


def test_documentation_describes_cloudflare_pages_manifest_generation() -> None:
    documentation = DOCUMENTATION_PATH.read_text(encoding="utf-8")

    assert "## Cloudflare Pages Deployment" in documentation
    assert "find results/models -maxdepth 1 -type f -name '*.json' -printf '%f\\n' | LC_ALL=C sort > results/models/list.txt" in documentation
    assert "Build output directory:" in documentation
    assert "SKIP_DEPENDENCY_INSTALL=true" in documentation
    assert "results" in documentation
