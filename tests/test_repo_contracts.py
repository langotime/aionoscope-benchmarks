from __future__ import annotations

from pathlib import Path

from aionoscope_benchmarks.constants import REPO_ROOT, RESULTS_ROOT
from aionoscope_benchmarks.repo_checks import (
    DESIGN_SYSTEM_STYLESHEET,
    SITE_SURFACES,
    check_no_checked_in_plan_markdown,
    check_site_shell,
    validate_repo_contracts,
)
from aionoscope_benchmarks.results import write_model_result


def test_repo_contracts_pass_for_current_repo() -> None:
    assert validate_repo_contracts() == []


def test_write_model_result_does_not_create_deploy_manifest(tmp_path: Path) -> None:
    models_dir = tmp_path / "results" / "models"
    out_path = models_dir / "Chronos-2__num_enabled_2.json"

    write_model_result(out_path=out_path, payload={"model": {"slug": "Chronos-2"}})

    assert out_path.exists()
    assert not (models_dir / "list.txt").exists()


def test_checked_in_plan_markdown_check_flags_temp_plan_dir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plans" / "001_example.md"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("# Example\n", encoding="utf-8")

    failures = check_no_checked_in_plan_markdown(tmp_path)

    assert len(failures) == 1
    assert "001_example.md" in failures[0].message


def test_site_shell_check_passes() -> None:
    assert check_site_shell() == []


def test_every_page_uses_the_hosted_design_system_and_no_local_copy() -> None:
    assert not (RESULTS_ROOT / "assets" / "langotime").exists()

    for page in (Path("results/index.html"), *SITE_SURFACES):
        markup = (REPO_ROOT / page).read_text(encoding="utf-8")
        assert DESIGN_SYSTEM_STYLESHEET in markup
        assert "assets/langotime/" not in markup


def test_index_owns_navigation_between_surfaces() -> None:
    index_markup = (RESULTS_ROOT / "index.html").read_text(encoding="utf-8")

    for surface in SITE_SURFACES:
        assert f'href="{surface.name}"' in index_markup
        surface_markup = (REPO_ROOT / surface).read_text(encoding="utf-8")
        assert 'href="index.html"' in surface_markup
        for other in SITE_SURFACES:
            if other != surface:
                assert f'href="{other.name}"' not in surface_markup


def test_legacy_article_pages_are_cloudflare_redirects() -> None:
    expected_redirects = {
        "/about-aionoscope.html": "https://blog.langotime.ai/about-aionoscope.html",
        "/about-lenepa.html": "https://blog.langotime.ai/about-lenepa.html",
        "/about-manifolds.html": "https://blog.langotime.ai/about-manifolds.html",
        "/about-manifolds-2.html": "https://blog.langotime.ai/about-manifolds-2.html",
        "/about-manifolds-3.html": "https://blog.langotime.ai/about-manifolds-3.html",
        "/about-manifolds-4.html": "https://blog.langotime.ai/about-manifolds-4.html",
    }

    redirects_path = RESULTS_ROOT / "_redirects"
    redirects = {}
    for line in redirects_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        source, target, status = line.split()
        redirects[source] = (target, status)

    assert redirects == {
        source: (target, "301") for source, target in expected_redirects.items()
    }
    for source in expected_redirects:
        assert not (RESULTS_ROOT / source.lstrip("/")).exists()
