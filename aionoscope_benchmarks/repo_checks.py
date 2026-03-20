from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .constants import MODEL_RESULTS_ROOT, REPO_ROOT
from .model_registry import MODEL_SPECS


MAX_AGENT_GUIDE_LINES = 80
REQUIRED_PATHS = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path("ARCHITECTURE.md"),
    Path("DOCUMENTATION.md"),
    Path("docs/index.md"),
    Path("docs/planning.md"),
    Path("docs/architecture-map.md"),
    Path("docs/benchmark-contract.md"),
    Path("docs/coding-standards.md"),
    Path("docs/adapter-guide.md"),
    Path("docs/results-schema.md"),
    Path("docs/dashboard-guide.md"),
    Path("docs/maintenance.md"),
    Path("docs/runbooks/foundational-sweep.md"),
    Path("docs/references/runtime-environments.md"),
    Path(".github/workflows/repo-checks.yml"),
    Path(".github/workflows/weekly-gardening.yml"),
    Path("results/AGENTS.md"),
    Path("results/dashboard.html"),
)
AGENTS_REQUIRED_TOKENS = (
    "docs/index.md",
    "docs/planning.md",
    "docs/coding-standards.md",
    "results/AGENTS.md",
    "Ignore `README.md` when gathering agent context.",
    "uv run python -m aionoscope_benchmarks.repo_checks",
    "uv run python -m aionoscope_benchmarks.dashboard_smoke",
)
PLANNING_REQUIRED_TOKENS = (
    "Execution plans live as GitHub issues, not as local Markdown files.",
    "gh issue create",
    "Part of #123",
    "plan-archive",
)
PLANNING_BANNED_TOKENS = (
    "glab",
    "GitLab",
)
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
JSON_FILENAME_RE = re.compile(r"^(?P<slug>.+)__num_enabled_(?P<num_enabled>\d+)\.json$")


@dataclass(frozen=True)
class CheckFailure:
    code: str
    message: str


def _repo_relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _markdown_paths() -> list[Path]:
    paths: list[Path] = []
    for path in REPO_ROOT.rglob("*.md"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(REPO_ROOT).parts
        if any(part.startswith(".venv") or part in {".git", "__pycache__"} for part in relative_parts):
            continue
        paths.append(path)
    return sorted(paths)


def _iter_markdown_links(text: str) -> list[str]:
    targets: list[str] = []
    for match in MARKDOWN_LINK_RE.finditer(text):
        target = match.group(1).strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        targets.append(target)
    return targets


def _resolve_local_markdown_target(source_path: Path, raw_target: str) -> Path | None:
    if raw_target.startswith("#"):
        return None
    parts = urlsplit(raw_target)
    if parts.scheme or parts.netloc:
        return None
    path_part = unquote(parts.path)
    if not path_part:
        return None
    return (source_path.parent / path_part).resolve()


def _import_targets(path: Path) -> list[str]:
    tree = ast.parse(_read_text(path), filename=str(path))
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = f"{'.' * node.level}{module}"
            targets.append(module)
    return targets


def _class_bases_by_name(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(_read_text(path), filename=str(path))
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases: list[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
        out[node.name] = bases
    return out


def _class_inherits_frozen_adapter(path: Path, class_name: str) -> bool:
    bases_by_name = _class_bases_by_name(path)

    def _inherits(name: str, seen: set[str]) -> bool:
        if name == "FrozenTimeSeriesAdapter":
            return True
        if name in seen:
            return False
        for base_name in bases_by_name.get(name, []):
            if base_name == "FrozenTimeSeriesAdapter":
                return True
            if _inherits(base_name, seen | {name}):
                return True
        return False

    return _inherits(class_name, set())


def check_required_paths() -> list[CheckFailure]:
    failures: list[CheckFailure] = []
    for relative_path in REQUIRED_PATHS:
        path = REPO_ROOT / relative_path
        if not path.exists():
            failures.append(
                CheckFailure(
                    code="required-path",
                    message=f"Missing required repository path: {_repo_relative(path)}",
                )
            )
    return failures


def check_no_dev_deploy_manifest() -> list[CheckFailure]:
    manifest_path = REPO_ROOT / "results" / "models" / "list.txt"
    if manifest_path.exists():
        return [
            CheckFailure(
                code="deploy-artifact",
                message=(
                    "results/models/list.txt should not exist in the dev tree; "
                    "it is generated during dashboard deployment."
                ),
            )
        ]
    return []


def check_markdown_links() -> list[CheckFailure]:
    failures: list[CheckFailure] = []
    for path in _markdown_paths():
        for raw_target in _iter_markdown_links(_read_text(path)):
            target = _resolve_local_markdown_target(path, raw_target)
            if target is None:
                continue
            if not target.exists():
                failures.append(
                    CheckFailure(
                        code="markdown-link",
                        message=(
                            f"Broken local Markdown link in {_repo_relative(path)}: "
                            f"{raw_target!r}"
                        ),
                    )
                )
    return failures


def check_agent_guide() -> list[CheckFailure]:
    failures: list[CheckFailure] = []
    path = REPO_ROOT / "AGENTS.md"
    text = _read_text(path)
    non_empty_line_count = sum(1 for line in text.splitlines() if line.strip())
    if non_empty_line_count > MAX_AGENT_GUIDE_LINES:
        failures.append(
            CheckFailure(
                code="agents-size",
                message=(
                    f"AGENTS.md should stay a concise table of contents; "
                    f"found {non_empty_line_count} non-empty lines"
                ),
            )
        )
    banned_tokens = (
        "Write plans to files in Markdown.",
        "Put plans into the `plans/` subdirectory",
        "do NOT delete them, the human will delete them",
    )
    for token in banned_tokens:
        if token in text:
            failures.append(
                CheckFailure(
                    code="agents-plans",
                    message=f"AGENTS.md still contains stale checked-in plan guidance: {token!r}",
                )
            )
    for token in AGENTS_REQUIRED_TOKENS:
        if token not in text:
            failures.append(
                CheckFailure(
                    code="agents-token",
                    message=f"AGENTS.md is missing required navigation or check token: {token!r}",
                )
            )
    return failures


def check_planning_workflow() -> list[CheckFailure]:
    failures: list[CheckFailure] = []
    text = _read_text(REPO_ROOT / "docs/planning.md")
    for token in PLANNING_REQUIRED_TOKENS:
        if token not in text:
            failures.append(
                CheckFailure(
                    code="planning-token",
                    message=f"docs/planning.md is missing required planning token: {token!r}",
                )
            )
    for token in PLANNING_BANNED_TOKENS:
        if token in text:
            failures.append(
                CheckFailure(
                    code="planning-token",
                    message=f"docs/planning.md contains stale planning token: {token!r}",
                )
            )
    return failures


def check_agent_context_docs() -> list[CheckFailure]:
    failures: list[CheckFailure] = []
    docs_index = _read_text(REPO_ROOT / "docs/index.md")
    documentation = _read_text(REPO_ROOT / "DOCUMENTATION.md")
    readme = _read_text(REPO_ROOT / "README.md")

    required_docs_index = (
        "Agent rule: ignore [../README.md](../README.md) when gathering repository context.",
        "human-facing onboarding only",
    )
    for token in required_docs_index:
        if token not in docs_index:
            failures.append(
                CheckFailure(
                    code="agent-context",
                    message=f"docs/index.md is missing required README agent-context guidance: {token!r}",
                )
            )

    required_documentation = "README.md` is human-facing onboarding; agents must keep it up to date, but should not use it as the repository context source."
    if required_documentation not in documentation:
        failures.append(
            CheckFailure(
                code="agent-context",
                message="DOCUMENTATION.md is missing the README human-only guidance for agents.",
            )
        )

    required_readme = "Agents should update it when public-facing workflow or scope changes, but they should gather repository context from"
    if required_readme not in readme:
        failures.append(
            CheckFailure(
                code="agent-context",
                message="README.md is missing the human-facing onboarding note for agents.",
            )
        )
    return failures


def check_retained_coding_defaults() -> list[CheckFailure]:
    failures: list[CheckFailure] = []
    coding_standards = _read_text(REPO_ROOT / "docs/coding-standards.md")
    required_tokens = (
        "PEP 585 plus PEP 604",
        "Keep code DRY, minimal, and KISS.",
        "avoid one-letter identifiers",
        "Always run relevant tests or smoke checks before reporting completion.",
        "Avoid `sys.path.insert` in normal package code.",
        "This repo does not maintain a top-level `examples/` tree.",
    )
    for token in required_tokens:
        if token not in coding_standards:
            failures.append(
                CheckFailure(
                    code="coding-standards",
                    message=f"docs/coding-standards.md is missing retained original guidance: {token!r}",
                )
            )
    return failures


def check_no_checked_in_plan_markdown(repo_root: Path = REPO_ROOT) -> list[CheckFailure]:
    failures: list[CheckFailure] = []
    plan_dir = repo_root / "plans"
    if not plan_dir.exists():
        return failures
    for path in sorted(plan_dir.rglob("*.md")):
        failures.append(
            CheckFailure(
                code="checked-in-plan",
                message=f"Checked-in plan Markdown must be migrated out of source control: {path}",
            )
        )
    return failures


def check_model_registry_boundaries() -> list[CheckFailure]:
    failures: list[CheckFailure] = []
    for spec in MODEL_SPECS.values():
        if not spec.module.startswith("aionoscope_benchmarks.adapters."):
            failures.append(
                CheckFailure(
                    code="registry-module",
                    message=(
                        f"Model {spec.name!r} must resolve through the adapters package, "
                        f"got module {spec.module!r}"
                    ),
                )
            )
            continue
        module_path = REPO_ROOT / f"{spec.module.replace('.', '/')}.py"
        if not module_path.exists():
            failures.append(
                CheckFailure(
                    code="registry-module",
                    message=f"Adapter module for {spec.name!r} does not exist: {module_path}",
                )
            )
            continue
        if not _class_inherits_frozen_adapter(module_path, spec.class_name):
            failures.append(
                CheckFailure(
                    code="registry-class",
                    message=(
                        f"Adapter class {spec.class_name!r} for model {spec.name!r} does not "
                        "inherit FrozenTimeSeriesAdapter through its local class hierarchy."
                    ),
                )
            )
    return failures


def check_module_layering() -> list[CheckFailure]:
    failures: list[CheckFailure] = []
    results_imports = _import_targets(REPO_ROOT / "aionoscope_benchmarks/results.py")
    if any("runtime_dataset" in target or "adapters" in target for target in results_imports):
        failures.append(
            CheckFailure(
                code="module-layering",
                message="aionoscope_benchmarks/results.py must stay independent of adapters and runtime_dataset.",
            )
        )
    runtime_imports = _import_targets(REPO_ROOT / "aionoscope_benchmarks/runtime_dataset.py")
    if any("results" in target for target in runtime_imports):
        failures.append(
            CheckFailure(
                code="module-layering",
                message="aionoscope_benchmarks/runtime_dataset.py must not depend on results assembly.",
            )
        )
    dashboard_html = _read_text(REPO_ROOT / "results/dashboard.html")
    for token in ("runtime_dataset.py", "offline_probe.py", "aionoscope_benchmarks.run_model"):
        if token in dashboard_html:
            failures.append(
                CheckFailure(
                    code="dashboard-boundary",
                    message=(
                        f"results/dashboard.html should stay a pure reader of JSON artifacts; "
                        f"found stale compute-layer token {token!r}"
                    ),
                )
            )
    return failures


def _load_result_payloads() -> list[tuple[Path, dict[str, object]]]:
    payloads: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(MODEL_RESULTS_ROOT.glob("*.json")):
        payloads.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return payloads


def check_result_corpus() -> list[CheckFailure]:
    failures: list[CheckFailure] = []
    payloads = _load_result_payloads()
    if not payloads:
        failures.append(
            CheckFailure(
                code="result-corpus",
                message="results/models/ does not contain any JSON artifacts.",
            )
        )
        return failures
    benchmark_pairs: set[tuple[str, str]] = set()
    for path, payload in payloads:
        model = payload.get("model")
        dataset = payload.get("dataset")
        if not isinstance(model, dict) or not isinstance(dataset, dict):
            failures.append(
                CheckFailure(
                    code="result-payload",
                    message=f"{_repo_relative(path)} is missing top-level model/dataset sections.",
                )
            )
            continue
        required_model_fields = (
            "slug",
            "family",
            "checkpoint_name",
        )
        for field in required_model_fields:
            if field not in model:
                failures.append(
                    CheckFailure(
                        code="result-payload",
                        message=f"{_repo_relative(path)} is missing model.{field}.",
                    )
                )
        architecture = model.get("architecture")
        training = model.get("training")
        if not isinstance(architecture, dict) or "backbone" not in architecture:
            failures.append(
                CheckFailure(
                    code="result-payload",
                    message=f"{_repo_relative(path)} is missing model.architecture.backbone.",
                )
            )
        if not isinstance(training, dict) or "paradigm" not in training:
            failures.append(
                CheckFailure(
                    code="result-payload",
                    message=f"{_repo_relative(path)} is missing model.training.paradigm.",
                )
            )
        for field in ("benchmark_family", "benchmark_version", "num_enabled"):
            if field not in dataset:
                failures.append(
                    CheckFailure(
                        code="result-payload",
                        message=f"{_repo_relative(path)} is missing dataset.{field}.",
                    )
                )
        family = str(dataset.get("benchmark_family"))
        version = str(dataset.get("benchmark_version"))
        benchmark_pairs.add((family, version))
        match = JSON_FILENAME_RE.match(path.name)
        if match is None:
            failures.append(
                CheckFailure(
                    code="result-filename",
                    message=f"{_repo_relative(path)} does not follow the canonical filename pattern.",
                )
            )
            continue
        expected_slug = str(model.get("slug"))
        expected_num_enabled = str(int(dataset.get("num_enabled")))
        if match.group("slug") != expected_slug or match.group("num_enabled") != expected_num_enabled:
            failures.append(
                CheckFailure(
                    code="result-filename",
                    message=(
                        f"{_repo_relative(path)} does not match model.slug/dataset.num_enabled "
                        "inside the JSON payload."
                    ),
                )
            )
    if len(benchmark_pairs) > 1:
        failures.append(
            CheckFailure(
                code="result-corpus",
                message=(
                    "results/models/ mixes multiple benchmark families or versions: "
                    f"{sorted(benchmark_pairs)}"
                ),
            )
        )
    return failures


def validate_repo_contracts() -> list[CheckFailure]:
    failures: list[CheckFailure] = []
    failures.extend(check_required_paths())
    failures.extend(check_no_dev_deploy_manifest())
    failures.extend(check_markdown_links())
    failures.extend(check_agent_guide())
    failures.extend(check_agent_context_docs())
    failures.extend(check_retained_coding_defaults())
    failures.extend(check_planning_workflow())
    failures.extend(check_no_checked_in_plan_markdown())
    failures.extend(check_model_registry_boundaries())
    failures.extend(check_module_layering())
    failures.extend(check_result_corpus())
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate repo-local docs, planning, and result-harness contracts.")
    parser.parse_args(argv)

    failures = validate_repo_contracts()
    if failures:
        for failure in failures:
            print(f"[{failure.code}] {failure.message}", file=sys.stderr)
        return 1

    print(
        "Repo contract checks passed: docs, planning workflow, result corpus, and structural boundaries are coherent.",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
