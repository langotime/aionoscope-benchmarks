# Planning Workflow

Execution plans live as GitHub issues, not as local Markdown files.

Use `gh` from the repository root so it picks up the authenticated host and repo automatically.

## Conventions

- Create one issue per meaningful workstream that needs a plan.
- Mark plan issues with the GitHub label `plan`.
- Keep the plan body as plain Markdown in the issue body.
- Update the same issue as the work changes instead of checking plan Markdown into the repo.
- Historical checked-in plan Markdown must be migrated into closed GitHub issues and then removed from source control.
- Make commits meaningful. Each commit should represent one coherent slice of work.
- Reference the active plan issue in every related commit message with a non-closing footer such as `Part of #123`.
- Split unrelated work into separate commits so each commit maps cleanly to one plan issue.
- Close the issue when the plan is finished or superseded. Reopen it if the work becomes active again.

## `gh` Workflow

Create the `plan` label once for the repository:

```bash
gh label create plan --description "Execution plans"
```

Create a new plan issue interactively:

```bash
gh issue create \
  --title "<short topic>" \
  --label plan
```

Without `--body` or `--body-file`, `gh` prompts for the issue body interactively.

Create a new plan issue non-interactively:

```bash
gh issue create \
  --title "<short topic>" \
  --label plan \
  --body-file /path/to/plan.md
```

For agent runs, scripts, or any non-interactive shell, prefer the non-interactive form above. Use `--body-file -` only when you intentionally pipe Markdown on stdin.

Create a commit linked to a plan issue:

```bash
git commit -m "Add operator timeline scaffold" -m "Part of #123"
```

Use a non-closing reference such as `Part of #123` while work is still in progress so GitHub cross-links the commit and plan without closing the issue early.

Archive a historical checked-in plan into a closed issue:

```bash
gh label create plan-archive --description "Historical migrated plans"
gh issue create \
  --title "Archive: <historical plan title>" \
  --label plan \
  --label plan-archive \
  --body-file /path/to/historical-plan.md
gh issue close <number>
```

List plan issues:

```bash
gh issue list --state all --label plan
```

Read a plan issue:

```bash
gh issue view <number>
```

Read a plan issue with comments:

```bash
gh issue view <number> --comments
```

Replace the plan body from a local Markdown file:

```bash
gh issue edit <number> --body-file /path/to/plan.md
```

Replace the plan body from stdin:

```bash
gh issue edit <number> --body-file -
```

Add the `plan` label to an existing issue:

```bash
gh issue edit <number> --add-label plan
```

Close or reopen the plan issue:

```bash
gh issue close <number>
gh issue reopen <number>
```
