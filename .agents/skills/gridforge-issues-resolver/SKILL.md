---
name: gridforge-issues-resolver
description: >
  Interactive issue resolver for this repo (Colombian unit-commitment/dispatch model —
  Python/Pyomo domain package, FastAPI + polling-worker backend, Next.js frontend).
  Lists open issues, helps diagnose the problem, sets up a branch off develop,
  implements the fix, runs the real validation gates (ruff, pytest, ty, frontend
  lint/test), and opens a PR against develop in this repo's actual style (English
  conventional-commit title, `## Summary`/`## Test plan` body).
  Trigger: /gridforge-issues-resolver, "resolver issue", "resolver bug", "solve issue", "solucionar issue".
tools:
  - Bash
  - Read
  - Edit
  - Write
  - AskUserQuestion
---

# GridForge Issues Resolver

Interactive agent that resolves a GitHub issue end-to-end: selection → diagnosis → branch setup → implementation & validation gates → PR.

Repo: whatever `git remote get-url origin` resolves to. Don't hardcode an owner/repo
slug in `gh` commands — omit `--repo` and let `gh` infer it from the local checkout.
(Repo is mid-rename `despacho-udea` → `gridforge`; a hardcoded slug breaks the moment
that lands.)

---

## Phase 0 — Load & Select Issue

```bash
gh auth status
gh issue list --limit 15 --json number,title,labels,state
```

Present the list; ask which issue to resolve (by number or title). Then fetch full detail:

```bash
gh issue view <N> --json number,title,labels,body,state
gh issue view <N> --comments
```

### Detect Type

This repo has **no title-prefix convention** for type (`bug(scope): ...` is not used
here) — type comes from the issue's label instead:

| Label | Type |
|---|---|
| `bug` | **Bug** |
| `enhancement` | **Feature** |
| `documentation` | **Docs** |
| `question` | Ask the user how to scope it — often needs a design decision before a fix is clear |

Type drives the branch prefix (Phase 2) and PR title (Phase 4).

### Display

Issue bodies here are freeform Spanish prose (not a fixed section template) — often
referencing a spec/plan under `docs/superpowers/specs|plans/` or a project memory.
If the body references one, read it before proceeding — it usually has the real
context (verified constraints, prior findings) the issue text only summarizes.

---

## Phase 1 — Locate the Code & Diagnose

1. Use [`overview.mdc`](../../rules/overview.mdc)'s "Layout rapido" section to map the
   issue's area to the right module (`app/model`, `app/data`, `app/pipeline` incl.
   `case_builder.py`, `app/schemas`, `app/storage`, `services/api`, `services/worker`,
   `frontend`) — this is a single Python package + one Next.js app, not a multi-service
   monorepo, so there's no separate repo-map skill/file to resolve a scope name against.
2. Grep/search for the failing function, endpoint, or config the issue names.
3. Cross-check the "Gotchas verificados" section of
   [`overview.mdc`](../../rules/overview.mdc) and
   [`python-patterns.mdc`](../../rules/python-patterns.mdc) — several real bugs in this
   repo trace back to a known pitfall (unit/scale conventions on `dispo_declarada.csv`/
   `ofertas.csv`/`PrId*.txt`, `thefuzz` `score_cutoff=70` guards, the flat-vs-organized
   data layout `resolve_input` falls back through, pydantic v2-only patterns, Storage
   abstraction instead of raw `open()`).
4. For anything non-trivial (touches `case_builder.py`, solver config, or a documented
   gotcha), call the `advisor` tool for a second opinion on the fix strategy before
   writing code.
5. Write a short fix outline: files to change, root cause, tests to add/update,
   validation gates that apply (Phase 3).

Present the plan and wait for confirmation:

> "Diagnostiqué el issue y este es el plan de fix. ¿Procedo a implementar?"

---

## Phase 2 — Branch Setup

Per [`overview.mdc`](../../rules/overview.mdc): every non-trivial fix/feature branches
off `develop` and PRs back into `develop` — never commit directly to `develop`. This
repo has no separate hotfix→`main` flow; promoting `develop` to `main` is a distinct,
manual, maintainer-driven process (`chore/promote-develop-to-main`-style PRs) outside
this skill's scope.

```bash
git checkout develop
git pull origin develop
git checkout -b <prefix>/<slug>
```

**Prefix by type** — matches this repo's real branch history (`fix/xm-download-url-migration`,
`chore/repo-reorganization`, `fase5a-frontend-visual-design`):

| Type | Prefix |
|---|---|
| Bug | `fix/` |
| Feature | `feat/` |
| Docs | `docs/` |
| Chore/refactor (no user-facing behavior change) | `chore/` |

Note this repo's branch names are a **plain descriptive slug**, not
`issue-<N>-<slug>` — real branches never embed the issue number (verified against PR
history). Reference the issue number in the PR body instead (Phase 4).

Confirm the branch name and base with the user before continuing.

---

## Phase 3 — Implement & Validate

Implement the fix per the confirmed plan, following
[`python-patterns.mdc`](../../rules/python-patterns.mdc) (pydantic v2 only, no `open()`
for app I/O — use `app.storage`, solver default stays `cbc`, don't add a dependency for
what stdlib/an installed package already does).

Run the gates that apply to what was touched:

| Area | Gate | Command |
|---|---|---|
| Python | Lint | `uv run ruff check .` |
| Python | Format check | `uv run ruff format --check .` |
| Python | Tests | `uv run pytest -q` |
| Python | Type check (informational, never blocking) | `uv run ty check app/ \|\| true` |
| Frontend (`frontend/`) | Lint | `cd frontend && npm run lint` |
| Frontend | Tests | `cd frontend && npm run test` |
| Whole repo | Pre-commit hooks | `pre-commit run --all-files` |

`ty` is informational only (`stages: [manual]` in `.pre-commit-config.yaml`) — a `ty`
finding does not block the PR; a `ruff`, `pytest`, or frontend lint/test failure does.

Do not proceed to Phase 4 until all gates for the touched area(s) pass.

---

## Phase 4 — Push & Open PR

```bash
git push origin <branch-name>
```

PR title and body are in **English**, conventional-commit style, matching this repo's
real PRs — e.g. `fix: XM download API migrated hosts, dead URL broke every real-date run`,
`feat(frontend): reskin login + real forgot-password flow`. This is the opposite
convention from issues (Spanish) — don't mix them up.

Body uses this repo's real template — `## Summary` + `## Test plan`, not the
`## Causa raíz`/`## Cambios`/`## Validación` shape some other repos use:

```bash
gh pr create \
  --base develop \
  --title "<type>[(scope)]: <short description>" \
  --body "$(cat <<'BODY'
## Summary
<1-3 bullet points: what changed and why, technical detail welcome>

## Test plan
- [x] <gate actually run, e.g. `uv run pytest -q` — N/N passing>
- [x] <manual verification actually performed, if any>

Closes #<N>
BODY
)"
```

GitHub's closing keywords (`Closes #N`) only auto-close on merge into the repo's
**default branch**. This PR targets `develop`, not the default branch — so `#<N>`
won't auto-close until `develop` is later promoted to `main`. Reference it inline, but
don't tell the user it auto-closed.

Never add an AI/model co-authorship line to the commit, PR title, or PR body —
repo-wide rule (`CLAUDE.md`), applies here without exception.

---

## Phase 5 — Confirm & Close

```
Issue resuelto, PR abierto.

Issue:  #<N> (<título>)
PR:     <PR_URL>
Branch: <branch-name> → develop
Gates:  <cuáles corrieron y pasaron>
```

---

## Skill Constraints

- Always use a branch + PR — never commit directly to `develop` or `main`.
- Always run `uv run` from the repo root (single package, single `.venv` — not a
  per-service venv like a monorepo would need).
- Never assume a gate exists that this repo doesn't use, and never treat `ty` as
  blocking.
- Run all validation gates for the touched area(s) before opening the PR.
- Write PR title/body in English; issue title/body stay Spanish if you also touch
  [gridforge-bug-reporter](../gridforge-bug-reporter/SKILL.md)'s output — never mix
  the two languages within one artifact.
- Never add an AI/model co-authorship line anywhere this skill writes text.

## Related Skills

- [gridforge-bug-reporter](../gridforge-bug-reporter/SKILL.md) — if a new issue needs filing instead of resolving an existing one
- [gridforge-pr-reviewer](../gridforge-pr-reviewer/SKILL.md) — review the PR against this repo's conventions before/after opening it
