---
name: gridforge-pr-reviewer
description: >
  Reviews a PR or local diff in this repo (Colombian unit-commitment/dispatch model —
  Python/Pyomo domain package, FastAPI + polling-worker backend, Next.js frontend)
  against this repo's specific conventions — pydantic v2 patterns, the Storage
  abstraction, Pyomo/solver defaults, dependency/lint rules, and known verified
  gotchas — not general bug-hunting (use /code-review or the code-reviewer agent for
  that).
  Trigger: /gridforge-pr-reviewer, "revisa este PR", "review this PR against our conventions",
  "check this against our rules".
tools:
  - Bash
  - Read
  - Grep
  - AskUserQuestion
---

# GridForge PR Reviewer

Convention-focused reviewer for this repo. It does **not** replace `/code-review` or
the `code-reviewer` agent (correctness bugs, simplification, general quality) — those
already exist and cover that ground well. This skill's job is narrower: catch
violations of the rules in `.agents/rules/` and the verified gotchas in
[`overview.mdc`](../../rules/overview.mdc) that a generic reviewer wouldn't know to
check for.

If the user wants a general correctness/quality review, tell them to use
`/code-review` instead (or run both — this skill first, then `/code-review`).

---

## Phase 0 — Get the Diff

Ask (if not already clear) what to review:

- A GitHub PR by number → retrieve the diff and metadata (no `--repo` — let `gh` infer
  it from the local checkout; this repo is mid-rename `despacho-udea` → `gridforge`):
  ```bash
  gh pr diff <N>
  gh pr view <N> --json title,body,baseRefName,headRefName,files,reviews
  ```
  Also check for existing review comments:
  ```bash
  gh api repos/{owner}/{repo}/pulls/<N>/comments --jq '.[] | {id: .id, body: .body, path: .path, line: .line, user: .user.login}'
  ```
  (Resolve `{owner}/{repo}` from `gh repo view --json nameWithOwner -q .nameWithOwner`
  rather than hardcoding it.)
- The current local branch's diff against its base → `git diff develop...HEAD`
- Uncommitted local changes → `git diff` / `git diff --staged`

This is a single Python package (`app/`, `services/api/`, `services/worker/`) plus one
Next.js app (`frontend/`) — not a multi-platform monorepo. Identify whether the diff
touches Python, `frontend/`, `docker/`, or `.github/workflows/` and only apply the
relevant checklist section below.

---

## Phase 1 — Checklist

### Python (`app/`, `services/`, `alembic/`, `tests/`)

Cross-check against [`python-patterns.mdc`](../../rules/python-patterns.mdc):

- [ ] Pydantic **v2** only — plain `BaseModel` for schemas without custom validation;
      any `@field_validator` uses `@classmethod` + `info: ValidationInfo` +
      `info.data.get(...)`, never v1's `values` dict or a `Config` class /
      `@validator`. Field order matters — a field is only in `info.data` if declared
      earlier in the model.
- [ ] No new dependency added for something stdlib/uv/an already-installed package
      already resolves in a few lines.
- [ ] App I/O goes through `app.storage.get_storage(root)`, not a raw `open()` — the
      only documented exception is the two `open(resolve_input(...))` calls in
      `case_builder.py` for `dCondIniP`/`dCondIniU` (deliberate Fase 1 decision, not a
      pattern to extend).
- [ ] File paths resolved via `app.data.paths.resolve_input(kind, date, data_dir)` —
      never a hardcoded path; it checks the organized layout first, falls back to the
      flat `data_dir/{fecha}/` layout.
- [ ] Solver default stays `"cbc"` (`DispatchCase.solver`, CLI `--solver`) — a change
      away from it needs to re-read
      `docs/superpowers/specs/2026-08-05-fase2-docker-design.md` §3 first
      (`appsi_highs` via the legacy Pyomo wrapper breaks on every solve, not just
      pricing — verified, not theoretical).
- [ ] MPO/marginal prices come from the fix-and-resolve LP (`_solve_pricing_lp`), not
      duals read directly off a MILP solve.
- [ ] New fixture CSVs live under `tests/fixtures/`, anchored with
      `Path(__file__).parent` (never a cwd-relative string), and if outside an
      already-excepted path, check `.gitignore`'s global `*.csv` doesn't silently
      swallow them (`tests/fixtures/**/*.csv` is the one standing exception).
- [ ] Tests validating XM parsing use/extend `tests/fixtures/xm_smoke/` rather than
      mocking `build_case` with toy data.
- [ ] `ruff` (`select = ["E", "F", "I"]`, line-length 100) is the blocking lint/format
      gate; `ty` is informational only (`stages: [manual]`) — don't ask for a PR to
      fix a `ty`-only finding, and don't flag someone for not running it.
- [ ] Any `per-file-ignores` in `pyproject.toml` are pre-existing and documented
      inline — new ones need their own inline justification, not silent addition.

### Frontend (`frontend/`, Next.js)

- [ ] `npm run lint` (eslint) / `npm run test` (vitest) pass for touched files.
- [ ] No hardcoded API base URL where an env var / existing client config already
      covers it.

### Docker (`docker/`)

- [ ] `docker/Dockerfile.cli` / `Dockerfile.api` / `Dockerfile.worker` and
      `docker-compose.yml` / `docker-compose.dev.yaml` stay consistent with each other
      if one changes (e.g. base image / Node version bumps — see PR #14/#15 history:
      a Node version mismatch between dev compose and Dockerfile broke the frontend
      container before).

### Cross-cutting (any diff)

- [ ] No secrets, tokens, or credentials introduced anywhere in the diff (`.env`
      files, hardcoded strings).
- [ ] `data/` is git-ignored — a diff should never assume it exists or contains real
      data; don't let a PR claim "works with real data" off fixture-only test passes.
- [ ] **No AI/model co-authorship line** anywhere — not in commit messages, not in the
      PR title, not in the PR body (`Co-Authored-By: ...`, `🤖 Generated with ...`, or
      equivalent). This is a hard repo-wide rule (`CLAUDE.md`) — flag it immediately
      if present, don't treat it as a style nit.
- [ ] Branch name and PR base match
      [gridforge-issues-resolver](../gridforge-issues-resolver/SKILL.md) conventions:
      plain descriptive slug (`fix/<slug>`, `feat/<slug>`, `docs/<slug>`,
      `chore/<slug>` — no `issue-<N>-` prefix, that's not this repo's convention), PR
      base `develop` (promotion to `main` is a separate, manual, maintainer-driven
      step).
- [ ] PR title/body in **English**, conventional-commit style, `## Summary` +
      `## Test plan` shape — not the Spanish narrative style this repo's *issues* use
      (issues and PRs intentionally use different languages here; don't flag a
      Spanish-titled issue as wrong, and don't flag an English-titled PR as wrong).

---

## Phase 2 — Report

Report findings as a flat list, most severe first. For each: file:line, what's wrong, which rule it violates, suggested fix. Don't pad with items that don't apply to this diff's area(s).
Reports **SHOULD BE** written in Spanish (matching this skill's own review-comment convention, independent of the PR's own English title/body).

```
Revisión de convenciones — <PR #N | branch <name> | diff local>
─────────────────────────────────────────────
[CONVENCIÓN] <archivo>:<línea> — <qué está mal>
  Regla: <python-patterns.mdc | overview.mdc gotcha | ...>
  Fix:   <sugerencia concreta>
─────────────────────────────────────────────
<repetir>

Sin hallazgos en: <secciones del checklist que salieron limpias>
```

If nothing violates a repo-specific rule, say so plainly — don't invent findings to fill the report.

---

## Phase 3 — Complete Workflow: Review, Fix, and Lifecycle Management

Only if reviewing a real PR (not a local diff), follow this end-to-end workflow.

### Step 1: Address Existing Review Comments & Suggestions (if any exist)

For each comment/review suggesting a change, determine whether the suggestion will be applied:

- **If NOT applied**: reply with a clear justification.
  ```bash
  gh api -X POST repos/{owner}/{repo}/pulls/comments/<comment_id>/replies -f body="<Justificación clara y explícita de por qué no se abordará la sugerencia>"
  ```
- **If applied**: apply the change, commit, push, and reply pointing at the commit:
  ```bash
  git add <files>
  git commit -m "<descripción del cambio>"
  git push origin <branch-name>
  gh api -X POST repos/{owner}/{repo}/pulls/comments/<comment_id>/replies -f body="Implementado en el commit <commit_hash>. <Descripción general de la implementación>."
  ```

### Step 2: Submit the Review Verdict

- **Clean**: try to approve, with a detailed summary of gates actually run.
  ```bash
  gh pr review <N> --approve --body "<summary of tests run, checks executed, results>"
  ```
  Fallback if `--approve` fails (self-approval restriction): post as a comment instead.
  ```bash
  gh pr review <N> --comment --body "<same summary>"
  ```
- **Changes needed**:
  ```bash
  gh pr review <N> --request-changes --body "<detailed findings in markdown>"
  ```
  Apply fixes locally, commit, push, then re-review and approve once clean.

### Step 3: Post-Merge Issue and Branch Cleanup

Once the PR has merged into `develop` (or, rarely, `main` for a promotion PR):

- **PR references an issue**: check whether it auto-closed. Since PRs here target
  `develop` (not the repo's default branch), GitHub's closing keywords do **not**
  auto-close on this merge — they'll fire later when `develop` promotes to `main`.
  Don't close the issue manually just because the PR merged; ask the user whether they
  want it closed now (referencing the fix-is-merged-to-develop state explicitly) or
  left open until promotion.
- **PR not related to an issue**: safe to delete the branch locally and remotely.
  ```bash
  git branch -d <branch-name>
  git push origin --delete <branch-name>
  ```
- **PR not merged**: never delete the branch (local or remote); leave a comment
  explaining it's being kept active.

---

## Skill Constraints

- This skill checks repo-specific conventions only — for general correctness/bug review, defer to `/code-review` or the `code-reviewer` agent instead of duplicating that work.
- Only apply checklist sections relevant to the area(s) actually touched by the diff.
- Keep findings concrete — reference the exact rule file and line, don't restate the checklist as generic advice.
- Review bodies (`--request-changes`, `--comment`, `--approve`) must be fully formatted markdown — never post unformatted text.
- Follow the fallback logic for approvals: try `--approve` first, fall back to `--comment` on failure.
- Never delete a branch unless it's confirmed merged.
- Flag any AI/model co-authorship line as a hard blocker, not a suggestion.

## Related Skills

- [gridforge-issues-resolver](../gridforge-issues-resolver/SKILL.md) — implementing the fix this PR addresses
- [gridforge-bug-reporter](../gridforge-bug-reporter/SKILL.md) — this repo's real issue conventions, for cross-checking PR↔issue consistency
