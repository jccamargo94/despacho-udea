# Fase 2A: Toolchain (uv + Python 3.12 + pydantic v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `requirement.txt` with a `uv`-managed `pyproject.toml` on Python 3.12, migrate the one pydantic v1 validator to v2, fix a dead-but-live `appsi_highs` default in `model.py`, and add a `ruff`-blocking / `ty`-informational pre-commit — ending with 78/78 tests green on the new toolchain.

**Architecture:** No application-code restructuring. `app/` stays exactly where it is, imported the same way tests already do (`conftest.py` sys.path insert). `pyproject.toml` gets `[tool.uv] package = false` so uv manages the venv/deps without trying to build `app` as an installable wheel — the smallest diff that gets dependency management without a src-layout migration.

**Tech Stack:** `uv` (already installed, 0.11.15), Python 3.12, `pydantic` 2.x, `ruff`, `pre-commit`.

## Global Constraints

- **Runtime library versions carry over unchanged from `requirement.txt`** — this migrates the dependency *manager* and the *interpreter*, not the libraries. `numpy==1.26.4`, `pandas==2.2.2`, `Pyomo==6.7.3` exactly, all other runtime pins exactly as currently frozen (see Task 1). The one deliberate exception is `pydantic` (v1.10.18 -> v2.x, Task 3) — already decided, isolated to its own task so a regression is attributable.
- **`numpy` must stay `<2`.** Verified: `pyomo==6.7.3` imports use `np.float_`, removed in NumPy 2.0 — `import pyomo.environ` raises `AttributeError` outright on unpinned numpy. Confirmed by installing the stack in a clean Python 3.12 venv.
- **`.stack()` semantics matter for correctness, not just style.** `app/pipeline/case_builder.py` calls `.stack()` at lines 194, 206, 364, 440, and `app/data/ofei.py:82` does too. Pandas 2.2.2 (pinned) is the version every finding in the Fase 2 design spec was verified against — do not let a dependency resolver silently pick a newer pandas.
- **No behavior changes to `app/` beyond the two named in this plan** (pydantic validator migration, `model.py` solver-string default). Everything else in `app/` is untouched — this plan is toolchain only.
- Reference: `docs/superpowers/specs/2026-08-05-fase2-docker-design.md` section 1 ("Toolchain") and section 6 (verified findings #3, #5).

---

### Task 1: `pyproject.toml` + `uv.lock` on Python 3.12, same library versions as today

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Modify: none yet (keep `requirement.txt` in place until Task 5)

**Interfaces:**
- Produces: a `uv`-managed venv at `.venv/`, activatable/runnable via `uv run <cmd>`. No new Python-level APIs — this task only changes how dependencies are installed.

- [ ] **Step 1: Pin the interpreter**

```bash
uv python install 3.12
echo "3.12" > .python-version
```

- [ ] **Step 2: Write `pyproject.toml`**

Runtime deps are the exact versions from `requirement.txt` for everything `app/` actually imports (verified by grepping `app/**/*.py` for `pyomo`, `pandas`, `numpy`, `thefuzz`, `typer`, `requests`, `plotly`, `openpyxl`, `xlrd`, `pydantic`, `pyyaml`, `highspy`, `pydataxm`). `pydantic` stays on v1 in this task — the v2 migration is Task 3, isolated so a regression here is attributable to the toolchain switch, not the pydantic bump.

```toml
[project]
name = "despacho-udea"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pandas==2.2.2",
    "numpy==1.26.4",
    "Pyomo==6.7.3",
    "highspy==1.7.2",
    "thefuzz==0.22.1",
    "rapidfuzz==3.9.6",
    "typer>=0.12",
    "requests",
    "plotly==5.23.0",
    "openpyxl==3.1.5",
    "xlrd==2.0.1",
    "pydantic==1.10.18",
    "PyYAML==6.0.1",
    "pydataxm==0.3.6",
]

[project.optional-dependencies]
notebooks = [
    "scikit-learn==1.4.2",
    "matplotlib==3.9.1",
    "holidays",
    "cloudpickle",
    "jupyterlab==4.2.4",
]

[dependency-groups]
dev = [
    "pytest",
    "ruff==0.11.13",
]

[tool.uv]
package = false
```

Notes an implementer needs, not guesses:
- `requests` and `holidays`/`cloudpickle` have no version pin because `requirement.txt` doesn't pin them either (`requests` only appears transitively via `requests-cache==1.2.1`; `cloudpickle`/`holidays` don't appear in the freeze at all despite being imported by `dispatch.ipynb`/`ideal_dispatch.ipynb` — a pre-existing gap in the freeze, not something this task invents).
- The `notebooks` list was derived by grepping every `.ipynb` in the repo root for `import`/`from` lines, not copied from `requirement.txt` — the freeze's `Orange3`, `PyQt5`, `catboost`, `xgboost` etc. are not imported by any notebook in this repo and are deliberately excluded. If a notebook is later found to need something not listed here, add it to this group when discovered — don't pre-guess further.
- `pytest` moves to `dev`, not runtime — `README.md` section 6 currently lists it as a runtime dep; that's incorrect and gets fixed in Task 5.

- [ ] **Step 3: Sync and verify the interpreter/version**

```bash
uv sync --group dev
uv run python --version   # expect: Python 3.12.x
uv run python -c "import numpy; print(numpy.__version__)"   # expect: 1.26.4
```

- [ ] **Step 4: Run the full test suite unchanged, on the new toolchain**

```bash
uv run pytest -q
```

Expected: 78 passed. This is the checkpoint that isolates toolchain risk from the pydantic v2 migration — if this fails, the cause is Python 3.12 or a dependency version, not the validator rewrite in Task 3.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .python-version
git commit -m "build: add uv-managed pyproject.toml pinned to current freeze versions, Python 3.12"
```

---

### Task 2: `cbc` availability check on the new toolchain

**Files:**
- None created/modified — this is a verification-only task with no code deliverable, folded in here rather than given its own task because it gates Task 1's commit being trustworthy (a green pytest run with `cbc` silently unavailable would mean the MILP solves never actually ran to completion, and pytest wouldn't necessarily catch that as a failure vs. a solver error each test already expects).

**Interfaces:** none.

- [ ] **Step 1: Confirm the OS-level `cbc` binary the venv depends on is still on `PATH`**

```bash
which cbc
uv run python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('cbc').available())"
```

Expected: a path to `cbc`, and `True`. `uv` does not install `cbc` — it's an OS package (`coinor-cbc`), untouched by this migration. This step exists only to confirm the new venv still finds it (Pyomo's `SolverFactory('cbc')` shells out to whatever `cbc` is on `PATH`, independent of the Python environment).

- [ ] **Step 2: No commit** — nothing changed in this task.

---

### Task 3: Migrate `BessScenario`'s validator to pydantic v2, bump the pin

**Files:**
- Modify: `app/schemas/bess.py`
- Modify: `pyproject.toml`
- Test: `tests/test_schemas_bess.py` (already exists, already covers this validator completely — no new test file needed, see Step 1)

**Interfaces:**
- Consumes: `app/schemas/bess.py`'s current `BessMode`, `BessUnit`, `BessScenario` (unchanged field names/types — only the validator's internals change).
- Produces: same public shape. `BessScenario(mode=..., penetration_level=..., units=[...])` still raises `pydantic.ValidationError` (v2's `ValidationError`, drop-in compatible with v1's for `try`/`except` call sites — verify no call site catches `pydantic.v1.error_wrappers.ValidationError` specifically; `grep -rn "ValidationError" app/ tests/` first).

- [ ] **Step 1: Confirm the existing regression coverage**

`tests/test_schemas_bess.py` already covers this validator completely: `test_arbitrage_requires_both_bids`, `test_arbitrage_with_both_bids_is_valid`, `test_generator_requires_discharge_bid_only`, `test_grid_asset_does_not_require_bids` — all four using `pytest.raises(ValidationError, match="charge_bid"|"discharge_bid")`. No new test needed; this file is the regression guard for the migration in the next step. Run it once now to confirm it's green on the current v1 code before touching anything:

```bash
uv run pytest tests/test_schemas_bess.py -v
```

Expected: 4 passed.

**Note for the migration step below**: pydantic v2 wraps a validator's raised `ValueError(msg)` into `ValidationError` as `"Value error, {msg}"` — the original message text is preserved in the combined string, so `match="charge_bid"` (a substring `re.search`, not an exact-match) keeps passing unchanged. If any of the four tests above fail after Step 4, that's the first thing to check.

- [ ] **Step 4: Migrate the validator to pydantic v2 syntax**

```python
# app/schemas/bess.py
from enum import Enum

from pydantic import BaseModel, field_validator, ValidationInfo


class BessMode(str, Enum):
    arbitrage = "arbitrage"
    grid_asset = "grid_asset"
    generator = "generator"


class BessUnit(BaseModel):
    name: str
    mwh_nom: float
    hours_to_deplete: float
    initial_soc: float
    min_soc: float
    max_soc: float
    efficiency: float
    charge_bid: float | None = None
    discharge_bid: float | None = None


class BessScenario(BaseModel):
    mode: BessMode
    penetration_level: str
    units: list[BessUnit]

    @field_validator("units")
    @classmethod
    def _check_bids(cls, units: list[BessUnit], info: ValidationInfo) -> list[BessUnit]:
        mode = info.data.get("mode")
        for u in units:
            if mode == BessMode.arbitrage and u.charge_bid is None:
                raise ValueError(f"{u.name}: charge_bid required in mode arbitrage")
            if mode in (BessMode.arbitrage, BessMode.generator) and u.discharge_bid is None:
                raise ValueError(f"{u.name}: discharge_bid required in mode {mode.value}")
        return units
```

`mode` is declared before `units` in the class body, so pydantic v2 validates it first and `info.data["mode"]` is populated by the time `_check_bids` runs — same ordering guarantee v1's `values.get("mode")` relied on.

- [ ] **Step 5: Bump the pin and re-sync**

```toml
# pyproject.toml, in [project] dependencies
"pydantic==2.11.10",
```

```bash
uv sync --group dev
```

- [ ] **Step 6: Run the full suite**

```bash
uv run pytest -q
```

Expected: 78 passed, unchanged.

- [ ] **Step 7: Commit**

```bash
git add app/schemas/bess.py pyproject.toml
git commit -m "build: migrate BessScenario validator to pydantic v2, bump pin to 2.11.10"
```

---

### Task 4: Fix `model.py`'s dead `appsi_highs` parameter default

**Files:**
- Modify: `app/model/model.py:326,352`

**Interfaces:**
- Consumes: nothing new.
- Produces: no change to any caller's behavior — `app/pipeline/runner.py:33` already always passes `solver=case.solver` explicitly, so this default is currently unreachable in the running system. This closes the gap for any future bare `UnitCommitmentModel(...).solve()` call (including test code) so it inherits the same default as the rest of the system instead of a solver string that crashes on first use (see design spec section 3, section 6 finding #3).

- [ ] **Step 1: Change both defaults**

```python
# app/model/model.py — solve()
def solve(
    self,
    solver: str = "cbc",
    solver_params: dict = {},
    compute_prices: bool = True,
    **kwargs,
):
```

```python
# app/model/model.py — _solve_pricing_lp()
def _solve_pricing_lp(
    self, solver: str = "cbc", solver_params: dict = {}, **kwargs
):
```

- [ ] **Step 2: Run the full suite**

```bash
uv run pytest -q
```

Expected: 78 passed, unchanged — every existing call site already passes `solver="cbc"` explicitly (`tests/test_results.py:32`, `tests/test_runner.py:34-35`), so no test's behavior depends on this default.

- [ ] **Step 3: Commit**

```bash
git add app/model/model.py
git commit -m "fix: default UnitCommitmentModel.solve()'s solver param to cbc, not the crashing appsi_highs"
```

---

### Task 5: Retire `requirement.txt`, fix `README.md` section 6

**Files:**
- Delete: `requirement.txt`
- Modify: `README.md` (section 6, "Instalacion local")

**Interfaces:** none — documentation and cleanup only.

- [ ] **Step 1: Update `README.md` section 6**

Replace the `pip install -r requirement.txt` instructions with:

```markdown
## 6. Instalacion local

Requiere Python 3.12 (gestionado via `uv python install 3.12`, ver
`.python-version`) y un solver compatible con Pyomo. El flujo documentado
usa **CBC** en el `PATH`.

En Debian/Ubuntu:

\`\`\`bash
sudo apt-get install coinor-cbc
\`\`\`

Entorno Python:

\`\`\`bash
uv sync --group dev
\`\`\`

Para trabajar con los notebooks exploratorios de la raiz, agregar el extra:

\`\`\`bash
uv sync --group dev --extra notebooks
\`\`\`

Verificacion basica:

\`\`\`bash
uv run python -c "import pyomo.environ as pyo; print('cbc', pyo.SolverFactory('cbc').available())"
uv run pytest -q
\`\`\`
```

(Keep everything below section 6 — data layout, CSV schemas, etc. — unchanged; only the install instructions move.)

- [ ] **Step 2: Delete the old freeze file**

```bash
git rm requirement.txt
```

- [ ] **Step 3: Confirm nothing else references it**

```bash
grep -rn "requirement.txt" --include="*.md" --include="*.py" --include="*.sh" --include="*.yml" --include="*.yaml" .
```

Expected: no hits outside `.git/`. If CI config or a script references it, update those references before committing (none known to exist as of this plan's writing — `git status`/`find` earlier in this session found no CI config files).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: replace requirement.txt install instructions with uv, drop the freeze file"
```

---

### Task 6: Add `ruff` (blocking) + `ty` (informational) pre-commit

**Files:**
- Create: `.pre-commit-config.yaml`
- Create: `pyproject.toml` additions (`[tool.ruff]` section)

**Interfaces:** none — tooling config only.

- [ ] **Step 1: Add a minimal `[tool.ruff]` section to `pyproject.toml`**

```toml
[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I"]
```

Deliberately minimal rule set (pyflakes + pycodestyle-errors + import-sort) — this is a pre-existing, previously-unlinted codebase; a strict rule set on the first pass would produce a wall of unrelated findings that has nothing to do with Fase 2. Widening the rule set is a separate, later decision.

- [ ] **Step 2: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.13
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: local
    hooks:
      - id: ty
        name: ty (informational, non-blocking)
        entry: bash -c 'uv run ty check app/ || true'
        language: system
        pass_filenames: false
        stages: [manual]
```

`ty` runs only under `pre-commit run --hook-stage manual --all-files ty`, never on a normal commit — it's pre-alpha and this codebase has never been type-checked, so a blocking first run would wall off every commit in this phase, including this migration's own. `|| true` additionally guarantees it can never fail the hook even if invoked manually with a nonzero exit.

- [ ] **Step 3: Run ruff against the current codebase and fix what it flags**

```bash
uv run --group dev ruff check app/ tests/ --fix
uv run --group dev ruff format app/ tests/
uv run pytest -q
```

Expected: pytest still 78 passed after ruff's autofixes (import sorting, whitespace) — if `ruff format` changes behavior-relevant code (it shouldn't; it's formatting-only), that's a bug in this step, not something to paper over.

- [ ] **Step 4: Install the git hook and verify it runs**

```bash
uv run --group dev pre-commit install
uv run --group dev pre-commit run --all-files
```

Add `pre-commit` to the `dev` group in `pyproject.toml` first if not already present from Task 1 (it wasn't listed there — add it now):

```toml
[dependency-groups]
dev = [
    "pytest",
    "ruff==0.11.13",
    "pre-commit==4.6.1",
]
```

- [ ] **Step 5: Commit**

```bash
git add .pre-commit-config.yaml pyproject.toml app/ tests/
git commit -m "build: add ruff pre-commit (blocking) and ty (informational, manual-stage only)"
```

---

### Task 7: Final verification — full suite green, Global Constraints held

**Files:** none.

**Interfaces:** none.

- [ ] **Step 1: Clean-room reinstall**

```bash
rm -rf .venv
uv sync --group dev
```

- [ ] **Step 2: Full suite**

```bash
uv run pytest -q
```

Expected: 78 passed (or 79 if Task 3 added a new test).

- [ ] **Step 3: Confirm the numpy/pandas pins held through every install**

```bash
uv run python -c "import numpy, pandas; print(numpy.__version__, pandas.__version__)"
```

Expected: `1.26.4 2.2.2`. If `uv.lock` drifted either version during any earlier task's `uv sync`, that's a Global Constraints violation — pin explicitly in `pyproject.toml` and re-sync before proceeding to Fase 2B.

- [ ] **Step 4: cbc still available**

```bash
uv run python -c "import pyomo.environ as pyo; assert pyo.SolverFactory('cbc').available()"
```

- [ ] **Step 5: No commit needed** (verification only) — but if any drift was found and fixed in Step 3, commit that fix:

```bash
git add pyproject.toml uv.lock
git commit -m "build: pin numpy/pandas exactly, drift found during final Fase 2A verification"
```

**This is the exit criterion for Fase 2A.** Fase 2B (the XM fixture) starts only after this task's Step 2 and Step 3 both pass — Fase 2B's own first step re-confirms the empty-CC finding (design spec finding #6) under whatever pandas version this task actually locked, since that finding's validity depends on exact `.stack()` behavior.
