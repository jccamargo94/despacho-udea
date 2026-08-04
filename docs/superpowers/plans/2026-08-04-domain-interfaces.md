# Domain Interfaces (DispatchCase/BessScenario/InputPack/RunResult) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `DispatchConfig`/`DispatchOptions` (string-encoded dispatch type) and the untyped BESS `dict` with 4 pydantic models — `DispatchCase`, `BessScenario`, `InputPack`, `RunResult` — reused unchanged by the future API/DB layer (Fase 3 of the roadmap).

**Architecture:** New `app/schemas/{bess,case,input_pack,run_result}.py` define the models. `app/model/model.py`, `app/pipeline/{case_builder,runner,results}.py`, `app/cli.py`, and the two legacy scripts are migrated to consume them. BESS participation mode (`arbitrage`/`grid_asset`/`generator`) becomes an explicit field instead of a substring baked into a dispatch-type string.

**Tech Stack:** Python 3.11, pydantic 1.10 (already pinned in `requirement.txt` — **use v1 API**: `@validator`, not `@model_validator`/`@field_validator`), Pyomo 6.7, pytest, cbc solver via `~/.local/share/virtualenvs/dam-worker-optimizer-W9GjOqr4/bin/python`.

## Global Constraints

- Reference spec: `docs/superpowers/specs/2026-08-04-domain-interfaces-design.md`.
- Reemplazo in-place, sin shim de compatibilidad — `DispatchConfig`/`DispatchOptions`/`CaseResult` are deleted, not deprecated-and-kept.
- Pydantic v1.10 syntax only (`@validator`, `class Config` if needed) — this repo does NOT have pydantic v2.
- No Pyomo formulation for `BessMode.generator` in this plan — guarded with an explicit `NotImplementedError` raised before any solver interaction.
- Run tests with: `~/.local/share/virtualenvs/dam-worker-optimizer-W9GjOqr4/bin/python -m pytest <path> -v` (has pyomo + cbc). Tests that don't touch Pyomo can use the repo's default `python3`.
- Every task that modifies a file already covered by an existing test file updates that test file in the same task — no task should leave the suite in a state where a *later* task is needed to make its own new/modified tests pass (temporary breakage of *other*, not-yet-migrated files' tests across the repo is expected mid-plan and is called out per task).

---

### Task 1: `BessScenario` schema

**Files:**
- Create: `app/schemas/bess.py`
- Create: `app/schemas/__init__.py` (empty package init is fine for now; populated in Task 4)
- Test: `tests/test_schemas_bess.py`

**Interfaces:**
- Produces: `BessMode(str, Enum)` with members `arbitrage`, `grid_asset`, `generator`. `BessUnit(BaseModel)` fields `name: str`, `mwh_nom: float`, `hours_to_deplete: float`, `initial_soc: float`, `min_soc: float`, `max_soc: float`, `efficiency: float`, `charge_bid: float | None = None`, `discharge_bid: float | None = None`. `BessScenario(BaseModel)` fields `mode: BessMode`, `penetration_level: str`, `units: list[BessUnit]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schemas_bess.py
import pytest
from pydantic import ValidationError

from app.schemas.bess import BessMode, BessUnit, BessScenario


def _unit(**overrides):
    base = dict(
        name="B1", mwh_nom=100.0, hours_to_deplete=4.0, initial_soc=0.5,
        min_soc=0.1, max_soc=0.9, efficiency=0.92,
    )
    base.update(overrides)
    return BessUnit(**base)


def test_arbitrage_requires_both_bids():
    with pytest.raises(ValidationError, match="charge_bid"):
        BessScenario(
            mode=BessMode.arbitrage, penetration_level="10pct",
            units=[_unit(discharge_bid=50.0)],
        )
    with pytest.raises(ValidationError, match="discharge_bid"):
        BessScenario(
            mode=BessMode.arbitrage, penetration_level="10pct",
            units=[_unit(charge_bid=20.0)],
        )


def test_arbitrage_with_both_bids_is_valid():
    s = BessScenario(
        mode=BessMode.arbitrage, penetration_level="10pct",
        units=[_unit(charge_bid=20.0, discharge_bid=50.0)],
    )
    assert s.units[0].charge_bid == 20.0


def test_generator_requires_discharge_bid_only():
    with pytest.raises(ValidationError, match="discharge_bid"):
        BessScenario(
            mode=BessMode.generator, penetration_level="10pct",
            units=[_unit()],
        )
    s = BessScenario(
        mode=BessMode.generator, penetration_level="10pct",
        units=[_unit(discharge_bid=50.0)],
    )
    assert s.units[0].discharge_bid == 50.0


def test_grid_asset_does_not_require_bids():
    s = BessScenario(
        mode=BessMode.grid_asset, penetration_level="10pct",
        units=[_unit()],
    )
    assert s.units[0].charge_bid is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_schemas_bess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.schemas.bess'`

- [ ] **Step 3: Implement**

```python
# app/schemas/bess.py
from enum import Enum

from pydantic import BaseModel, validator


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

    @validator("units")
    def _check_bids(cls, units: list[BessUnit], values: dict) -> list[BessUnit]:
        mode = values.get("mode")
        for u in units:
            if mode == BessMode.arbitrage and u.charge_bid is None:
                raise ValueError(f"{u.name}: charge_bid required in mode arbitrage")
            if mode in (BessMode.arbitrage, BessMode.generator) and u.discharge_bid is None:
                raise ValueError(f"{u.name}: discharge_bid required in mode {mode.value}")
        return units
```

```python
# app/schemas/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_schemas_bess.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/schemas/bess.py app/schemas/__init__.py tests/test_schemas_bess.py
git commit -m "feat: add BessScenario schema with per-mode bid validation"
```

---

### Task 2: `InputPack` schema

**Files:**
- Create: `app/schemas/input_pack.py`
- Test: `tests/test_schemas_input_pack.py`

**Interfaces:**
- Produces: `InputSource(str, Enum)` members `historical`, `live`, `forecast`. `InputPack(BaseModel)` fields `dispatch_date: date`, `source: InputSource`, `data_dir: str`, `checksum: str | None = None`, `downloaded_at: datetime | None = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schemas_input_pack.py
from datetime import date

from app.schemas.input_pack import InputPack, InputSource


def test_input_pack_defaults():
    p = InputPack(dispatch_date=date(2024, 4, 18), source=InputSource.historical, data_dir="data")
    assert p.checksum is None
    assert p.downloaded_at is None


def test_input_pack_serializes_source_as_string():
    p = InputPack(dispatch_date=date(2024, 4, 18), source="live", data_dir="data")
    assert p.source == InputSource.live
    assert p.dict()["source"] == "live"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_schemas_input_pack.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.schemas.input_pack'`

- [ ] **Step 3: Implement**

```python
# app/schemas/input_pack.py
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel


class InputSource(str, Enum):
    historical = "historical"
    live = "live"
    forecast = "forecast"


class InputPack(BaseModel):
    dispatch_date: date
    source: InputSource
    data_dir: str
    checksum: str | None = None
    downloaded_at: datetime | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_schemas_input_pack.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/schemas/input_pack.py tests/test_schemas_input_pack.py
git commit -m "feat: add InputPack schema with source provenance tag"
```

---

### Task 3: `DispatchCase` schema

**Files:**
- Create: `app/schemas/case.py`
- Test: `tests/test_schemas_case.py`

**Interfaces:**
- Consumes: `BessScenario`, `BessMode`, `BessUnit` from `app.schemas.bess` (Task 1).
- Produces: `DispatchLevel(str, Enum)` members `preideal`, `ideal`. `DispatchCase(BaseModel)` fields `dispatch_date: date`, `level: DispatchLevel`, `bess_scenario: BessScenario | None = None`, `solver: str = "cbc"`, `compute_prices: bool = True`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schemas_case.py
from datetime import date

from app.schemas.case import DispatchCase, DispatchLevel
from app.schemas.bess import BessMode, BessScenario, BessUnit


def test_case_without_bess():
    c = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    assert c.bess_scenario is None
    assert c.solver == "cbc"
    assert c.compute_prices is True


def test_case_with_bess_scenario():
    scenario = BessScenario(
        mode=BessMode.grid_asset, penetration_level="10pct",
        units=[BessUnit(
            name="B1", mwh_nom=100.0, hours_to_deplete=4.0, initial_soc=0.5,
            min_soc=0.1, max_soc=0.9, efficiency=0.92,
        )],
    )
    c = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.ideal, bess_scenario=scenario)
    assert c.bess_scenario.mode == BessMode.grid_asset


def test_level_rejects_unknown_value():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DispatchCase(dispatch_date=date(2024, 4, 18), level="bess_ideal_resource")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_schemas_case.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.schemas.case'`

- [ ] **Step 3: Implement**

```python
# app/schemas/case.py
from datetime import date
from enum import Enum

from pydantic import BaseModel

from app.schemas.bess import BessScenario


class DispatchLevel(str, Enum):
    preideal = "preideal"
    ideal = "ideal"


class DispatchCase(BaseModel):
    dispatch_date: date
    level: DispatchLevel
    bess_scenario: BessScenario | None = None
    solver: str = "cbc"
    compute_prices: bool = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_schemas_case.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/schemas/case.py tests/test_schemas_case.py
git commit -m "feat: add DispatchCase schema, splitting level from BESS mode"
```

---

### Task 4: `RunResult` schema + `app/schemas` package exports

**Files:**
- Create: `app/schemas/run_result.py`
- Modify: `app/schemas/__init__.py`
- Test: `tests/test_schemas_run_result.py`

**Interfaces:**
- Consumes: `DispatchCase` from `app.schemas.case` (Task 3).
- Produces: `RunResult(BaseModel)` fields `case: DispatchCase`, `ok: bool`, `dispatch_path: str | None = None`, `price_path: str | None = None`, `metrics_path: str | None = None`, `metrics: dict[str, float] | None = None`, `error: str | None = None`. `app/schemas/__init__.py` re-exports `BessMode, BessUnit, BessScenario, DispatchLevel, DispatchCase, InputSource, InputPack, RunResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schemas_run_result.py
from datetime import date

from app.schemas import DispatchCase, DispatchLevel, RunResult


def test_run_result_ok():
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    r = RunResult(case=case, ok=True, dispatch_path="a.csv", price_path="b.csv",
                   metrics={"mae": 1.2})
    assert r.ok is True
    assert r.metrics["mae"] == 1.2
    assert r.error is None


def test_run_result_failure():
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.ideal)
    r = RunResult(case=case, ok=False, error="RuntimeError: boom")
    assert r.ok is False
    assert r.dispatch_path is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_schemas_run_result.py -v`
Expected: FAIL with `ImportError: cannot import name 'RunResult' from 'app.schemas'`

- [ ] **Step 3: Implement**

```python
# app/schemas/run_result.py
from pydantic import BaseModel

from app.schemas.case import DispatchCase


class RunResult(BaseModel):
    case: DispatchCase
    ok: bool
    dispatch_path: str | None = None
    price_path: str | None = None
    metrics_path: str | None = None
    metrics: dict[str, float] | None = None
    error: str | None = None
```

```python
# app/schemas/__init__.py
from app.schemas.bess import BessMode, BessUnit, BessScenario
from app.schemas.case import DispatchLevel, DispatchCase
from app.schemas.input_pack import InputSource, InputPack
from app.schemas.run_result import RunResult

__all__ = [
    "BessMode", "BessUnit", "BessScenario",
    "DispatchLevel", "DispatchCase",
    "InputSource", "InputPack",
    "RunResult",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_schemas_run_result.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/schemas/run_result.py app/schemas/__init__.py tests/test_schemas_run_result.py
git commit -m "feat: add RunResult schema and app.schemas package exports"
```

---

### Task 5: Migrate `app/model/model.py` to `DispatchCase`/`BessScenario`

**Files:**
- Modify: `app/model/model.py` (remove `DispatchOptions`/`DispatchConfig` at lines 44-58; `UnitCommitmentModel.__init__` at lines 61-66; `create_model` branching at lines 308-324; objective selection inside `_add_bess_operation` at lines 575-590)
- Modify: `app/model/__init__.py` (drops `DispatchConfig`/`DispatchOptions` re-export — those types no longer exist)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `DispatchCase`, `DispatchLevel` from `app.schemas.case`; `BessMode`, `BessScenario`, `BessUnit` from `app.schemas.bess` (Tasks 1, 3).
- Produces: `UnitCommitmentModel(case: DispatchCase)` — constructor parameter renamed from `config` to `case`. `create_model` behavior: raises `NotImplementedError` immediately if `case.bess_scenario.mode == BessMode.generator`. Thermal-feature constraints now added whenever `case.level == DispatchLevel.ideal`, regardless of BESS presence (this is a deliberate fix: the old string-matching logic silently skipped thermal constraints for `bess_ideal_resource`, since `"bess_ideal_resource" != DispatchOptions.bess_ideal`; splitting level from mode removes that bug by construction).

Note: `app/model/constraints/bess/soc.py` (`power_balance_with_bess_rule`, `same_soc_start_and_end`) reads `model._dispatch_type` as a raw string (`"bess" in ...`, `"resource" in ...`) and is **out of scope** for this plan. `UnitCommitmentModel` keeps synthesizing that exact legacy string internally so those two functions keep working unmodified.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model.py
"""UnitCommitmentModel dispatch-case branching: level -> thermal constraints,
BESS mode -> objective choice / NotImplementedError guard.

Uses the same tiny 2-generator toy fixture as test_results.py/test_runner.py,
extended with one BESS unit for the BESS-mode tests.
"""
from datetime import date

import pytest

from app.model.model import UnitCommitmentModel
from app.schemas.case import DispatchCase, DispatchLevel
from app.schemas.bess import BessMode, BessScenario, BessUnit


def _toy_sets_and_params():
    set_data = {
        "G": [], "I": ["A", "B"], "T": [1], "combined_cycle": [],
        "excluded_resource": {}, "gen_on": [], "gen_off": [],
    }
    param_data = {
        "Pmin": {("A", 1): 0.0, ("B", 1): 0.0},
        "Pmax": {("A", 1): 100.0, ("B", 1): 100.0},
        "max_min_op": 0, "ramp_up": {}, "ramp_down": {},
        "beta": {"A": 10.0, "B": 50.0}, "cold_start": {},
        "demand": {1: 130.0}, "TMG": {}, "Ton": {}, "z_on_t0_minus_1": {},
    }
    return set_data, param_data


def _toy_bess_params():
    return {
        "BESS": ["B1"],
        "bess_soc_0": {"B1": 50.0}, "bess_charge_bid": {"B1": 5.0},
        "bess_discharge_bid": {"B1": 60.0}, "bess_min_soc": {"B1": 10.0},
        "bess_max_soc": {"B1": 90.0}, "efficiency": {"B1": 0.9},
        "bess_max_charge": {"B1": 25.0}, "bess_max_discharge": {"B1": 25.0},
    }


def test_ideal_level_adds_thermal_constraints():
    set_data, param_data = _toy_sets_and_params()
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.ideal)
    m = UnitCommitmentModel(case=case)
    m.create_model(set_data=set_data, param_data=param_data)
    assert hasattr(m._model, "up_ramps_thermal_gen")


def test_preideal_level_skips_thermal_constraints():
    set_data, param_data = _toy_sets_and_params()
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    m = UnitCommitmentModel(case=case)
    m.create_model(set_data=set_data, param_data=param_data)
    assert not hasattr(m._model, "up_ramps_thermal_gen")


def test_bess_ideal_resource_still_gets_thermal_constraints():
    """Regression check for the fixed bug: BESS grid_asset + ideal must get
    thermal constraints, which the old string-matching logic skipped."""
    set_data, param_data = _toy_sets_and_params()
    set_data.update(BESS=["B1"])
    param_data.update(_toy_bess_params())
    scenario = BessScenario(
        mode=BessMode.grid_asset, penetration_level="10pct",
        units=[BessUnit(name="B1", mwh_nom=100.0, hours_to_deplete=4.0,
                         initial_soc=0.5, min_soc=0.1, max_soc=0.9, efficiency=0.9)],
    )
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.ideal, bess_scenario=scenario)
    m = UnitCommitmentModel(case=case)
    m.create_model(set_data=set_data, param_data=param_data)
    assert hasattr(m._model, "up_ramps_thermal_gen")


def test_grid_asset_mode_uses_resource_objective():
    set_data, param_data = _toy_sets_and_params()
    set_data.update(BESS=["B1"])
    param_data.update(_toy_bess_params())
    scenario = BessScenario(
        mode=BessMode.grid_asset, penetration_level="10pct",
        units=[BessUnit(name="B1", mwh_nom=100.0, hours_to_deplete=4.0,
                         initial_soc=0.5, min_soc=0.1, max_soc=0.9, efficiency=0.9)],
    )
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal, bess_scenario=scenario)
    m = UnitCommitmentModel(case=case)
    m.create_model(set_data=set_data, param_data=param_data)
    assert m._model.objective.doc == "Maximize social welfare"


def test_generator_mode_raises_not_implemented():
    set_data, param_data = _toy_sets_and_params()
    set_data.update(BESS=["B1"])
    param_data.update(_toy_bess_params())
    scenario = BessScenario(
        mode=BessMode.generator, penetration_level="10pct",
        units=[BessUnit(name="B1", mwh_nom=100.0, hours_to_deplete=4.0,
                         initial_soc=0.5, min_soc=0.1, max_soc=0.9, efficiency=0.9,
                         discharge_bid=60.0)],
    )
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal, bess_scenario=scenario)
    m = UnitCommitmentModel(case=case)
    with pytest.raises(NotImplementedError, match="generator"):
        m.create_model(set_data=set_data, param_data=param_data)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-W9GjOqr4/bin/python -m pytest tests/test_model.py -v`
Expected: FAIL — `UnitCommitmentModel(case=...)` doesn't match the current `config=` keyword / `DispatchConfig` still required.

- [ ] **Step 3: Implement**

Replace lines 44-58 of `app/model/model.py` (the `DispatchOptions`/`DispatchConfig` definitions) with an import:

```python
from app.schemas.case import DispatchCase, DispatchLevel
from app.schemas.bess import BessMode
```

Replace `UnitCommitmentModel.__init__` (lines 61-66):

```python
class UnitCommitmentModel:
    def __init__(self, case: DispatchCase):
        self._model = pyo.ConcreteModel()
        self._dispatch_case = case
        self._model._dispatch_type = self._legacy_type_tag(case)

    @staticmethod
    def _legacy_type_tag(case: DispatchCase) -> str:
        """Reconstruct the old DispatchOptions string ("bess_ideal_resource",
        etc.) for app/model/constraints/bess/soc.py, which still does
        substring matching on model._dispatch_type. Out of scope to refactor
        those rule functions in this change."""
        tag = case.level.value
        if case.bess_scenario is not None:
            tag = f"bess_{tag}"
            if case.bess_scenario.mode == BessMode.grid_asset:
                tag += "_resource"
        return tag
```

Replace the `create_model` branching (lines 308-324):

```python
    def create_model(self, set_data: dict, param_data: dict) -> None:
        self._create_sets(set_data=set_data)
        self._create_parameters(param_data=param_data)
        self._create_variables()
        self._create_objective()
        self._create_constraints()
        case = self._dispatch_case
        if case.level == DispatchLevel.ideal:
            self._create_thermal_feature_constraints(
                set_data=set_data, param_data=param_data
            )
        if case.bess_scenario is not None:
            if case.bess_scenario.mode == BessMode.generator:
                raise NotImplementedError(
                    "BESS mode 'generator' has no Pyomo formulation yet"
                )
            self._add_bess_operation(set_data=set_data, param_data=param_data)
```

In `_add_bess_operation`, replace the objective-selection block (around line 579):

```python
        if self._dispatch_case.bess_scenario.mode == BessMode.grid_asset:
            self._model.objective = pyo.Objective(
                rule=maximize_social_welfare_as_resource,
                doc=maximize_social_welfare_as_resource.__doc__,
                sense=pyo.maximize,
            )
        else:
            self._model.objective = pyo.Objective(
                rule=maximize_social_welfare,
                doc=maximize_social_welfare.__doc__,
                sense=pyo.maximize,
            )
```

(`maximize_social_welfare.__doc__` is already the string `"\n    Maximize social welfare\n    "` per the docstring in `app/model/constraints/bess/soc.py:85-88` — pyomo's `Objective.doc` strips it to `"Maximize social welfare"`; the test above asserts on that.)

Replace `app/model/__init__.py` (it still imports the now-deleted `DispatchConfig`/`DispatchOptions`, which breaks *any* import of `app.model.*` — including `test_model.py`'s own `from app.model.model import UnitCommitmentModel`, since Python always executes a package's `__init__.py` before its submodules):

```python
# app/model/__init__.py
from .model import UnitCommitmentModel

__all__ = ["UnitCommitmentModel"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-W9GjOqr4/bin/python -m pytest tests/test_model.py -v`
Expected: PASS (5 tests)

Note: `tests/test_cli.py`, `tests/test_runner.py`, `tests/test_results.py` will now fail to *import* (`app.model` no longer exports `DispatchConfig`) — expected, fixed in Tasks 7-9.

- [ ] **Step 5: Commit**

```bash
git add app/model/model.py tests/test_model.py
git commit -m "refactor: migrate UnitCommitmentModel to DispatchCase/BessScenario"
```

---

### Task 6: Pure BESS-scenario-to-Pyomo-params mapping in `case_builder.py`

**Files:**
- Modify: `app/pipeline/case_builder.py` (add function; the existing BESS block at lines 450-495 is replaced in Task 7)
- Test: `tests/test_case_builder_bess_mapping.py`

**Interfaces:**
- Consumes: `BessScenario`, `BessUnit` from `app.schemas.bess`.
- Produces: `bess_scenario_to_params(scenario: BessScenario) -> tuple[list[str], dict[str, dict[str, float]]]` — `(bess_names, param_dict)` where `param_dict` has exactly the keys `UnitCommitmentModel._add_bess_operation` reads (`bess_soc_0`, `bess_charge_bid`, `bess_discharge_bid`, `bess_min_soc`, `bess_max_soc`, `efficiency`, `bess_max_charge`, `bess_max_discharge`).

This isolates the one piece of `case_builder.py`'s BESS handling that's pure data transformation (no XM I/O), so it can be unit-tested directly — `case_builder.py`'s XM-loading path stays untested against real data, matching the existing, already-known gap (see project memory `cli-app-verification-status`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_case_builder_bess_mapping.py
from app.pipeline.case_builder import bess_scenario_to_params
from app.schemas.bess import BessMode, BessScenario, BessUnit


def test_maps_units_to_pyomo_param_dicts():
    scenario = BessScenario(
        mode=BessMode.arbitrage, penetration_level="10pct",
        units=[
            BessUnit(name="B1", mwh_nom=100.0, hours_to_deplete=4.0, initial_soc=0.5,
                      min_soc=0.1, max_soc=0.9, efficiency=0.92,
                      charge_bid=20.0, discharge_bid=60.0),
            BessUnit(name="B2", mwh_nom=50.0, hours_to_deplete=2.0, initial_soc=1.0,
                      min_soc=0.0, max_soc=1.0, efficiency=0.85,
                      charge_bid=15.0, discharge_bid=55.0),
        ],
    )
    names, params = bess_scenario_to_params(scenario)

    assert names == ["B1", "B2"]
    assert params["bess_soc_0"] == {"B1": 50.0, "B2": 50.0}
    assert params["bess_min_soc"] == {"B1": 10.0, "B2": 0.0}
    assert params["bess_max_soc"] == {"B1": 90.0, "B2": 50.0}
    assert params["bess_max_charge"] == {"B1": 25.0, "B2": 25.0}
    assert params["bess_max_discharge"] == {"B1": 25.0, "B2": 25.0}
    assert params["efficiency"] == {"B1": 0.92, "B2": 0.85}
    assert params["bess_charge_bid"] == {"B1": 20.0, "B2": 15.0}
    assert params["bess_discharge_bid"] == {"B1": 60.0, "B2": 55.0}


def test_grid_asset_scenario_omits_absent_bids():
    scenario = BessScenario(
        mode=BessMode.grid_asset, penetration_level="10pct",
        units=[BessUnit(name="B1", mwh_nom=100.0, hours_to_deplete=4.0,
                          initial_soc=0.5, min_soc=0.1, max_soc=0.9, efficiency=0.9)],
    )
    _, params = bess_scenario_to_params(scenario)
    assert params["bess_charge_bid"] == {}
    assert params["bess_discharge_bid"] == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_case_builder_bess_mapping.py -v`
Expected: FAIL with `ImportError: cannot import name 'bess_scenario_to_params'`

- [ ] **Step 3: Implement**

Add to `app/pipeline/case_builder.py` (module level, near the top after imports — do not remove anything yet, Task 7 wires it in):

```python
from app.schemas.bess import BessScenario


def bess_scenario_to_params(scenario: BessScenario) -> tuple[list[str], dict]:
    """Map a BessScenario's units to the pyomo-level set/param dicts consumed
    by UnitCommitmentModel._add_bess_operation. Mirrors the historical bess
    dict shape 1:1 (initial_soc/min_soc/max_soc are fractions of mwh_nom;
    max_charge/max_discharge = mwh_nom / hours_to_deplete)."""
    names = [u.name for u in scenario.units]
    params: dict[str, dict] = {
        "bess_soc_0": {}, "bess_charge_bid": {}, "bess_discharge_bid": {},
        "bess_min_soc": {}, "bess_max_soc": {}, "efficiency": {},
        "bess_max_charge": {}, "bess_max_discharge": {},
    }
    for u in scenario.units:
        params["bess_soc_0"][u.name] = u.initial_soc * u.mwh_nom
        params["bess_min_soc"][u.name] = u.min_soc * u.mwh_nom
        params["bess_max_soc"][u.name] = u.max_soc * u.mwh_nom
        params["efficiency"][u.name] = u.efficiency
        params["bess_max_charge"][u.name] = u.mwh_nom / u.hours_to_deplete
        params["bess_max_discharge"][u.name] = u.mwh_nom / u.hours_to_deplete
        if u.charge_bid is not None:
            params["bess_charge_bid"][u.name] = u.charge_bid
        if u.discharge_bid is not None:
            params["bess_discharge_bid"][u.name] = u.discharge_bid
    return names, params
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_case_builder_bess_mapping.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/case_builder.py tests/test_case_builder_bess_mapping.py
git commit -m "feat: add pure BessScenario-to-pyomo-params mapping, unit tested"
```

---

### Task 7: Wire `build_case` to `(DispatchCase, InputPack)`

**Files:**
- Modify: `app/pipeline/case_builder.py` (signature at lines 28-45; `config.dispatch_type == "ideal"` at line 50; `"preideal" in config.dispatch_type` at lines 377, 390, 393; BESS block at lines 450-495)

**Interfaces:**
- Consumes: `DispatchCase`, `DispatchLevel` from `app.schemas.case`; `InputPack` from `app.schemas.input_pack`; `bess_scenario_to_params` from Task 6.
- Produces: `build_case(case: DispatchCase, inputs: InputPack, *, ders: int | None = None) -> tuple[dict, dict, dict]` — same `(set_data, param_data, meta)` return shape as before. `ders` stays a separate keyword (expansion-resource scenarios are outside the 4 frozen interfaces per spec scope).

No dedicated test in this task: `build_case`'s XM-loading path has no fixtures today (needs real `data/`, per project memory `cli-app-verification-status` — this task doesn't change that status, it only re-types the signature). Verified by Task 6's unit test for the BESS sub-piece and by the full suite at the end of Task 9.

- [ ] **Step 1: Update the signature and imports**

Replace lines 21 and 28-45:

```python
from app.schemas.case import DispatchCase, DispatchLevel
from app.schemas.input_pack import InputPack
```

```python
def build_case(
    case: DispatchCase,
    inputs: InputPack,
    *,
    ders: int | None = None,
) -> tuple[dict, dict, dict]:
    """Return (set_data, param_data, meta) for `UnitCommitmentModel`.

    meta keys: timestamps, precio_bolsa, CC, initial_condition_df,
    major_generators, generators, fixed_fuel_fire, pmax_new_resources,
    expansion_sources.
    """
    DISPATCH_DATE = case.dispatch_date
    DERS = ders
    dd = inputs.data_dir

    ensure_data_for_date(DISPATCH_DATE, data_dir=dd)
```

- [ ] **Step 2: Replace the level checks**

Line 50: `if config.dispatch_type == "ideal":` → `if case.level == DispatchLevel.ideal:`

Lines 377, 390, 393: `"preideal" in config.dispatch_type` → `case.level == DispatchLevel.preideal`

- [ ] **Step 3: Replace the BESS block (lines 450-495)**

```python
    if case.bess_scenario is not None:
        bess_names, bess_params_model = bess_scenario_to_params(case.bess_scenario)
        set_data.update(BESS=bess_names)
        param_data.update(**bess_params_model)
```

(delete the old `BESS_PARAMS_NAMES` loop entirely — replaced by the Task 6 helper.)

- [ ] **Step 4: Sanity-check imports resolve**

Run: `python3 -c "import app.pipeline.case_builder"`
Expected: no `ImportError` (this only proves the module parses/imports; `build_case` itself can't run without real `data/`).

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/case_builder.py
git commit -m "refactor: build_case takes (DispatchCase, InputPack) instead of loose args"
```

---

### Task 8: Migrate `runner.py` and `results.py` to `RunResult`

**Files:**
- Modify: `app/pipeline/runner.py` (delete `CaseResult` dataclass at lines 19-26; rewrite `run_case`/`run_many` at lines 29-96)
- Modify: `app/pipeline/results.py` (`save_results` at lines 33-51)
- Modify: `tests/test_runner.py`
- Modify: `tests/test_results.py`

**Interfaces:**
- Consumes: `DispatchCase`, `RunResult`, `InputPack`, `InputSource` from `app.schemas`; `build_case` from Task 7; `bess_scenario_to_params` unaffected.
- Produces: `run_case(case: DispatchCase, *, evaluate: bool = True, input_source: InputSource = InputSource.historical, ders: int | None = None, out: str = "data/results", data_dir: str = "data") -> RunResult`. `run_many(cases: list[DispatchCase], *, out: str = "data/results", **kw) -> list[RunResult]`. `save_results(model, case: DispatchCase, out: str = "data/results") -> RunResult` (now returns a `RunResult` with `ok=True` and paths filled in, instead of a bare `dict`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_runner.py  (full replacement)
"""Runner orchestration: a failing case must not abort the batch."""
from datetime import date

import app.pipeline.runner as runner
from app.schemas import DispatchCase, DispatchLevel


def _toy_case():
    set_data = {
        "G": [], "I": ["A", "B"], "T": [1], "combined_cycle": [],
        "excluded_resource": {}, "gen_on": [], "gen_off": [],
    }
    param_data = {
        "Pmin": {("A", 1): 0.0, ("B", 1): 0.0},
        "Pmax": {("A", 1): 100.0, ("B", 1): 100.0},
        "max_min_op": 0, "ramp_up": {}, "ramp_down": {},
        "beta": {"A": 10.0, "B": 50.0}, "cold_start": {},
        "demand": {1: 150.0}, "TMG": {}, "Ton": {}, "z_on_t0_minus_1": {},
    }
    return set_data, param_data, {}


def test_failure_isolated(monkeypatch, tmp_path):
    good, bad = date(2024, 4, 18), date(2024, 4, 19)

    def fake_build(case, inputs, **kw):
        if case.dispatch_date == bad:
            raise RuntimeError("boom")
        return _toy_case()

    monkeypatch.setattr(runner, "build_case", fake_build)

    cases = [
        DispatchCase(dispatch_date=good, level=DispatchLevel.preideal, solver="cbc"),
        DispatchCase(dispatch_date=bad, level=DispatchLevel.preideal, solver="cbc"),
    ]
    results = runner.run_many(cases, evaluate=False, out=str(tmp_path))
    assert len(results) == 2
    ok = {r.case.dispatch_date: r.ok for r in results}
    assert ok[good] is True
    assert ok[bad] is False
    bad_r = next(r for r in results if r.case.dispatch_date == bad)
    assert "boom" in bad_r.error
```

```python
# tests/test_results.py  (full replacement)
"""Results extraction tested against a tiny solvable model.

Cheap gen A (beta=10, Pmax=100), expensive B (beta=50). demand=150 -> A=100,
B=50; marginal unit is B so MPO should equal B's cost (50).
"""
from datetime import date

from app.model.model import UnitCommitmentModel
from app.schemas import DispatchCase, DispatchLevel
from app.pipeline.results import extract_mpo, extract_dispatch, save_results


def _toy_model():
    set_data = {
        "G": [], "I": ["A", "B"], "T": [1], "combined_cycle": [],
        "excluded_resource": {}, "gen_on": [], "gen_off": [],
    }
    param_data = {
        "Pmin": {("A", 1): 0.0, ("B", 1): 0.0},
        "Pmax": {("A", 1): 100.0, ("B", 1): 100.0},
        "max_min_op": 0, "ramp_up": {}, "ramp_down": {},
        "beta": {"A": 10.0, "B": 50.0}, "cold_start": {},
        "demand": {1: 150.0}, "TMG": {}, "Ton": {}, "z_on_t0_minus_1": {},
    }
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    m = UnitCommitmentModel(case=case)
    m.create_model(set_data=set_data, param_data=param_data)
    m.solve(solver="cbc")
    return m, case


def test_extract_mpo_is_marginal_cost():
    m, _ = _toy_model()
    mpo = extract_mpo(m)
    assert len(mpo) == 1
    assert abs(list(mpo.values())[0] - 50.0) < 1e-6


def test_extract_dispatch_rows():
    m, _ = _toy_model()
    df = extract_dispatch(m)
    assert set(df.columns) == {"generador", "datetime", "dispatch"}
    by_gen = df.set_index("generador")["dispatch"].to_dict()
    assert abs(by_gen["A"] - 100.0) < 1e-6
    assert abs(by_gen["B"] - 50.0) < 1e-6


def test_save_results_writes_csvs(tmp_path):
    m, case = _toy_model()
    result = save_results(m, case, out=str(tmp_path))
    assert (tmp_path / f"dispatch_by_gen-{case.dispatch_date}-{case.level.value}.csv").exists()
    assert (tmp_path / f"marginal_price-{case.dispatch_date}-{case.level.value}.csv").exists()
    assert result.ok is True
    assert result.dispatch_path is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-W9GjOqr4/bin/python -m pytest tests/test_runner.py tests/test_results.py -v`
Expected: FAIL (old `save_results`/`run_many` signatures, `RunResult` not returned).

- [ ] **Step 3: Implement `results.py`**

Replace `save_results` (lines 33-51):

```python
from app.schemas import DispatchCase, RunResult


def save_results(model, case: DispatchCase, out: str = "data/results") -> RunResult:
    Path(out).mkdir(parents=True, exist_ok=True)
    t = case.level.value

    dispatch = extract_dispatch(model)
    dispatch_path = f"{out}/dispatch_by_gen-{case.dispatch_date}-{t}.csv"
    dispatch.to_csv(dispatch_path, sep=",", index=False)

    mpo = extract_mpo(model)
    price_path = f"{out}/marginal_price-{case.dispatch_date}-{t}.csv"
    pd.DataFrame(
        data=mpo.values(), index=mpo.keys(), columns=["ideal_marginal_price"]
    ).reset_index(drop=False, names=["datetime"]).to_csv(
        price_path, sep=",", index=False
    )

    return RunResult(case=case, ok=True, dispatch_path=dispatch_path, price_path=price_path)
```

Remove the now-unused `from app.model import DispatchConfig` import at the top of `results.py`; `extract_mpo`/`extract_dispatch` are untouched.

- [ ] **Step 4: Implement `runner.py`**

Full replacement:

```python
"""Orchestrate dispatch runs: ensure data -> build -> solve -> save -> evaluate.

Per-case failures are isolated: one bad case does not abort the batch.
"""
import traceback

import pandas as pd

from app.model.model import UnitCommitmentModel
from app.schemas import DispatchCase, InputPack, InputSource, RunResult
from app.pipeline.case_builder import build_case
from app.pipeline.results import save_results
from app.data.actuals import load_actual_price
from app.utils.metrics import price_metrics


def run_case(
    case: DispatchCase,
    *,
    evaluate: bool = True,
    input_source: InputSource = InputSource.historical,
    ders: int | None = None,
    out: str = "data/results",
    data_dir: str = "data",
) -> RunResult:
    t = case.level.value
    try:
        inputs = InputPack(dispatch_date=case.dispatch_date, source=input_source, data_dir=data_dir)
        set_data, param_data, _meta = build_case(case, inputs, ders=ders)
        model = UnitCommitmentModel(case=case)
        model.create_model(set_data=set_data, param_data=param_data)
        model.solve(solver=case.solver, compute_prices=case.compute_prices)
        result = save_results(model, case, out=out)

        if evaluate:
            try:
                xm = load_actual_price(case.dispatch_date, data_dir=data_dir)
                model_mpo = extract_mpo_sorted(model)
                n = min(len(xm), len(model_mpo))
                metrics = price_metrics(xm[:n], model_mpo[:n])
                metrics_path = f"{out}/metrics-{case.dispatch_date}-{t}.csv"
                pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
                result.metrics = metrics
                result.metrics_path = metrics_path
            except FileNotFoundError:
                print(f"  ! no XM actuals for {case.dispatch_date}; skipping metrics")

        return result
    except Exception as e:
        traceback.print_exc()
        return RunResult(case=case, ok=False, error=f"{type(e).__name__}: {e}")


def extract_mpo_sorted(model) -> list[float]:
    from app.pipeline.results import extract_mpo
    mpo = extract_mpo(model)
    return [v for _, v in sorted(mpo.items())]


def run_many(
    cases: list[DispatchCase],
    *,
    out: str = "data/results",
    **kw,
) -> list[RunResult]:
    results: list[RunResult] = []
    for case in cases:
        print(f"==> {case.dispatch_date} [{case.level.value}]")
        results.append(run_case(case, out=out, **kw))

    rows = [
        {"date": r.case.dispatch_date, "type": r.case.level.value, **r.metrics}
        for r in results
        if r.ok and r.metrics
    ]
    if rows:
        from pathlib import Path

        Path(out).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(f"{out}/metrics-summary.csv", index=False)
    return results
```

(`extract_mpo_sorted` replaces the inline `[v for _, v in sorted(paths["mpo"].items())]` from the old code, which relied on `save_results`'s returned `dict` carrying `"mpo"` — `RunResult` doesn't carry raw MPO values, only the CSV path, so `run_case` re-derives the sorted list straight from the model via `extract_mpo`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-W9GjOqr4/bin/python -m pytest tests/test_runner.py tests/test_results.py -v`
Expected: PASS (4 tests total)

Note: `tests/test_cli.py` still fails at this point (Task 9 fixes it).

- [ ] **Step 6: Commit**

```bash
git add app/pipeline/runner.py app/pipeline/results.py tests/test_runner.py tests/test_results.py
git commit -m "refactor: runner/results produce RunResult instead of CaseResult+dict"
```

---

### Task 9: Migrate `app/cli.py` to `DispatchCase`

**Files:**
- Modify: `app/cli.py` (imports at line 13; `run` command body at lines 44-89)
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `DispatchCase`, `DispatchLevel` from `app.schemas`; `run_many` from Task 8.
- Produces: no new public interface — `python -m app run` keeps its existing flags/behavior. `--type/-t` now selects `DispatchLevel` values (`preideal`, `ideal`, or `all` for both) instead of the old 6-way `DispatchOptions`; BESS scenarios aren't exposed on the CLI yet (Fase 1 per the spec's "Fuera de este cambio").

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py  (full replacement)
from datetime import date

from typer.testing import CliRunner

import app.cli as cli
from app.schemas import DispatchCase, DispatchLevel, RunResult

runner = CliRunner()


def _stub_dates(monkeypatch):
    monkeypatch.setattr(cli, "_available_dates", lambda data_dir: [date(2024, 4, 18)])


def test_run_success(monkeypatch):
    _stub_dates(monkeypatch)
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    monkeypatch.setattr(
        cli, "run_many",
        lambda *a, **k: [RunResult(case=case, ok=True)],
    )
    result = runner.invoke(cli.app, ["run", "2024-04-18", "-t", "preideal"])
    assert result.exit_code == 0
    assert "1 ok, 0 failed" in result.output


def test_run_reports_failure(monkeypatch):
    _stub_dates(monkeypatch)
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    monkeypatch.setattr(
        cli, "run_many",
        lambda *a, **k: [RunResult(case=case, ok=False, error="X")],
    )
    result = runner.invoke(cli.app, ["run", "2024-04-18"])
    assert result.exit_code == 1
    assert "1 failed" in result.output


def test_no_dates_selected(monkeypatch):
    monkeypatch.setattr(cli, "_available_dates", lambda data_dir: [])
    result = runner.invoke(cli.app, ["run", "2024-05-01:2024-05-02"])
    assert result.exit_code == 1
    assert "No dates selected" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cli.py -v`
Expected: FAIL (`app.cli` still imports `DispatchConfig`/`DispatchOptions`, which no longer exist).

- [ ] **Step 3: Implement**

Replace the import at line 13:

```python
from app.schemas import DispatchCase, DispatchLevel
```

Replace the `run` command body (lines 61-89):

```python
    avail = _available_dates(data_dir)
    selected = parse_dates_arg(dates, avail)
    skip = _parse_skip(skip_dates)
    selected = [d for d in selected if d not in skip]

    levels = list(DispatchLevel) if "all" in type else [DispatchLevel(t) for t in type]
    cases = [
        DispatchCase(dispatch_date=d, level=lvl, solver=solver, compute_prices=prices)
        for d in selected
        for lvl in levels
    ]

    if not cases:
        typer.echo("No dates selected.")
        raise typer.Exit(code=1)

    typer.echo(
        f"Running {len(selected)} date(s) x {len(levels)} level(s) with solver={solver}"
    )
    results = run_many(cases, evaluate=eval, out=out, data_dir=data_dir)
    failed = [r for r in results if not r.ok]
    typer.echo(f"\nDone: {len(results) - len(failed)} ok, {len(failed)} failed.")
    for r in failed:
        typer.echo(f"  FAIL {r.case.dispatch_date} [{r.case.level.value}]: {r.error}")
    raise typer.Exit(code=1 if failed else 0)
```

Update the `type` option's default/help text (line 49) from `["preideal"]` (unchanged) to clarify it's a level:

```python
    type: list[str] = typer.Option(
        ["preideal"], "--type", "-t", help="dispatch level (preideal/ideal), repeatable, or 'all'"
    ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cli.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full suite**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-W9GjOqr4/bin/python -m pytest tests/ -v`
Expected: PASS — every test file in `tests/` is green now (this is the first point since Task 5 where the whole suite is consistent again).

- [ ] **Step 6: Commit**

```bash
git add app/cli.py tests/test_cli.py
git commit -m "refactor: CLI builds DispatchCase per date x level instead of DispatchConfig"
```

---

### Task 10: Migrate legacy scripts `run_dispatch.py` / `get_date_results.py`

**Files:**
- Modify: `run_dispatch.py` (imports at line 20; `run_dispatch()` signature and body at lines 25-34)
- Modify: `get_date_results.py` (imports at line 12; `main()` at lines 49-52)

**Interfaces:**
- Consumes: `DispatchCase`, `DispatchLevel` from `app.schemas`; `build_case`, `run_many` (Tasks 7, 8).
- No new interfaces produced — these are thin legacy wrappers kept only so old notebooks that call `run_dispatch()` keep working.

- [ ] **Step 1: Update `run_dispatch.py`**

Replace line 20:

```python
from app.schemas.case import DispatchCase
```

Replace the function signature and `build_case`/`UnitCommitmentModel` calls (lines 25-46):

```python
def run_dispatch(
    case: DispatchCase,
    show_figs: bool = False,
    BESS: dict | None = None,
    DERS: int | None = None,
):
    from app.schemas.input_pack import InputPack, InputSource

    inputs = InputPack(dispatch_date=case.dispatch_date, source=InputSource.historical, data_dir="data")
    set_data, param_data, meta = build_case(case, inputs, ders=DERS)
    precio_bolsa = meta["precio_bolsa"]
    CC = meta["CC"]
    initial_condition_df = meta["initial_condition_df"]
    major_generators = meta["major_generators"]
    generators = meta["generators"]
    fixed_fuel_fire = meta["fixed_fuel_fire"]
    pmax_new_resources = meta["pmax_new_resources"]
    expansion_sources = meta["expansion_sources"]

    # ## 1.9 Solving model
    model = UnitCommitmentModel(case=case)
    model.create_model(set_data=set_data, param_data=param_data)

    results = model.solve(solver="cbc")
```

Note: this legacy signature drops raw `bess: dict` support — callers that need BESS must build a `BessScenario` and set it on `case.bess_scenario` before calling `build_case`/`run_dispatch`, same as everywhere else in the migrated code. The `BESS` parameter is left in the signature (unused) only so existing notebook call sites that pass `BESS=...` positionally/by-keyword don't hard-crash with a `TypeError`; it's not wired to anything — flag this to the user if any notebook actually relies on it.

- [ ] **Step 2: Update `get_date_results.py`**

Replace line 12:

```python
from app.schemas import DispatchCase, DispatchLevel
```

Replace `main()` (lines 49-52):

```python
def main():
    dates_ = [d for d in discover_dates() if d not in SKIP_DATES]
    cases = [
        DispatchCase(dispatch_date=d, level=lvl)
        for d in dates_
        for lvl in DispatchLevel
    ]
    results = run_many(cases)
    failed = [r for r in results if not r.ok]
    print(f"\nDone: {len(results) - len(failed)} ok, {len(failed)} failed.")
    for r in failed:
        print(f"  FAIL {r.case.dispatch_date} [{r.case.level.value}]: {r.error}")
```

This changes behavior from iterating all 6 old `DispatchOptions` (4 of which passed `bess=None` into `build_case`, which would `AttributeError` on `None.keys()` the moment BESS-specific code ran — already broken before this plan) to iterating the 2 working `DispatchLevel` values. Equivalent *working* behavior is preserved; BESS batch runs need an explicit `BessScenario`, which this script doesn't have data for (out of scope, same as noted in the spec's Fase 1 follow-up).

- [ ] **Step 3: Verify both scripts import cleanly**

Run: `python3 -c "import ast; ast.parse(open('run_dispatch.py').read()); ast.parse(open('get_date_results.py').read())"`
Expected: no `SyntaxError` (full runtime import needs `thefuzz`/`plotly`/pandas installed — parse-check is the practical smoke test here; both scripts are notebook-invoked, not covered by `tests/`).

- [ ] **Step 4: Commit**

```bash
git add run_dispatch.py get_date_results.py
git commit -m "refactor: migrate legacy run_dispatch/get_date_results to DispatchCase"
```

---

## Post-plan follow-ups (not part of this plan)

- `app/model/constraints/bess/soc.py` still does substring matching on the synthesized legacy `_dispatch_type` string — candidate for a follow-up cleanup once nothing else needs it.
- Fase 1 (roadmap): expose `BessScenario` on the CLI via a YAML/JSON scenario file.
- `case_builder.py`'s XM-loading path (everything except `bess_scenario_to_params`) is still unverified against real data — unchanged status from before this plan.
