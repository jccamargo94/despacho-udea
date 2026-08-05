# Fase 2B: Fixture XM real + smoke test end-to-end Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a synthetic-but-real-format XM fixture that survives `case_builder.build_case` end to end for `level=preideal`, and prove it with three layers of tests (loader-level, `build_case`-level, full `run_case`/CLI-level) — closing the Fase 0 gap ("verify `case_builder` with real data") with synthetic data, and giving Fase 2C a fixture it can run inside Docker.

**Architecture:** One committed fixture directory (`tests/fixtures/xm_smoke/`) generated once by a small deterministic script and checked in as static files (not regenerated at test time). Two synthetic thermal generators, no BESS, no DERS, no combined-cycle. Tests are layered by risk: loader/parser sanity first (cheapest to debug), then `build_case` value assertions (the layer most likely to break per the continuation prompt's own warning), then a full `cbc` solve via `run_case`, then the literal CLI command from the spec's exit criterion.

**Tech Stack:** Existing toolchain from Fase 2A (`uv`, Python 3.12, `pandas==2.2.2`, `pyomo`, `cbc`). No new dependencies.

## Global Constraints

- **Order dependency**: this plan assumes Fase 2A is merged and `uv run pytest -q` passes 78/78 on the current `develop` tip before starting.
- **Layout finding (new, not in the spec doc — verified this session, fold in here rather than reopening the spec)**: `ensure_data_for_date` (`app/data/download.py:71-80`) short-circuits on `storage.list_dir(str(dispatch_date))` — i.e. it checks whether `data_dir/{fecha}/` (the *flat* per-date layout) is non-empty. `resolve_input` (`app/data/paths.py:34-49`) tries the *organized* subdirectory first, then falls back to the same flat `{fecha}/` folder. Putting `OFEI{mmdd}.txt` and `PrId{mmdd}_NAL.txt` directly in `data_dir/{fecha}/` satisfies both checks with one directory. `dCondIniP{mmdd}.txt`/`dCondIniU{mmdd}.txt` go under `condicion_inicial/{fecha}/` (the organized layout, `resolve_input`'s preferred candidate) — this also makes `cli.py:_available_dates` (`app/cli.py:29-40`, reads `list_dir("condicion_inicial")`) recognize the fixture date, for free. **Verified by direct execution this session**: with exactly this layout, `build_case` printed `"... files already downloaded. Skipping download"` and no `requests.get` call occurred.
- **Unit conventions, verified by direct execution, not inferred** (this is the same class of bug flagged in Fase 1 — MPO ×1000):
  - `dispo_declarada.csv` → `dispo` column is **kW**; `case_builder.py:314-319` multiplies by `1e-3` to get MW.
  - `ofertas.csv` → `Value` column is **COP/kWh**; `case_builder.py:322-326` (`beta`) multiplies by `1e3` to get COP/MWh.
  - `PrId{mmdd}_NAL.txt` → hourly values are used **raw, in MW, no scaling** (`case_builder.py:330-332`, `demand_pronos`). This is `DEMANDA` for `level=preideal`. Do not scale it like `dispo`.
  - `precio_bolsa/precio_bolsa_2024.csv` → loader multiplies by `1e3` (`app/data/loaders.py:46`), but this value only reaches `meta["precio_bolsa"]`, never `set_data`/`param_data` — content is cosmetic for this fixture.
- **Dead-but-present computations, verified by grep — do not spend fixture-tuning effort on these**: `Pmin` (`case_builder.py:320`) is computed from `minimo_operativo` (OFEI `MO` lines) but the returned `param_data["Pmin"]` is hardcoded to `{}` (`case_builder.py:438`) — every `(i,t)` pair gets Pyomo's `default=param_data["max_min_op"]` instead (`app/model/model.py:96`). `agc_indexed` (`case_builder.py:327`) is computed and never read again. `fixed_fuel_fired_map` (`case_builder.py:352-361`) is computed and never read again. MO/PAP lines and `agc_asignado.csv` still need to exist and parse cleanly (guards below), just don't need meaningful values.
- **Guards that crash on empty/mismatched fuzzy matches** (`case_builder.py`): `price_bid_map` (:135), `CC_MAP` (:183), `generators_pap_map` (:293) all use `thefuzz.process.extractOne(..., score_cutoff=70)[0]` — a match below 70 returns `None`, and `[0]` on `None` raises `TypeError`. `precio_arranque` must have ≥1 row per thermal generator with `type` containing `"C"`, or `case_builder.py:307-310`'s `.values[0]` raises `IndexError` on an empty slice. This plan avoids the risk entirely by using exact-identical resource names (`TERMO1`, `TERMO2`) across every file — a 100-score match is not a fuzzy-match risk.
- **No combined-cycle, no DERS, no BESS.** `ders=None` (build_case default) skips the `Supuestos Modelo de despacho.xlsx` dependency entirely. `CC={}` (no OFEI lines matching `"CC" in line`) verified safe under `pandas==2.2.2` by direct execution: `pd.DataFrame({}).stack().reset_index()` succeeds, and the full CC-synthesis block (`case_builder.py:183-217`) completes without error on an empty `CC`. `preideal_dispatch_map.json = {}` also verified safe by direct execution (empty-frame `set_index().to_dict()` path).
- **`evaluate=True` (the CLI default) is safe without an XM actuals file.** `run_case` (`app/pipeline/runner.py:38-44`) only catches `FileNotFoundError` around `load_actual_price`; `load_actual_price` (`app/data/actuals.py:11-16`) opens `preideal_price/{date}.txt` via a bare `open()` — a missing file raises exactly `FileNotFoundError`, caught cleanly, printed as `"no XM actuals for ...; skipping metrics"`, `result.ok` stays `True`. No `preideal_price/` file needed in the fixture.
- **Known harmless noise**: `case_builder.py:217`'s `pd.concat([ofertas, new_cc_bid], ...)` emits a pandas `FutureWarning` about empty-frame concatenation whenever `CC={}` (which is always, in this fixture). This is pre-existing `case_builder.py` behavior, out of scope for this plan (no `pytest` `filterwarnings` config exists in `pyproject.toml`, so it does not fail tests). Do not "fix" it as a side quest.
- **Fixture values below are already verified end-to-end** (not proposed — this plan documents a design that was built and run against `build_case`, `run_case`, and the literal CLI command in this session, with real `cbc`, before being written down). Task 1-4 steps reproduce exactly that verified run.
- **`.gitignore:35` has a blanket `*.csv` rule** (leftover from excluding `data/` output CSVs) that silently swallows every CSV under `tests/fixtures/xm_smoke/` — verified with `git check-ignore -v`. Task 1 adds a scoped negation (`!tests/fixtures/**/*.csv`) so the fixture actually gets committed; without it, every test in this plan still passes locally (the files exist on disk) while CI and Fase 2C's container get nothing. Always sanity-check with `git add --dry-run` after generating fixture files, not just `git status`.
- **All fixture-path references in test files must be anchored on `Path(__file__).parent`, not a cwd-relative string literal.** `tests/test_ofei.py:6` already establishes this convention (`FIX = Path(__file__).parent / "fixtures" / "OFEI_sample.txt"`). A literal `"tests/fixtures/xm_smoke"` string only resolves correctly when pytest/the subprocess is invoked from the repo root — verified to break (`FileNotFoundError` / `No module named app`) when run from a different cwd. This is exactly the "works on my machine" failure class this plan exists to catch before Fase 2C.
- Reference: `docs/superpowers/specs/2026-08-05-fase2-docker-design.md` section 2 ("Fixture XM real") and section 6 (findings #6, #7).

---

### Task 1: Fixture files (`tests/fixtures/xm_smoke/`) + loader-level sanity test

**Files:**
- Create: `tests/fixtures/xm_smoke/generate_fixture.py` (one-off generator, committed for provenance/regeneration)
- Create (generated by the script above, then committed as static files): `tests/fixtures/xm_smoke/dispo_declarada.csv`, `ofertas.csv`, `demaCome.csv`, `agc_asignado.csv`, `parametros_plantas.csv`, `precio_bolsa/precio_bolsa_2024.csv`, `ramps.json`, `preideal_dispatch_map.json`, `2024-04-18/OFEI0418.txt`, `2024-04-18/PrId0418_NAL.txt`, `condicion_inicial/2024-04-18/dCondIniP0418.txt`, `condicion_inicial/2024-04-18/dCondIniU0418.txt`
- Modify: `.gitignore` (scope the blanket `*.csv` rule so fixture CSVs aren't silently dropped)
- Test: `tests/test_xm_smoke_loaders.py`

**Interfaces:**
- Produces: a fixture directory usable as `data_dir` by every `app.data.loaders`/`app.data.ofei`/`app.data.paths` function, and by `ensure_data_for_date` as a no-op (no network). Two thermal generators: `TERMO1` (300 MW capacity, initially on at 150 MW), `TERMO2` (200 MW capacity, initially off). Flat demand: 350 MW every hour (below the 500 MW combined capacity, above `TERMO1` alone — forces both generators committed).

- [ ] **Step 1: Write the fixture generator script**

```python
# tests/fixtures/xm_smoke/generate_fixture.py
"""One-off generator for the Fase 2B smoke-test fixture.

Run once (`uv run python tests/fixtures/xm_smoke/generate_fixture.py` from
repo root) and commit the output alongside this script. Not run at test
time — the fixture files it produces are the actual test input.
"""

import csv
from datetime import date, datetime, timedelta
from pathlib import Path

FECHA = date(2024, 4, 18)
MMDD = "0418"
BASE = Path(__file__).parent
HOURS = [datetime(2024, 4, 18) + timedelta(hours=h) for h in range(24)]

GENERATORS = [
    {
        "name": "TERMO1", "dispo_kw": 300_000, "bid_cop_kwh": 150, "pap_cop": 1_500_000,
        "mo": 10, "gpini": 150, "conf": "CONF1", "tconf": 5,
    },
    {
        "name": "TERMO2", "dispo_kw": 200_000, "bid_cop_kwh": 180, "pap_cop": 1_500_000,
        "mo": 5, "gpini": 0, "conf": "CONF0", "tconf": 0,
    },
]

with open(BASE / "dispo_declarada.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["datetime", "resource_name", "dispo", "gen_type"])
    for g in GENERATORS:
        for h in HOURS:
            w.writerow([h.isoformat(sep=" "), g["name"], g["dispo_kw"], "TERMICA"])

with open(BASE / "ofertas.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Date", "resource_name", "Value"])
    for g in GENERATORS:
        w.writerow([FECHA.isoformat(), g["name"], g["bid_cop_kwh"]])

with open(BASE / "demaCome.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["datetime", "dema"])
    for h in HOURS:
        w.writerow([h.isoformat(sep=" "), 350_000])

with open(BASE / "agc_asignado.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["datetime", "recurso", "agc"])
    w.writerow([HOURS[0].isoformat(sep=" "), "TERMO1", 0])

with open(BASE / "parametros_plantas.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["generador", "TMG"])
    for g in GENERATORS:
        w.writerow([g["name"], 1])

(BASE / "precio_bolsa").mkdir(exist_ok=True)
with open(BASE / "precio_bolsa" / "precio_bolsa_2024.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["datetime", "precio_bolsa"])
    for h in HOURS:
        w.writerow([h.isoformat(sep=" "), 200])

(BASE / "ramps.json").write_text("{}")
(BASE / "preideal_dispatch_map.json").write_text("{}")

flat_dir = BASE / str(FECHA)
flat_dir.mkdir(exist_ok=True)
ci_dir = BASE / "condicion_inicial" / str(FECHA)
ci_dir.mkdir(parents=True, exist_ok=True)

ofei_lines = []
for g in GENERATORS:
    ofei_lines.append(f'{g["name"]},C PAPC,{g["pap_cop"]}')
for g in GENERATORS:
    mo_vals = ",".join(str(g["mo"]) for _ in range(24))
    ofei_lines.append(f'{g["name"]}, MO,{mo_vals}')
(flat_dir / f"OFEI{MMDD}.txt").write_text("\n".join(ofei_lines) + "\n")

prid_row = ["TOTAL"] + ["350"] * 24
(flat_dir / f"PrId{MMDD}_NAL.txt").write_text(",".join(prid_row) + "\n", encoding="latin1")

with open(ci_dir / f"dCondIniP{MMDD}.txt", "w") as f:
    f.write("Recurso,Tipo,Gpini-1,Conf_Pini-1,T_CONF_Pini-1\n")
    for g in GENERATORS:
        f.write(f'{g["name"]},T,{g["gpini"]},{g["conf"]},{g["tconf"]}\n')

(ci_dir / f"dCondIniU{MMDD}.txt").write_text("Recurso,Tipo,Gini-1,Cini-1\n")

print("fixture written to", BASE)
```

- [ ] **Step 2: Run the generator and inspect the tree**

```bash
uv run python tests/fixtures/xm_smoke/generate_fixture.py
find tests/fixtures/xm_smoke -type f | sort
```

Expected tree (13 files):
```
tests/fixtures/xm_smoke/2024-04-18/OFEI0418.txt
tests/fixtures/xm_smoke/2024-04-18/PrId0418_NAL.txt
tests/fixtures/xm_smoke/agc_asignado.csv
tests/fixtures/xm_smoke/condicion_inicial/2024-04-18/dCondIniP0418.txt
tests/fixtures/xm_smoke/condicion_inicial/2024-04-18/dCondIniU0418.txt
tests/fixtures/xm_smoke/demaCome.csv
tests/fixtures/xm_smoke/dispo_declarada.csv
tests/fixtures/xm_smoke/generate_fixture.py
tests/fixtures/xm_smoke/ofertas.csv
tests/fixtures/xm_smoke/parametros_plantas.csv
tests/fixtures/xm_smoke/precio_bolsa/precio_bolsa_2024.csv
tests/fixtures/xm_smoke/preideal_dispatch_map.json
tests/fixtures/xm_smoke/ramps.json
```

- [ ] **Step 3: Scope the blanket `.gitignore` `*.csv` rule so the fixture actually gets committed**

`.gitignore:35` has a bare `*.csv` (meant for `data/` output CSVs, not test fixtures). Add a negation right after it:

```diff
 *.csv
+!tests/fixtures/**/*.csv
 # Installer logs
```

Verify every fixture file — CSVs included — will actually be staged:

```bash
git add --dry-run tests/fixtures/xm_smoke/
```

Expected: 13 `add '...'` lines, one per file in the tree from Step 2 (no silent drops). `git check-ignore -v tests/fixtures/xm_smoke/dispo_declarada.csv` before this step reproduces the bug (prints the `*.csv` rule as the match); after this step it must exit non-zero.

- [ ] **Step 4: Write the failing loader-level sanity test**

```python
# tests/test_xm_smoke_loaders.py
"""Layer-1 check: every raw fixture file parses through its real loader/parser
with no exceptions, before build_case's own logic gets involved. Isolates
format bugs (encoding, column names, delimiter) from case_builder logic bugs."""

from datetime import date
from pathlib import Path

from app.data import loaders
from app.data.download import ensure_data_for_date
from app.data.ofei import parse_ofei
from app.data.paths import resolve_input

DD = str(Path(__file__).parent / "fixtures" / "xm_smoke")
FECHA = date(2024, 4, 18)


def test_ensure_data_for_date_is_a_noop(monkeypatch):
    def _no_network(*a, **kw):
        raise AssertionError(f"unexpected network call: {a} {kw}")

    monkeypatch.setattr("app.data.download.requests.get", _no_network)
    ensure_data_for_date(FECHA, data_dir=DD)


def test_root_csvs_load():
    dispo = loaders.load_dispo(DD)
    assert len(dispo[dispo["datetime"].dt.date == FECHA]) == 48  # 2 generators x 24h

    ofertas = loaders.load_ofertas(DD)
    assert len(ofertas[ofertas["Date"].dt.date == FECHA]) == 2

    demanda = loaders.load_demanda(DD)
    assert len(demanda[demanda["datetime"].dt.date == FECHA]) == 24

    agc = loaders.load_agc(DD)
    assert "agc" in agc.columns

    params = loaders.load_parametros_plantas(DD)
    assert set(params["generador"]) == {"TERMO1", "TERMO2"}

    precio_bolsa = loaders.load_precio_bolsa(DD)
    assert len(precio_bolsa[precio_bolsa["datetime"].dt.date == FECHA]) == 24


def test_ofei_parses():
    ofei_path = resolve_input("OFEI", FECHA, DD)
    ofei = parse_ofei(ofei_path, FECHA)
    assert set(ofei.precio_arranque["resource"]) == {"TERMO1", "TERMO2"}
    assert all(ofei.precio_arranque["type"].str.contains("C"))
    assert set(ofei.minimo_operativo["resource"]) == {"TERMO1", "TERMO2"}
    assert ofei.cc == {}


def test_condicion_inicial_files_readable():
    p_path = resolve_input("dCondIniP", FECHA, DD)
    with open(p_path) as f:
        lines = f.readlines()
    assert len(lines) == 3  # header + 2 generators

    u_path = resolve_input("dCondIniU", FECHA, DD)
    with open(u_path) as f:
        assert f.readline().strip() == "Recurso,Tipo,Gini-1,Cini-1"


def test_prid_readable_latin1():
    import pandas as pd

    prid_path = resolve_input("PrId", FECHA, DD)
    df = pd.read_csv(prid_path, header=None, encoding="latin1")
    assert df.shape == (1, 25)  # 1 generator row, name + 24 hours
```

- [ ] **Step 5: Run the tests and verify they pass, from repo root and from a different cwd**

```bash
uv run pytest tests/test_xm_smoke_loaders.py -v
cd tests && uv run pytest ../tests/test_xm_smoke_loaders.py -v && cd ..
```

Expected: 5 passed both times (this reproduces the `"... files already downloaded. Skipping download"` no-network path verified by direct execution before this plan was written — the second run from `tests/` proves the `Path(__file__).parent`-anchored `DD` isn't cwd-dependent).

- [ ] **Step 6: Commit**

```bash
git add .gitignore tests/fixtures/xm_smoke tests/test_xm_smoke_loaders.py
git commit -m "test: add XM smoke-test fixture (2 thermal gens, preideal) + loader sanity checks"
```

---

### Task 2: `build_case()` direct-call test with exact value assertions

**Files:**
- Test: `tests/test_xm_smoke_build_case.py`

**Interfaces:**
- Consumes: `app.pipeline.case_builder.build_case(case, inputs, *, ders=None)` returning `(set_data, param_data, meta)` (`app/pipeline/case_builder.py:63`). `DispatchCase(dispatch_date, level, solver="cbc")` (`app/schemas/case.py`). `InputPack(dispatch_date, source, data_dir)` (`app/schemas/input_pack.py`).
- Produces: nothing consumed by later tasks — this is a leaf verification of exact `set_data`/`param_data` values, so a future change to `case_builder.py` that silently changes the fixture's numbers is caught here first, not at the model-solve layer where it's harder to diagnose.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_xm_smoke_build_case.py
"""Layer-2 check: build_case's own data-assembly logic (name mapping, unit
scaling, CC-empty path, initial-condition parsing) against the fixture,
independent of whether the model actually solves."""

from datetime import date
from pathlib import Path

import pytest

from app.pipeline.case_builder import build_case
from app.schemas.case import DispatchCase, DispatchLevel
from app.schemas.input_pack import InputPack, InputSource

DD = str(Path(__file__).parent / "fixtures" / "xm_smoke")
FECHA = date(2024, 4, 18)


@pytest.fixture
def built():
    case = DispatchCase(dispatch_date=FECHA, level=DispatchLevel.preideal, solver="cbc")
    inputs = InputPack(dispatch_date=FECHA, source=InputSource.historical, data_dir=DD)
    return build_case(case, inputs, ders=None)


def test_sets(built):
    set_data, _, _ = built
    assert sorted(set_data["G"]) == ["TERMO1", "TERMO2"]
    assert sorted(set_data["I"]) == ["TERMO1", "TERMO2"]
    assert list(set_data["gen_on"]) == ["TERMO1"]
    assert list(set_data["gen_off"]) == ["TERMO2"]
    assert set_data["combined_cycle"] == []
    assert len(list(set_data["T"])) == 24


def test_params_scaled_correctly(built):
    _, param_data, _ = built
    ts = date(2024, 4, 18)
    # dispo x1e-3 (kW -> MW): 300_000 kW -> 300 MW, 200_000 kW -> 200 MW
    pmax = param_data["Pmax"]
    for hour in range(24):
        import pandas as pd

        t = pd.Timestamp(ts) + pd.Timedelta(hours=hour)
        assert pmax[("TERMO1", t)] == 300.0
        assert pmax[("TERMO2", t)] == 200.0
    # PrId used raw (MW, no scaling): 350 every hour
    assert set(param_data["demand"].values()) == {350}
    # ofertas x1e3 (COP/kWh -> COP/MWh)
    beta = dict(param_data["beta"])
    assert beta == {"TERMO1": 150000.0, "TERMO2": 180000.0}


def test_cold_start_and_commitment_state(built):
    _, param_data, _ = built
    assert param_data["cold_start"] == {"TERMO1": 1500000.0, "TERMO2": 1500000.0}
    assert dict(param_data["TMG"]) == {"TERMO1": 1, "TERMO2": 1}
    assert dict(param_data["Ton"]) == {"TERMO1": 5}
    assert param_data["z_on_t0_minus_1"] == {"TERMO1": 1}
    assert param_data["ramp_up"] == {}  # ramps.json={}, RU/RD fall to model default=10000
```

- [ ] **Step 2: Red phase**

Task 1's fixture is a hard prerequisite for this task (`build_case` needs every file Task 1 produces), so there is no meaningful "write the test against nothing" red phase here — the red phase already happened while designing the fixture itself: an early fixture draft without a `precio_arranque` row containing `"C"` per generator raised `IndexError` at `case_builder.py:310` (`.values[0]` on an empty slice), and a draft with mismatched resource names across files raised `TypeError` from `thefuzz.process.extractOne(...)[0]` returning `None`. Both are the guard conditions listed in Global Constraints. Confirm you understand why before trusting the green below — if you're implementing this fresh, deliberately typo one resource name (e.g. `ofertas.csv`'s `TERMO1` -> `TERM01`) and re-run to see the `TypeError` yourself, then revert.

- [ ] **Step 3: Run the tests and verify they pass**

Run: `uv run pytest tests/test_xm_smoke_build_case.py -v`
Expected: 3 passed. These exact values were confirmed by direct execution before this plan was written — a failure here means either the fixture files or `case_builder.py` changed since.

- [ ] **Step 4: Commit**

```bash
git add tests/test_xm_smoke_build_case.py
git commit -m "test: assert build_case output values against the XM smoke fixture"
```

---

### Task 3: `run_case()` full `cbc` solve + no-network guard

**Files:**
- Test: `tests/test_xm_smoke_run.py`

**Interfaces:**
- Consumes: `app.pipeline.runner.run_case(case, *, evaluate=True, out="data/results", data_dir="data") -> RunResult` (`app/pipeline/runner.py:19`). `RunResult` fields: `ok: bool`, `error: str | None`, `dispatch_path`, `price_path` (`app/schemas/run_result.py`).
- Produces: nothing consumed by later tasks — this is the model-solve-layer verification. Task 4 re-derives the same result through the CLI subprocess instead of the Python API, so a bug isolated to `cli.py`'s argument wiring (not `run_case` itself) is still caught.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_xm_smoke_run.py
"""Layer-3 check: the fixture survives a real cbc solve via run_case, with
no XM actuals file and no network access — the two conditions a Docker
smoke test (Fase 2C) will also run under."""

from datetime import date
from pathlib import Path

import pandas as pd

from app.pipeline.runner import run_case
from app.schemas.case import DispatchCase, DispatchLevel

DD = str(Path(__file__).parent / "fixtures" / "xm_smoke")
FECHA = date(2024, 4, 18)


def test_run_case_solves_with_no_network_and_no_actuals(tmp_path, monkeypatch):
    def _no_network(*a, **kw):
        raise AssertionError(f"unexpected network call: {a} {kw}")

    monkeypatch.setattr("app.data.download.requests.get", _no_network)

    case = DispatchCase(dispatch_date=FECHA, level=DispatchLevel.preideal, solver="cbc")
    out = str(tmp_path / "results")
    result = run_case(case, evaluate=True, out=out, data_dir=DD)

    assert result.ok, result.error
    assert result.error is None
    # no preideal_price/ fixture file exists -> metrics silently skipped, not a failure
    assert result.metrics is None

    price = pd.read_csv(result.price_path)
    assert len(price) == 24
    assert (price["ideal_marginal_price"] > 0).all()
    # TERMO2 (180 COP/kWh bid, more expensive) is the marginal unit once TERMO1
    # (300 MW cap) is exhausted against 350 MW demand -> MPO = TERMO2's beta.
    assert (price["ideal_marginal_price"] == 180000.0).all()

    dispatch = pd.read_csv(result.dispatch_path)
    termo1 = dispatch[dispatch["generador"] == "TERMO1"]["dispatch"]
    termo2 = dispatch[dispatch["generador"] == "TERMO2"]["dispatch"]
    assert (termo1 == 300.0).all()  # at its cap every hour
    assert (termo2 == 50.0).all()  # covers the remaining 350 - 300
```

- [ ] **Step 2: Run the tests and verify they pass**

Run: `uv run pytest tests/test_xm_smoke_run.py -v`
Expected: 1 passed. These exact dispatch/price values (TERMO1=300, TERMO2=50, MPO=180000) were confirmed by a real `cbc` solve, direct execution, before this plan was written — this is not a guessed expectation.

- [ ] **Step 3: Commit**

```bash
git add tests/test_xm_smoke_run.py
git commit -m "test: solve the XM smoke fixture end-to-end via run_case with cbc"
```

---

### Task 4: CLI subprocess smoke test (literal spec exit criterion)

**Files:**
- Test: `tests/test_xm_smoke_cli.py`

**Interfaces:**
- Consumes: `python -m app run <fecha> -t preideal --data-dir <dir> --out <dir>` (Typer CLI, `app/cli.py:53-96`), invoked via `subprocess.run`. This is the literal command the Fase 2 spec's exit criterion for Fase B names, and the command Fase 2C's Docker smoke test will reuse verbatim (just prefixed with `docker run ...`).
- Produces: nothing consumed by later tasks. This is the last layer; Fase 2C's design should point at this test as "the same command, run in a container."

- [ ] **Step 1: Write the failing test**

```python
# tests/test_xm_smoke_cli.py
"""Layer-4 check: the literal CLI command from the Fase 2B spec exit
criterion, run as a subprocess exactly as a Docker CMD would invoke it."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DD = REPO_ROOT / "tests" / "fixtures" / "xm_smoke"


def test_cli_run_against_fixture(tmp_path):
    out = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable, "-m", "app", "run", "2024-04-18",
            "-t", "preideal",
            "--data-dir", str(DD),
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Done: 1 ok, 0 failed." in result.stdout
    assert (out / "marginal_price-2024-04-18-preideal.csv").exists()
    assert (out / "dispatch_by_gen-2024-04-18-preideal.csv").exists()
```

- [ ] **Step 2: Run the test and verify it passes**

Run: `uv run pytest tests/test_xm_smoke_cli.py -v`
Expected: 1 passed. Also verify manually, matching the spec's exit criterion wording exactly:

```bash
uv run python -m app run 2024-04-18 -t preideal --data-dir tests/fixtures/xm_smoke --out /tmp/fase2b-smoke-check
echo "exit code: $?"
```

Expected: `exit code: 0`, `Done: 1 ok, 0 failed.` printed (confirmed by direct execution before this plan was written).

- [ ] **Step 3: Run the full test suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: 78 (Fase 2A baseline) + 5 (Task 1) + 3 (Task 2) + 1 (Task 3) + 1 (Task 4) = 88 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_xm_smoke_cli.py
git commit -m "test: run the literal Fase 2B CLI smoke-test command as a subprocess"
```

---

## Exit criterion for Fase 2B

`uv run python -m app run 2024-04-18 -t preideal --data-dir tests/fixtures/xm_smoke --out <any-dir>` exits 0 on the host, with `marginal_price-*.csv` and `dispatch_by_gen-*.csv` produced and MPO non-empty — matching the spec's own exit criterion for section 2 verbatim. All four tasks' tests green (`uv run pytest -q` → 88 passed) is the mechanical proxy for that criterion in CI.

**Note for whoever writes the Fase 2C plan**: the Dockerfile's `COPY` step must include `tests/fixtures/xm_smoke/` (not just `app/`) for the container smoke test to reach this fixture — `data/` is git-ignored and volume-mounted (spec section 4), but this fixture lives under `tests/` and is git-tracked, so it needs an explicit `COPY tests/fixtures/xm_smoke tests/fixtures/xm_smoke` (or equivalent) in the image, separate from the `data/` volume mount.
