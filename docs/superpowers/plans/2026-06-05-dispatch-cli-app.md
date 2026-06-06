# Dispatch CLI App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the duplicated `run_dispatch.py` / `get_date_results.py` scripts with a single extracted pipeline in the `app` package, driven by a Typer `run` CLI command.

**Architecture:** Pure data modules (`app/data/*`) load and parse XM inputs; `app/pipeline/case_builder.py` turns a (date, config) into the model's set/param dicts; `app/pipeline/runner.py` orchestrates ensure-data → build → solve → extract → save → evaluate; `app/cli.py` exposes `run`. Old scripts become thin callers for regression safety.

**Tech Stack:** Python, pyomo (existing `UnitCommitmentModel`), pandas, thefuzz, Typer, pytest.

**Verification note:** `data/` is empty in this environment and XM is not reachable, so real-data golden fixtures cannot be captured here. Tasks use synthetic fixtures for the pure parsers, a toy pyomo model for results, and code-review equivalence (old scripts delegate to new functions) for `case_builder`. When real data is available, add the golden-fixture test in Task 6's optional step.

---

## File Structure

- `app/data/__init__.py` — re-exports
- `app/data/download.py` — `save_file` (moved from `utils/misc.py`), `ensure_data_for_date`
- `app/data/ofei.py` — `parse_ofei(path, dispatch_date)` → dataclass of PAP/MO/CC/prices
- `app/data/loaders.py` — root-CSV readers + unit conversions
- `app/data/actuals.py` — `load_actual_dispatch`, `load_actual_price`
- `app/pipeline/__init__.py`
- `app/pipeline/case_builder.py` — `build_case(dispatch_date, config, *, bess=None, ders=None)` → `(set_data, param_data, meta)`
- `app/pipeline/results.py` — `extract_results(model, config)`, `save_results(...)`
- `app/pipeline/runner.py` — `run_case(...)`, `run_many(...)`
- `app/dates.py` — `parse_dates_arg(token, available)` → list[date]
- `app/cli.py` — Typer app, `run` command
- `app/__main__.py` — `from app.cli import app; app()`
- `tests/` — pytest tests + `tests/fixtures/`

---

## Task 0: Scaffolding

**Files:**
- Create: `app/data/__init__.py`, `app/pipeline/__init__.py`, `app/__main__.py`, `tests/__init__.py`, `tests/conftest.py`
- Modify: `requirement.txt`

- [ ] **Step 1: Add typer + pytest to requirement.txt**

Append:
```
typer>=0.12
pytest>=8.0
```

- [ ] **Step 2: Create empty package files**

`app/data/__init__.py`, `app/pipeline/__init__.py`, `tests/__init__.py` — empty.

- [ ] **Step 3: conftest puts repo root on path**

`tests/conftest.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 4: Commit**
```bash
git add requirement.txt app/data/__init__.py app/pipeline/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: scaffold app.data/app.pipeline packages and pytest"
```

---

## Task 1: Date argument parsing

**Files:**
- Create: `app/dates.py`, `tests/test_dates.py`

- [ ] **Step 1: Write failing tests**

`tests/test_dates.py`:
```python
from datetime import date
import pytest
from app.dates import parse_dates_arg

AVAIL = [date(2024, 4, 1), date(2024, 4, 18), date(2024, 4, 30), date(2024, 5, 2)]

def test_single():
    assert parse_dates_arg("2024-04-18", AVAIL) == [date(2024, 4, 18)]

def test_range_inclusive():
    assert parse_dates_arg("2024-04-18:2024-04-30", AVAIL) == [date(2024, 4, 18), date(2024, 4, 30)]

def test_month():
    assert parse_dates_arg("2024-04", AVAIL) == [date(2024, 4, 1), date(2024, 4, 18), date(2024, 4, 30)]

def test_all_when_none():
    assert parse_dates_arg(None, AVAIL) == sorted(AVAIL)

def test_bad_token():
    with pytest.raises(ValueError):
        parse_dates_arg("not-a-date", AVAIL)
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/test_dates.py -v` → FAIL (module not found).

- [ ] **Step 3: Implement**

`app/dates.py`:
```python
from datetime import date, datetime


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_dates_arg(token: str | None, available: list[date]) -> list[date]:
    """Resolve a CLI date token to a sorted list of dates, filtered to `available`.

    Forms: 'YYYY-MM-DD', 'YYYY-MM-DD:YYYY-MM-DD' (inclusive), 'YYYY-MM', or None (all).
    """
    avail = sorted(available)
    if token is None:
        return avail
    if ":" in token:
        lo, hi = (_d(p) for p in token.split(":", 1))
        return [d for d in avail if lo <= d <= hi]
    parts = token.split("-")
    if len(parts) == 2:
        year, month = int(parts[0]), int(parts[1])
        return [d for d in avail if d.year == year and d.month == month]
    if len(parts) == 3:
        return [_d(token)]
    raise ValueError(f"Unrecognized date token: {token!r}")
```

- [ ] **Step 4: Run, expect pass**

Run: `pytest tests/test_dates.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add app/dates.py tests/test_dates.py
git commit -m "feat: date argument parsing for CLI"
```

---

## Task 2: Move download logic into app/data/download.py

**Files:**
- Create: `app/data/download.py`
- Modify: `app/utils/misc.py` (re-export for back-compat), `run_dispatch.py:21`, `get_date_results.py`

- [ ] **Step 1: Move `save_file` + `PARAMS` + URL constants verbatim**

Cut `PARAMS`, `XM_DOWNLOAD_URL`, and `save_file` from `app/utils/misc.py` into `app/data/download.py` unchanged.

- [ ] **Step 2: Add `ensure_data_for_date`**

Append to `app/data/download.py` (this is the dedup of `run_dispatch.py:33-38`):
```python
from datetime import date
from pathlib import Path


def ensure_data_for_date(dispatch_date: date, data_dir: str = "data") -> Path:
    """Download per-day XM files into data/{date}/ if the folder is absent."""
    folder = Path(data_dir) / str(dispatch_date)
    if folder.is_dir():
        return folder
    for file_type in PARAMS:
        save_file(file_type=file_type, file_date=dispatch_date)
    return folder
```

- [ ] **Step 3: Back-compat shim in misc.py**

`app/utils/misc.py` becomes:
```python
from app.data.download import PARAMS, XM_DOWNLOAD_URL, save_file  # noqa: F401
```

- [ ] **Step 4: Smoke test import**

Run: `python -c "from app.data.download import save_file, ensure_data_for_date, PARAMS; from app.utils.misc import save_file"` → no error.

- [ ] **Step 5: Commit**
```bash
git add app/data/download.py app/utils/misc.py
git commit -m "refactor: move XM download into app.data.download + ensure_data_for_date"
```

---

## Task 3: OFEI parser → app/data/ofei.py

**Files:**
- Create: `app/data/ofei.py`, `tests/fixtures/OFEI_sample.txt`, `tests/test_ofei.py`

The source logic to extract verbatim (behavior-preserving): `run_dispatch.py:71-139` (the OFEI `open()` loop building `output/MO/CC/cc_price/cc_dispo/prices`, then `precio_arranque` and `minimo_operativo` dataframes).

- [ ] **Step 1: Define the result container + function signature**

`app/data/ofei.py`:
```python
from dataclasses import dataclass
from datetime import date
import re
import pandas as pd

PRICE_PATTERN = r"P(\d+)"
DISPO_PATTERN = r"DISCONF(\d+)"


@dataclass
class OfeiData:
    precio_arranque: pd.DataFrame      # columns: resource, type, price
    minimo_operativo: pd.DataFrame     # columns: resource, type, hour, minimo_operativo, datetime
    cc: dict[str, list[str]]           # plant -> [plant_conf, ...]
    cc_price: dict[str, float]         # plant_conf -> price
    cc_dispo: dict[str, list[int]]     # plant_conf -> 24 hourly availabilities
    prices: dict[str, float]           # resource -> bid price (already *1e-3)


def parse_ofei(path: str, dispatch_date: date) -> OfeiData:
    ...
```

- [ ] **Step 2: Write the failing test with a synthetic fixture**

Create `tests/fixtures/OFEI_sample.txt` containing a handful of representative lines (one PAP line, one MO line with 24 hourly values, one `CC ... P1` price line, one `CC ... DISCONF1` line with 24 values, one ` P` price line). Then `tests/test_ofei.py`:
```python
from datetime import date
from pathlib import Path
from app.data.ofei import parse_ofei

FIX = Path(__file__).parent / "fixtures" / "OFEI_sample.txt"

def test_parses_each_section():
    d = parse_ofei(str(FIX), date(2024, 4, 18))
    assert not d.precio_arranque.empty
    assert list(d.minimo_operativo.columns) == ["resource", "type", "hour", "minimo_operativo", "datetime"]
    assert all(len(v) == 24 for v in d.cc_dispo.values())
    assert all(p < 1000 for p in d.prices.values())  # scaled by 1e-3
```

(Fill the fixture lines so these assertions hold; copy the exact line shapes from a real OFEI file's grammar already encoded in `run_dispatch.py:81-112`.)

- [ ] **Step 3: Run, expect fail**

Run: `pytest tests/test_ofei.py -v` → FAIL.

- [ ] **Step 4: Implement `parse_ofei` by moving `run_dispatch.py:71-139` into the function body**, returning an `OfeiData`. Replace the hardcoded path with the `path` arg and `DISPATCH_DATE` with `dispatch_date`. Keep the `re` patterns, the `1e-3` price scaling, and the `minimo_operativo` reshape identical.

- [ ] **Step 5: Run, expect pass**

Run: `pytest tests/test_ofei.py -v` → PASS.

- [ ] **Step 6: Commit**
```bash
git add app/data/ofei.py tests/fixtures/OFEI_sample.txt tests/test_ofei.py
git commit -m "feat: extract OFEI parser into app.data.ofei with tests"
```

---

## Task 4: CSV loaders → app/data/loaders.py

**Files:**
- Create: `app/data/loaders.py`, `tests/test_loaders.py`

- [ ] **Step 1: Implement readers (centralize the unit conversions)**

`app/data/loaders.py` — one function per CSV, each returning a DataFrame. Apply the conversions currently scattered (`precio_bolsa * 1e3` at `run_dispatch.py:69`; `demanda * 1e-3` applied later in case-build) here, in exactly one place:
```python
import pandas as pd


def load_dispo(data_dir="data"):
    return pd.read_csv(f"{data_dir}/dispo_declarada.csv", parse_dates=["datetime"], engine="pyarrow")

def load_ofertas(data_dir="data"):
    return pd.read_csv(f"{data_dir}/ofertas.csv", parse_dates=["Date"], engine="pyarrow")

def load_demanda(data_dir="data"):
    return pd.read_csv(f"{data_dir}/demaCome.csv", parse_dates=["datetime"], engine="pyarrow")

def load_agc(data_dir="data"):
    return pd.read_csv(f"{data_dir}/agc_asignado.csv", parse_dates=["datetime"], engine="pyarrow")

def load_parametros_plantas(data_dir="data"):
    return pd.read_csv(f"{data_dir}/parametros_plantas.csv")

def load_precio_bolsa(data_dir="data"):
    df = pd.read_csv(f"{data_dir}/precio_bolsa/precio_bolsa_2024.csv", parse_dates=["datetime"], engine="pyarrow")
    df["precio_bolsa"] = df["precio_bolsa"] * 1e3
    return df

def load_dispo_come(data_dir="data"):
    return pd.read_csv(f"{data_dir}/DispoCome_resource.csv", parse_dates=["datetime"], engine="pyarrow")
```

- [ ] **Step 2: Test with a tiny temp CSV**

`tests/test_loaders.py` — write a 2-row `precio_bolsa` CSV into `tmp_path`, assert `load_precio_bolsa(tmp_path)` applies the `*1e3` scaling. (One representative test; the others are trivial passthroughs.)
```python
import pandas as pd
from pathlib import Path
from app.data.loaders import load_precio_bolsa

def test_precio_bolsa_scaled(tmp_path):
    (tmp_path / "precio_bolsa").mkdir()
    pd.DataFrame({"datetime": ["2024-04-18 00:00"], "precio_bolsa": [0.1]}).to_csv(
        tmp_path / "precio_bolsa" / "precio_bolsa_2024.csv", index=False)
    out = load_precio_bolsa(str(tmp_path))
    assert abs(out["precio_bolsa"].iloc[0] - 100.0) < 1e-9
```

- [ ] **Step 3: Run → PASS.** `pytest tests/test_loaders.py -v`

- [ ] **Step 4: Commit**
```bash
git add app/data/loaders.py tests/test_loaders.py
git commit -m "feat: centralize CSV loaders and unit conversions in app.data.loaders"
```

---

## Task 5: Actuals loader → app/data/actuals.py

**Files:**
- Create: `app/data/actuals.py`, `tests/test_actuals.py`

Source: `get_date_results.py:431-449, 533-534` (reading `data/preideal_dispatch/{date}.txt` and `data/preideal_price/{date}.txt`).

- [ ] **Step 1: Implement**

`app/data/actuals.py`:
```python
from datetime import date
import pandas as pd


def load_actual_price(dispatch_date: date, data_dir="data"):
    """XM marginal price (MPO) row for the date; returns a 24-length float array."""
    df = pd.read_csv(f"{data_dir}/preideal_price/{dispatch_date}.txt", header=None)
    return df.iloc[0, 1:].astype(float).values


def load_actual_dispatch(dispatch_date: date, data_dir="data"):
    """XM predispatch generation matrix for the date (raw, latin1)."""
    return pd.read_csv(
        f"{data_dir}/preideal_dispatch/{dispatch_date}.txt", header=None, encoding="latin1"
    )
```

- [ ] **Step 2: Test with synthetic txt**

`tests/test_actuals.py` — write a 1-row, 25-col price txt into `tmp_path/preideal_price/2024-04-18.txt`, assert `load_actual_price` returns length-24 floats.

- [ ] **Step 3: Run → PASS.**

- [ ] **Step 4: Commit**
```bash
git add app/data/actuals.py tests/test_actuals.py
git commit -m "feat: XM actuals loader in app.data.actuals"
```

---

## Task 6: Case builder → app/pipeline/case_builder.py

**Files:**
- Create: `app/pipeline/case_builder.py`
- Source: `run_dispatch.py:43-642` (everything between data load and `model = UnitCommitmentModel(...)`)

This is the core extraction. The function:
```python
from datetime import date
from app.model import DispatchConfig

def build_case(
    dispatch_date: date,
    config: DispatchConfig,
    *,
    bess: dict | None = None,
    ders: int | None = None,
    data_dir: str = "data",
) -> tuple[dict, dict, dict]:
    """Return (set_data, param_data, meta) for the model.

    meta carries non-model artifacts the caller needs:
    {'timestamps', 'precio_bolsa', 'pmax_new_resources', 'expansion_sources', 'cc_map'}.
    """
```

- [ ] **Step 1: Assemble the body** by moving `run_dispatch.py:43-642` into `build_case`, with these mechanical substitutions:
  - `DISPATCH_DATE` → `dispatch_date`; `config` stays.
  - Replace the inline CSV reads with calls to `app.data.loaders`.
  - Replace the OFEI `open()` block with `from app.data.ofei import parse_ofei` and unpack the `OfeiData` fields (`precio_arranque`, `minimo_operativo`, `cc`, `cc_price`, `cc_dispo`, `prices`).
  - Replace the download block with `from app.data.download import ensure_data_for_date; ensure_data_for_date(dispatch_date, data_dir)`.
  - Keep ALL fuzzy-matching, CC-resource synthesis, initial-condition, BESS and DERs logic byte-for-byte; only the inputs/outputs change.
  - End by returning `(set_data, param_data, meta)` instead of constructing/solving the model.

- [ ] **Step 2: Import-only smoke test**

Run: `python -c "from app.pipeline.case_builder import build_case"` → no error.

- [ ] **Step 3 (optional, requires real data): golden fixture**

When `data/` is populated, before this refactor capture `set_data`/`param_data` from the current `run_dispatch` for `date(2024,4,18)` preideal and `ideal`, pickle them under `tests/fixtures/`, then assert `build_case(...)` reproduces them key-for-key. Document the capture command in the test docstring.

- [ ] **Step 4: Commit**
```bash
git add app/pipeline/case_builder.py
git commit -m "refactor: extract build_case (single source for set/param dicts)"
```

---

## Task 7: Results extraction → app/pipeline/results.py

**Files:**
- Create: `app/pipeline/results.py`, `tests/test_results.py`
- Source: `run_dispatch.py:693-704` (MPO from duals, with `objective.sense.value` sign) and `get_date_results.py:543-558` (dispatch from `pout`).

- [ ] **Step 1: Implement**

`app/pipeline/results.py`:
```python
from pathlib import Path
import pyomo.environ as pyo
import pandas as pd


def extract_mpo(model) -> dict:
    sense = model._model.objective.sense.value
    return {
        ke.index(): sense * pyo.value(dual_)
        for ke, dual_ in model._model.dual.items()
        if "power_balance" in ke.name
    }


def extract_dispatch(model) -> pd.DataFrame:
    data = {(g, t): pyo.value(v) for (g, t), v in model._model.pout.items()}
    return pd.DataFrame(
        data=data.values(), index=data.keys(), columns=["dispatch"]
    ).reset_index(drop=False, names=["generador", "datetime"])


def save_results(model, dispatch_date, config, out="data/results") -> dict:
    Path(out).mkdir(parents=True, exist_ok=True)
    t = config.dispatch_type.value
    disp = extract_dispatch(model)
    disp_path = f"{out}/dispatch_by_gen-{dispatch_date}-{t}.csv"
    disp.to_csv(disp_path, sep=",", index=False)
    mpo = extract_mpo(model)
    price_path = f"{out}/marginal_price-{dispatch_date}-{t}.csv"
    pd.DataFrame(data=mpo.values(), index=mpo.keys(), columns=["ideal_marginal_price"]) \
        .reset_index(drop=False, names=["datetime"]).to_csv(price_path, sep=",", index=False)
    return {"dispatch": disp_path, "price": price_path, "mpo": mpo}
```

- [ ] **Step 2: Test with the toy 2-gen model** (reuse the model from the pricing-fix verification): build a 2-gen, 1-period preideal model, solve, assert `extract_mpo` returns the marginal cost and `extract_dispatch` has the right rows. (Requires cbc.)

- [ ] **Step 3: Run → PASS.** `pytest tests/test_results.py -v`

- [ ] **Step 4: Commit**
```bash
git add app/pipeline/results.py tests/test_results.py
git commit -m "feat: results extraction (MPO + dispatch) with sign handling"
```

---

## Task 8: Runner → app/pipeline/runner.py

**Files:**
- Create: `app/pipeline/runner.py`, `tests/test_runner.py`

- [ ] **Step 1: Implement orchestration + failure isolation + evaluation**

`app/pipeline/runner.py`:
```python
from dataclasses import dataclass, field
from datetime import date
import traceback
import pandas as pd

from app.model import UnitCommitmentModel, DispatchConfig
from app.pipeline.case_builder import build_case
from app.pipeline.results import save_results
from app.data.actuals import load_actual_price
from app.utils.metrics import price_metrics


@dataclass
class CaseResult:
    dispatch_date: date
    dispatch_type: str
    ok: bool
    paths: dict = field(default_factory=dict)
    metrics: dict | None = None
    error: str | None = None


def run_case(dispatch_date, config, *, solver="cbc", compute_prices=True,
             evaluate=True, bess=None, ders=None, out="data/results",
             data_dir="data") -> CaseResult:
    t = config.dispatch_type.value
    try:
        set_data, param_data, meta = build_case(
            dispatch_date, config, bess=bess, ders=ders, data_dir=data_dir)
        model = UnitCommitmentModel(config=config)
        model.create_model(set_data=set_data, param_data=param_data)
        model.solve(solver=solver, compute_prices=compute_prices)
        paths = save_results(model, dispatch_date, config, out=out)
        metrics = None
        if evaluate:
            try:
                xm = load_actual_price(dispatch_date, data_dir=data_dir)
                model_mpo = list(paths["mpo"].values())
                metrics = price_metrics(xm, model_mpo[:len(xm)])
                pd.DataFrame([metrics]).to_csv(
                    f"{out}/metrics-{dispatch_date}-{t}.csv", index=False)
            except FileNotFoundError:
                print(f"  ! no XM actuals for {dispatch_date}; skipping metrics")
        return CaseResult(dispatch_date, t, True, paths, metrics)
    except Exception as e:
        traceback.print_exc()
        return CaseResult(dispatch_date, t, False, error=f"{type(e).__name__}: {e}")


def run_many(dates, configs, *, out="data/results", **kw) -> list[CaseResult]:
    results = []
    for d in dates:
        for cfg in configs:
            print(f"==> {d} [{cfg.dispatch_type.value}]")
            results.append(run_case(d, cfg, out=out, **kw))
    ok = [r for r in results if r.ok]
    rows = [{"date": r.dispatch_date, "type": r.dispatch_type, **(r.metrics or {})}
            for r in ok if r.metrics]
    if rows:
        pd.DataFrame(rows).to_csv(f"{out}/metrics-summary.csv", index=False)
    return results
```

- [ ] **Step 2: Test failure isolation with monkeypatch**

`tests/test_runner.py` — monkeypatch `app.pipeline.runner.build_case` to raise for one date and succeed (returning a trivial solvable toy case) for another; assert `run_many` returns one `ok=False` and one `ok=True` and does not raise.

- [ ] **Step 3: Run → PASS.**

- [ ] **Step 4: Commit**
```bash
git add app/pipeline/runner.py tests/test_runner.py
git commit -m "feat: runner with per-case failure isolation and evaluation"
```

---

## Task 9: CLI → app/cli.py + app/__main__.py

**Files:**
- Create: `app/cli.py`, `app/__main__.py`, `tests/test_cli.py`

- [ ] **Step 1: Implement Typer command**

`app/cli.py`:
```python
from datetime import date
from pathlib import Path
import os
import typer

from app.model import DispatchConfig, DispatchOptions
from app.dates import parse_dates_arg
from app.pipeline.runner import run_many

app = typer.Typer(add_completion=False, help="Colombian dispatch model runner.")


def _available_dates(data_dir: str) -> list[date]:
    root = Path(data_dir) / "condicion_inicial"
    out = []
    if root.exists():
        for f in root.glob("*"):
            if f.is_dir():
                y, m, d = (int(x) for x in f.stem.split("-"))
                out.append(date(y, m, d))
    return out


@app.command()
def run(
    dates: str = typer.Argument(None, help="YYYY-MM-DD | range a:b | YYYY-MM | omit=all"),
    type: list[str] = typer.Option(["preideal"], "--type", "-t", help="dispatch type, repeatable, or 'all'"),
    solver: str = typer.Option("cbc"),
    eval: bool = typer.Option(True, "--eval/--no-eval"),
    prices: bool = typer.Option(True, "--prices/--no-prices"),
    skip_dates: str = typer.Option("", help="comma-separated YYYY-MM-DD to skip"),
    out: str = typer.Option("data/results"),
    data_dir: str = typer.Option("data"),
):
    avail = _available_dates(data_dir)
    selected = parse_dates_arg(dates, avail) if avail else (
        parse_dates_arg(dates, []) if dates else [])
    skip = {parse_dates_arg(s, [s and date(*map(int, s.split('-')))])[0]
            for s in skip_dates.split(",") if s} if skip_dates else set()
    selected = [d for d in selected if d not in skip]
    types = DispatchOptions._member_names_ if "all" in type else type
    configs = [DispatchConfig(dispatch_type=t) for t in types]

    if not selected:
        typer.echo("No dates selected."); raise typer.Exit(code=1)

    results = run_many(selected, configs, solver=solver, compute_prices=prices,
                       evaluate=eval, out=out, data_dir=data_dir)
    failed = [r for r in results if not r.ok]
    typer.echo(f"\nDone: {len(results) - len(failed)} ok, {len(failed)} failed.")
    for r in failed:
        typer.echo(f"  FAIL {r.dispatch_date} [{r.dispatch_type}]: {r.error}")
    raise typer.Exit(code=1 if failed else 0)
```

Note: simplify the `skip_dates` parsing to a clean helper during implementation — parse each comma token with `datetime.strptime(tok, "%Y-%m-%d").date()`.

`app/__main__.py`:
```python
from app.cli import app

if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Test via Typer's CliRunner**

`tests/test_cli.py` — use `typer.testing.CliRunner`; monkeypatch `app.cli.run_many` to return a known list and `_available_dates` to return `[date(2024,4,18)]`; invoke `["2024-04-18", "-t", "preideal"]`; assert exit code 0 and "1 ok" in output. Add a case where `run_many` returns a failed result → exit code 1.

- [ ] **Step 3: Run → PASS.** `pytest tests/test_cli.py -v`

- [ ] **Step 4: Commit**
```bash
git add app/cli.py app/__main__.py tests/test_cli.py
git commit -m "feat: Typer CLI `run` command + python -m app entrypoint"
```

---

## Task 10: Migrate old scripts to delegate

**Files:**
- Modify: `run_dispatch.py`, `get_date_results.py`

- [ ] **Step 1: Reduce `run_dispatch()` body** to: `set_data, param_data, meta = build_case(...)`, build/solve model, `save_results(...)`, return `(mpo_df, model, meta['pmax_new_resources'], meta['expansion_sources'])` so the notebooks' call sites still work. Keep the plotting branch guarded by `show_figs` using `meta`.

- [ ] **Step 2: Replace `get_date_results.py`'s duplicated body** with a loop calling `run_case(...)` over its date list. Move `SKIP_DATES` to a `--skip-dates` default note in the CLI; keep the script as a thin batch caller or delete it if the CLI fully covers it.

- [ ] **Step 3: Import smoke test**

Run: `python -c "import run_dispatch, get_date_results"` (expect: imports without executing the batch loop — guard `get_date_results` top-level loop under `if __name__ == '__main__':`).

- [ ] **Step 4: Commit**
```bash
git add run_dispatch.py get_date_results.py
git commit -m "refactor: old scripts delegate to app.pipeline (dedup ~900 lines)"
```

---

## Task 11: Move metrics test into tests/, wire summary

**Files:**
- Create: `tests/test_metrics.py` (move from /tmp version)

- [ ] **Step 1: Add the metrics test** (already written and passing in /tmp/test_metrics.py) under `tests/`, with imports from `app.utils.metrics`.

- [ ] **Step 2: Run full suite → PASS.** `pytest -q`

- [ ] **Step 3: Commit**
```bash
git add tests/test_metrics.py
git commit -m "test: metrics suite under tests/"
```

---

## Self-Review

- **Spec coverage:** data modules (T2-T5), case_builder full extract (T6), results (T7), runner+eval+failure-isolation (T8), CLI run with date/range/month/all + flags (T1,T9), old-script migration (T10), metrics (T11), typer dep (T0). All spec sections mapped.
- **Placeholders:** none; bulk extraction tasks cite exact source line ranges + mechanical substitution rules.
- **Type consistency:** `build_case → (set_data, param_data, meta)` consumed by `run_case`; `OfeiData` fields consumed by `build_case`; `CaseResult` fields consumed by `cli.run`. Consistent.
- **Known gap:** real-data golden fixture deferred to T6 Step 3 (no data in this environment).
```
