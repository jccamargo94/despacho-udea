# Fase 1: CLI completo + storage layer + escenarios BESS — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the roadmap's Fase 1 — `fetch`/`evaluate`/`compare` CLI commands, a YAML BESS scenario library wired into `run`, persisted BESS results, a consolidated per-run summary, and a local/GCS storage abstraction underneath all file I/O.

**Architecture:** A thin `Storage` protocol (`exists`/`open`/`list_dir`) with a `LocalStorage` implementation replaces raw `pathlib`/`open()` calls across `app/data/*` and `app/pipeline/*`, selected by a `get_storage(root)` factory keyed on the existing `--data-dir`/`--out` strings (no new CLI flags; `gs://...` raises `NotImplementedError` until a `GcsStorage` lands later). On top of that: a `scenarios/bess/*.yaml` library loaded into `BessScenario` via pydantic, three new Typer subcommands, and extensions to `RunResult`/`save_results`/`run_many` for BESS persistence and summary rows.

**Tech Stack:** Python 3, Typer CLI, pydantic v1, pandas, PyYAML (already installed, no new dependency), pytest, Pyomo/CBC for the two BESS-solving tests.

## Global Constraints

- pydantic v1 API throughout (`BessScenario.parse_obj(...)`, not `model_validate`).
- No new dependencies. PyYAML 6.0.1 is already present in the project venv; `google-cloud-storage`/`gcsfs` are NOT installed and must not be added — `GcsStorage` is out of scope for this plan.
- Every public function signature that exists today (`data_dir: str`, `out: str`, `dates: str`) stays unchanged in the storage-migration tasks (1–5) — these are behavior-preserving refactors; existing test assertions must not need edits.
- Run tests with the project venv: `~/.local/share/virtualenvs/dam-worker-optimizer-W9GjOqr4/bin/python -m pytest tests/ -v` (has pyomo 6.7.3 + cbc + pydantic 1.10.8 + pytest/typer/thefuzz/pyyaml).
- `app/pipeline/case_builder.py`'s two `open(resolve_input(...), "r")` blocks (dCondIniP/dCondIniU) are **not** migrated to `Storage` in this plan — `resolve_input` keeps returning a plain local path string (see Task 2), so wrapping those two `open()` calls in `storage.open()` would only add indirection with no behavior change until a GCS-aware `resolve_input` exists. Left as plain `open()`, noted here so it isn't mistaken for an oversight.
- BESS `revenue`/`cost` use MPO-based settlement (`energy_MWh * price_COP_per_kWh * 1000`), not bid-based — see Task 8. `grid_asset` units frequently have no bids at all.
- Known, undisturbed limitation: `same_soc_start_and_end` (`app/model/constraints/bess/soc.py`) only binds in `grid_asset` mode (via the legacy `_dispatch_type` substring tag). End-of-day SOC differs by mode, so `bess_net_revenue` (Task 8/9) is not directly comparable between `arbitrage` and `grid_asset` scenarios. Not fixed in this plan.

---

### Task 1: Storage core (`Storage` protocol, `LocalStorage`, `get_storage`)

**Files:**
- Create: `app/storage/__init__.py`
- Create: `app/storage/base.py`
- Create: `app/storage/local.py`
- Create: `app/storage/factory.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Produces: `Storage` (Protocol) with `exists(path: str) -> bool`, `open(path: str, mode: str = "r") -> ContextManager[IO]`, `list_dir(path: str) -> list[str]`. `LocalStorage(root: str)` implementing it. `get_storage(root: str) -> Storage` factory (raises `NotImplementedError` for `gs://` prefixes). All importable as `from app.storage import Storage, LocalStorage, get_storage`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage.py
from datetime import date

import pytest

from app.storage import LocalStorage, get_storage


def test_exists_false_for_missing_file(tmp_path):
    storage = LocalStorage(str(tmp_path))
    assert storage.exists("missing.txt") is False


def test_open_write_then_read_roundtrip(tmp_path):
    storage = LocalStorage(str(tmp_path))
    with storage.open("a.txt", "w") as f:
        f.write("hello")
    assert storage.exists("a.txt") is True
    with storage.open("a.txt", "r") as f:
        assert f.read() == "hello"


def test_open_write_creates_parent_dirs(tmp_path):
    storage = LocalStorage(str(tmp_path))
    with storage.open("nested/dir/b.txt", "w") as f:
        f.write("x")
    assert (tmp_path / "nested" / "dir" / "b.txt").read_text() == "x"


def test_list_dir_returns_entry_names(tmp_path):
    (tmp_path / "condicion_inicial").mkdir()
    (tmp_path / "condicion_inicial" / "2024-04-18").mkdir()
    (tmp_path / "condicion_inicial" / "2024-04-19").mkdir()
    storage = LocalStorage(str(tmp_path))
    assert sorted(storage.list_dir("condicion_inicial")) == ["2024-04-18", "2024-04-19"]


def test_list_dir_missing_returns_empty(tmp_path):
    storage = LocalStorage(str(tmp_path))
    assert storage.list_dir("does-not-exist") == []


def test_get_storage_returns_local_for_plain_path(tmp_path):
    storage = get_storage(str(tmp_path))
    assert isinstance(storage, LocalStorage)


def test_get_storage_raises_not_implemented_for_gcs():
    with pytest.raises(NotImplementedError):
        get_storage("gs://some-bucket/prefix")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_storage.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'app.storage'`

- [ ] **Step 3: Write the implementation**

```python
# app/storage/base.py
from __future__ import annotations

from typing import IO, ContextManager, Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    def exists(self, path: str) -> bool: ...
    def open(self, path: str, mode: str = "r") -> ContextManager[IO]: ...
    def list_dir(self, path: str) -> list[str]: ...
```

```python
# app/storage/local.py
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator


class LocalStorage:
    """Storage backed by the local filesystem, rooted at `root`."""

    def __init__(self, root: str):
        self.root = Path(root)

    def _resolve(self, path: str) -> Path:
        return self.root / path

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    @contextmanager
    def open(self, path: str, mode: str = "r") -> Iterator[IO]:
        p = self._resolve(path)
        if "w" in mode or "a" in mode:
            p.parent.mkdir(parents=True, exist_ok=True)
        f = open(p, mode)
        try:
            yield f
        finally:
            f.close()

    def list_dir(self, path: str) -> list[str]:
        p = self._resolve(path)
        if not p.is_dir():
            return []
        return [entry.name for entry in p.iterdir()]
```

```python
# app/storage/factory.py
from app.storage.base import Storage
from app.storage.local import LocalStorage


def get_storage(root: str) -> Storage:
    if root.startswith("gs://"):
        raise NotImplementedError("GCS backend not implemented yet")
    return LocalStorage(root)
```

```python
# app/storage/__init__.py
from app.storage.base import Storage
from app.storage.local import LocalStorage
from app.storage.factory import get_storage

__all__ = ["Storage", "LocalStorage", "get_storage"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_storage.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add app/storage tests/test_storage.py
git commit -m "feat: add Storage protocol with local backend

GCS backend deferred; get_storage raises NotImplementedError for gs:// roots."
```

---

### Task 2: Migrate `loaders.py` + `paths.py` + `actuals.py` to `Storage`

**Files:**
- Modify: `app/data/loaders.py`
- Modify: `app/data/paths.py`
- Modify: `app/data/actuals.py`
- Test: `tests/test_loaders.py` (existing, unmodified), `tests/test_paths.py` (existing, unmodified), `tests/test_actuals.py` (existing, unmodified)

**Interfaces:**
- Consumes: `get_storage` from Task 1.
- Produces: no signature changes — `load_dispo`, `load_ofertas`, `load_demanda`, `load_agc`, `load_parametros_plantas`, `load_precio_bolsa`, `load_dispo_come`, `resolve_input`, `load_actual_price`, `load_actual_dispatch` all keep their existing signatures and return values.

This task is a behavior-preserving refactor: existing tests are the acceptance check, no new assertions needed.

- [ ] **Step 1: Confirm baseline — existing tests pass on current code**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_loaders.py tests/test_paths.py tests/test_actuals.py -v`
Expected: all pass (baseline before refactor)

- [ ] **Step 2: Migrate `loaders.py`**

```python
# app/data/loaders.py
"""Readers for the root-level XM CSVs.

Unit conversions that were previously scattered across the scripts are applied
here, in exactly one place (e.g. precio_bolsa is scaled to COP/MWh).
"""

import pandas as pd

from app.storage import get_storage


def load_dispo(data_dir: str = "data") -> pd.DataFrame:
    storage = get_storage(data_dir)
    with storage.open("dispo_declarada.csv") as f:
        return pd.read_csv(f, parse_dates=["datetime"])


def load_ofertas(data_dir: str = "data") -> pd.DataFrame:
    storage = get_storage(data_dir)
    with storage.open("ofertas.csv") as f:
        return pd.read_csv(f, parse_dates=["Date"])


def load_demanda(data_dir: str = "data") -> pd.DataFrame:
    storage = get_storage(data_dir)
    with storage.open("demaCome.csv") as f:
        return pd.read_csv(f, parse_dates=["datetime"])


def load_agc(data_dir: str = "data") -> pd.DataFrame:
    storage = get_storage(data_dir)
    with storage.open("agc_asignado.csv") as f:
        return pd.read_csv(f, parse_dates=["datetime"])


def load_parametros_plantas(data_dir: str = "data") -> pd.DataFrame:
    storage = get_storage(data_dir)
    with storage.open("parametros_plantas.csv") as f:
        return pd.read_csv(f)


def load_precio_bolsa(data_dir: str = "data") -> pd.DataFrame:
    storage = get_storage(data_dir)
    with storage.open("precio_bolsa/precio_bolsa_2024.csv") as f:
        df = pd.read_csv(f, parse_dates=["datetime"])
    df["precio_bolsa"] = df["precio_bolsa"] * 1e3
    return df


def load_dispo_come(data_dir: str = "data") -> pd.DataFrame:
    storage = get_storage(data_dir)
    with storage.open("DispoCome_resource.csv") as f:
        return pd.read_csv(f, parse_dates=["datetime"])
```

- [ ] **Step 3: Migrate `paths.py`** (existence check goes through `storage.exists`; the returned path string is unchanged — still `str(Path(data_dir) / sub / filename)`)

```python
# app/data/paths.py
"""Input-file path resolution across the two data-layout conventions.

Historical/offline layout uses organized directories (oferta_inicial/,
condicion_inicial/{date}/, ...). The live single-date layout puts the freshly
downloaded files under data/{date}/. The resolver tries the organized location
first, then falls back to the per-date download folder, so both work.
"""

from datetime import date
from pathlib import Path

from app.storage import get_storage

CANDIDATE_SUBDIRS = {
    "OFEI": ["oferta_inicial", "{date}"],
    "dCondIniP": ["condicion_inicial/{date}", "{date}"],
    "dCondIniU": ["condicion_inicial/{date}", "{date}"],
    "PrId": ["predespacho_ideal", "{date}"],
    "iMAR": ["predespacho_ideal", "{date}"],
}


def _filename(kind: str, dispatch_date: date) -> str:
    mmdd = f"{dispatch_date.month:0>2}{dispatch_date.day:0>2}"
    complement = "_NAL" if kind in {"PrId", "iMAR"} else ""
    return f"{kind}{mmdd}{complement}.txt"


def resolve_input(kind: str, dispatch_date: date, data_dir: str = "data") -> str:
    """Return the first existing path for `kind` on `dispatch_date`.

    Raises FileNotFoundError listing every candidate that was tried.
    """
    storage = get_storage(data_dir)
    filename = _filename(kind, dispatch_date)
    tried = []
    for sub in CANDIDATE_SUBDIRS[kind]:
        sub = sub.format(date=dispatch_date)
        rel = f"{sub}/{filename}"
        p = Path(data_dir) / sub / filename
        tried.append(str(p))
        if storage.exists(rel):
            return str(p)
    raise FileNotFoundError(
        f"Could not find {kind} file for {dispatch_date}. Tried: {tried}"
    )
```

- [ ] **Step 4: Migrate `actuals.py`**

```python
# app/data/actuals.py
"""Loaders for XM's actual predispatch results (the evaluation targets)."""

from datetime import date

import numpy as np
import pandas as pd

from app.storage import get_storage


def load_actual_price(dispatch_date: date, data_dir: str = "data") -> np.ndarray:
    """XM marginal price (MPO) for the date as a 24-length float array."""
    storage = get_storage(data_dir)
    with storage.open(f"preideal_price/{dispatch_date}.txt") as f:
        df = pd.read_csv(f, header=None)
    return df.iloc[0, 1:].astype(float).values


def load_actual_dispatch(dispatch_date: date, data_dir: str = "data") -> pd.DataFrame:
    """XM predispatch generation matrix for the date (raw, latin1-encoded)."""
    storage = get_storage(data_dir)
    with storage.open(f"preideal_dispatch/{dispatch_date}.txt") as f:
        return pd.read_csv(f, header=None, encoding="latin1")
```

- [ ] **Step 5: Run tests to verify no regression**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_loaders.py tests/test_paths.py tests/test_actuals.py -v`
Expected: all pass, unmodified assertions

- [ ] **Step 6: Commit**

```bash
git add app/data/loaders.py app/data/paths.py app/data/actuals.py
git commit -m "refactor: route loaders/paths/actuals reads through Storage"
```

---

### Task 3: Migrate `download.py` to `Storage`

**Files:**
- Modify: `app/data/download.py`
- Test: `tests/test_download.py` (new)

**Interfaces:**
- Consumes: `get_storage`, `Storage` from Task 1.
- Produces: `save_file(file_type: str, file_date: date, storage: Storage) -> None` (signature changes — gains a required `storage` param, drops the hardcoded `"data"` root it previously ignored `data_dir` for). `ensure_data_for_date(dispatch_date: date, data_dir: str = "data") -> Path` unchanged signature.

This also fixes a latent bug: `save_file` previously wrote to a hardcoded `Path("data")` regardless of the caller's `data_dir`. Passing `storage` (built from `data_dir` in `ensure_data_for_date`) makes it respect the actual root.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_download.py
import json
from datetime import date

from app.data.download import ensure_data_for_date, save_file
from app.storage import LocalStorage


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.content = payload if isinstance(payload, bytes) else b""

    def json(self):
        return self._payload


def test_save_file_writes_via_storage(monkeypatch, tmp_path):
    calls = iter([
        _FakeResponse({"ficheros": [{"nombre": "OFEI0418.txt"}]}),
        _FakeResponse({"url": "https://example.invalid/OFEI0418.txt"}),
        _FakeResponse(b"file-contents"),
    ])
    monkeypatch.setattr(
        "app.data.download.requests.get", lambda *a, **k: next(calls)
    )
    storage = LocalStorage(str(tmp_path))
    save_file(file_type="OFEI", file_date=date(2024, 4, 18), storage=storage)
    assert (tmp_path / "2024-04-18" / "OFEI0418.txt").read_text() == "file-contents"


def test_ensure_data_for_date_skips_when_folder_exists(monkeypatch, tmp_path):
    (tmp_path / "2024-04-18").mkdir()
    (tmp_path / "2024-04-18" / "marker.txt").write_text("already here")

    def _boom(*a, **k):
        raise AssertionError("save_file should not be called when folder exists")

    monkeypatch.setattr("app.data.download.save_file", _boom)
    folder = ensure_data_for_date(date(2024, 4, 18), data_dir=str(tmp_path))
    assert folder == tmp_path / "2024-04-18"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_download.py -v`
Expected: FAIL — `save_file() missing 1 required positional argument: 'storage'` (current signature has no `storage` param)

- [ ] **Step 3: Write the implementation**

```python
# app/data/download.py
from datetime import date
from pathlib import Path
import os

import requests

from app.storage import Storage, get_storage


PARAMS = {
    "OFEI": {
        "initial_path": "M:/InformacionAgentes/Usuarios/Publico/OFERTAS/INICIAL",
    },
    "dCondIniU": {
        "initial_path": "M:/InformacionAgentes/Usuarios/Publico/DESPACHO",
    },
    "dCondIniP": {
        "initial_path": "M:/InformacionAgentes/Usuarios/Publico/DESPACHO",
    },
    "PrId": {
        "initial_path": "M:/InformacionAgentes/Usuarios/Publico/PredespachoIdeal",
    },
    "iMAR": {
        "initial_path": "M:/InformacionAgentes/Usuarios/Publico/PredespachoIdeal",
    },
}

XM_DOWNLOAD_URL = "https://app-portalxmcore01.azurewebsites.net/administracion-archivos/ficheros/descarga-archivo"


def save_file(file_type: str, file_date: date, storage: Storage) -> None:
    init_path = PARAMS[file_type]["initial_path"]
    path = os.path.join(init_path, f"{file_date.year}-{file_date.month:0>2}")
    complement = "_NAL" if file_type in {"PrId", "iMAR"} else ""
    filename_ = f"{file_type}{file_date.month:0>2}{file_date.day:0>2}{complement}"

    container_name: str = ("storageportalxm",)
    ordenarPor: str = ("nombre",)
    orden: str = ("DESC",)
    pagina: int = (1,)
    resultadosPorPagina: int = (10,)
    response = requests.get(
        url="https://app-portalxmcore01.azurewebsites.net/administracion-archivos/ficheros",
        params={
            "nombre": f"{filename_}.txt",
            "ruta": f"/{path}",
            "contenedor": container_name,
            "ordenarPor": ordenarPor,
            "orden": orden,
            "pagina": pagina,
            "resultadosPorPagina": resultadosPorPagina,
        },
    )
    filename = response.json()["ficheros"][0]["nombre"]
    # Fetch url to download
    print(f"...DOwnloading file {filename}")
    r = requests.get(
        XM_DOWNLOAD_URL,
        params={
            "ruta": f"{path}/{filename}",
            "fileName": filename,
        },
    )
    url = r.json()["url"]
    file_byte = requests.get(url).content
    with storage.open(f"{file_date}/{filename_}.txt", "w") as file:
        file.write(file_byte.decode("latin-1"))


def ensure_data_for_date(dispatch_date: date, data_dir: str = "data") -> Path:
    """Download the per-day XM files into data/{date}/ if the folder is absent."""
    storage = get_storage(data_dir)
    folder_rel = str(dispatch_date)
    if storage.list_dir(folder_rel):
        print("... files already downloaded. Skipping download")
        return Path(data_dir) / folder_rel
    for file_type in PARAMS:
        save_file(file_type=file_type, file_date=dispatch_date, storage=storage)
    return Path(data_dir) / folder_rel
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_download.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add app/data/download.py tests/test_download.py
git commit -m "refactor: route download.py through Storage, fix save_file ignoring data_dir"
```

---

### Task 4: Migrate `results.py::save_results` writes to `Storage`

**Files:**
- Modify: `app/pipeline/results.py`
- Test: `tests/test_results.py` (existing, unmodified assertions for the two existing tests)

**Interfaces:**
- Consumes: `get_storage` from Task 1.
- Produces: `save_results(model, case, out) -> RunResult` unchanged signature/behavior. `extract_mpo`, `extract_dispatch` unchanged (not touched).

- [ ] **Step 1: Confirm baseline**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_results.py -v`
Expected: all pass (baseline)

- [ ] **Step 2: Migrate `save_results`**

```python
# app/pipeline/results.py
"""Extract and persist dispatch results from a solved model.

The marginal price (MPO) is the dual of the power_balance constraint. The sign
is multiplied by the objective sense so that maximize (BESS welfare) and
minimize (cost) cases both yield a positive price.
"""

import pandas as pd
import pyomo.environ as pyo

from app.schemas import DispatchCase, RunResult
from app.storage import get_storage


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


def save_results(model, case: DispatchCase, out: str = "data/results") -> RunResult:
    storage = get_storage(out)
    t = case.level.value

    dispatch = extract_dispatch(model)
    dispatch_name = f"dispatch_by_gen-{case.dispatch_date}-{t}.csv"
    with storage.open(dispatch_name, "w") as f:
        dispatch.to_csv(f, sep=",", index=False)

    mpo = extract_mpo(model)
    price_name = f"marginal_price-{case.dispatch_date}-{t}.csv"
    with storage.open(price_name, "w") as f:
        pd.DataFrame(
            data=mpo.values(), index=mpo.keys(), columns=["ideal_marginal_price"]
        ).reset_index(drop=False, names=["datetime"]).to_csv(f, sep=",", index=False)

    return RunResult(
        case=case,
        ok=True,
        dispatch_path=f"{out}/{dispatch_name}",
        price_path=f"{out}/{price_name}",
    )
```

- [ ] **Step 3: Run tests to verify no regression**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_results.py -v`
Expected: all pass, unmodified assertions

- [ ] **Step 4: Commit**

```bash
git add app/pipeline/results.py
git commit -m "refactor: route save_results writes through Storage"
```

---

### Task 5: Migrate `cli.py::_available_dates` to `Storage`

**Files:**
- Modify: `app/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `get_storage` from Task 1.
- Produces: `_available_dates(data_dir: str) -> list[date]` unchanged signature.

This is a behavior-preserving refactor like Tasks 2–4: the new test documents
`_available_dates`' contract and already passes against the *current*
`Path.glob`/`f.is_dir()` implementation (`is_dir()` already excludes
`notes.txt`) — its job here is to pin that contract down before the
implementation swap, and to keep passing after it.

- [ ] **Step 1: Add the regression test and confirm it passes on current code**

```python
# add to tests/test_cli.py
def test_available_dates_reads_condicion_inicial_dirs(tmp_path):
    root = tmp_path / "condicion_inicial"
    root.mkdir()
    (root / "2024-04-18").mkdir()
    (root / "2024-04-19").mkdir()
    (root / "notes.txt").write_text("not a date dir")
    dates = cli._available_dates(str(tmp_path))
    assert sorted(dates) == [date(2024, 4, 18), date(2024, 4, 19)]
```

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_cli.py::test_available_dates_reads_condicion_inicial_dirs -v`
Expected: PASS (baseline, on the pre-migration implementation)

- [ ] **Step 2: Migrate the implementation**

```python
# in app/cli.py, replace _available_dates
from app.storage import get_storage


def _available_dates(data_dir: str) -> list[date]:
    storage = get_storage(data_dir)
    out: list[date] = []
    for name in storage.list_dir("condicion_inicial"):
        parts = name.split("-")
        if len(parts) != 3:
            continue
        try:
            y, m, d = (int(x) for x in parts)
        except ValueError:
            continue
        out.append(date(y, m, d))
    return out
```

- [ ] **Step 3: Run tests to verify no regression**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_cli.py -v`
Expected: all pass, including the new test

- [ ] **Step 4: Commit**

```bash
git add app/cli.py tests/test_cli.py
git commit -m "refactor: route _available_dates through Storage"
```

---

### Task 6: BESS scenario YAML library (`scenarios.py`)

**Files:**
- Create: `app/pipeline/scenarios.py`
- Create: `scenarios/bess/20pct_arbitrage.yaml`
- Create: `scenarios/bess/10pct_grid_asset.yaml`
- Test: `tests/test_scenarios.py`

**Interfaces:**
- Consumes: `get_storage`, `Storage` from Task 1. `BessScenario` from `app.schemas.bess`.
- Produces: `load_bess_scenario(name_or_path: str, storage: Storage | None = None) -> BessScenario`, importable as `from app.pipeline.scenarios import load_bess_scenario`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenarios.py
import pytest

from app.pipeline.scenarios import load_bess_scenario
from app.schemas.bess import BessMode


def test_loads_named_scenario_from_library():
    scenario = load_bess_scenario("20pct_arbitrage")
    assert scenario.mode == BessMode.arbitrage
    assert scenario.penetration_level == "20pct"
    assert scenario.units[0].name == "BESS1"
    assert scenario.units[0].charge_bid == 20.0


def test_loads_grid_asset_named_scenario():
    scenario = load_bess_scenario("10pct_grid_asset")
    assert scenario.mode == BessMode.grid_asset
    assert scenario.units[0].charge_bid is None


def test_loads_literal_path(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "mode: grid_asset\n"
        "penetration_level: custom\n"
        "units:\n"
        "  - name: B1\n"
        "    mwh_nom: 50.0\n"
        "    hours_to_deplete: 2.0\n"
        "    initial_soc: 0.5\n"
        "    min_soc: 0.0\n"
        "    max_soc: 1.0\n"
        "    efficiency: 0.9\n"
    )
    scenario = load_bess_scenario(str(path))
    assert scenario.penetration_level == "custom"


def test_unknown_name_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_bess_scenario("does-not-exist-anywhere")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_scenarios.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.scenarios'`

- [ ] **Step 3: Write the implementation**

```python
# app/pipeline/scenarios.py
"""Load reusable BESS scenarios from the scenarios/bess/ YAML library."""

import yaml

from app.schemas.bess import BessScenario
from app.storage import Storage, get_storage

SCENARIOS_ROOT = "scenarios/bess"


def load_bess_scenario(name_or_path: str, storage: Storage | None = None) -> BessScenario:
    """Resolve `name_or_path` to a BessScenario.

    If `scenarios/bess/{name_or_path}.yaml` exists (relative to the current
    working directory, or under `storage` if given), load it from there.
    Otherwise treat `name_or_path` as a literal filesystem path.
    """
    library = storage or get_storage(".")
    candidate = f"{SCENARIOS_ROOT}/{name_or_path}.yaml"
    if library.exists(candidate):
        with library.open(candidate) as f:
            data = yaml.safe_load(f)
    else:
        with open(name_or_path, "r") as f:
            data = yaml.safe_load(f)
    return BessScenario.parse_obj(data)
```

```yaml
# scenarios/bess/20pct_arbitrage.yaml
mode: arbitrage
penetration_level: "20pct"
units:
  - name: BESS1
    mwh_nom: 200.0
    hours_to_deplete: 4.0
    initial_soc: 0.5
    min_soc: 0.1
    max_soc: 0.9
    efficiency: 0.9
    charge_bid: 20.0
    discharge_bid: 180.0
```

```yaml
# scenarios/bess/10pct_grid_asset.yaml
mode: grid_asset
penetration_level: "10pct"
units:
  - name: BESS1
    mwh_nom: 100.0
    hours_to_deplete: 4.0
    initial_soc: 0.5
    min_soc: 0.1
    max_soc: 0.9
    efficiency: 0.9
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_scenarios.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/scenarios.py scenarios/bess tests/test_scenarios.py
git commit -m "feat: add BESS scenario YAML library and loader"
```

---

### Task 7: `--bess-scenario` on `run`

**Files:**
- Modify: `app/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_bess_scenario` from Task 6.
- Produces: `run` command gains `--bess-scenario` option; no change to `run`'s other behavior.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_cli.py
def test_run_loads_bess_scenario(monkeypatch):
    _stub_dates(monkeypatch)
    captured = {}

    def fake_run_many(cases, **k):
        captured["cases"] = cases
        return [RunResult(case=c, ok=True) for c in cases]

    monkeypatch.setattr(cli, "run_many", fake_run_many)
    result = runner.invoke(
        cli.app, ["run", "2024-04-18", "-t", "preideal", "--bess-scenario", "20pct_arbitrage"]
    )
    assert result.exit_code == 0
    case = captured["cases"][0]
    assert case.bess_scenario is not None
    assert case.bess_scenario.penetration_level == "20pct"


def test_run_without_bess_scenario_flag_has_none(monkeypatch):
    _stub_dates(monkeypatch)
    captured = {}

    def fake_run_many(cases, **k):
        captured["cases"] = cases
        return [RunResult(case=c, ok=True) for c in cases]

    monkeypatch.setattr(cli, "run_many", fake_run_many)
    runner.invoke(cli.app, ["run", "2024-04-18", "-t", "preideal"])
    assert captured["cases"][0].bess_scenario is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_cli.py::test_run_loads_bess_scenario -v`
Expected: FAIL — `No such option: --bess-scenario`

- [ ] **Step 3: Write the implementation**

```python
# app/cli.py — add import
from app.pipeline.scenarios import load_bess_scenario
```

```python
# app/cli.py — modify the `run` command signature and body
@app.command()
def run(
    dates: str = typer.Argument(
        None, help="YYYY-MM-DD | range a:b | YYYY-MM | omit = all available"
    ),
    type: list[str] = typer.Option(
        ["preideal"], "--type", "-t", help="dispatch level (preideal/ideal), repeatable, or 'all'"
    ),
    solver: str = typer.Option("cbc", help="pyomo solver name"),
    eval: bool = typer.Option(True, "--eval/--no-eval", help="evaluate vs XM actuals"),
    prices: bool = typer.Option(
        True, "--prices/--no-prices", help="fix-integers LP pricing re-solve"
    ),
    bess_scenario: str = typer.Option(
        None, "--bess-scenario", help="named scenario (scenarios/bess/<name>.yaml) or path to a scenario YAML"
    ),
    skip_dates: str = typer.Option("", help="comma-separated YYYY-MM-DD to skip"),
    out: str = typer.Option("data/results", help="results directory"),
    data_dir: str = typer.Option("data", help="input data directory"),
):
    avail = _available_dates(data_dir)
    selected = parse_dates_arg(dates, avail)
    skip = _parse_skip(skip_dates)
    selected = [d for d in selected if d not in skip]

    scenario = load_bess_scenario(bess_scenario) if bess_scenario else None

    levels = list(DispatchLevel) if "all" in type else [DispatchLevel(t) for t in type]
    cases = [
        DispatchCase(
            dispatch_date=d, level=lvl, solver=solver, compute_prices=prices,
            bess_scenario=scenario,
        )
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

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_cli.py -v`
Expected: all pass, including the two new tests

- [ ] **Step 5: Commit**

```bash
git add app/cli.py tests/test_cli.py
git commit -m "feat: add --bess-scenario flag to run command"
```

---

### Task 8: BESS results persistence (`extract_bess`, CSV, `RunResult.bess_path`/`bess_summary`)

**Files:**
- Modify: `app/schemas/run_result.py`
- Modify: `app/pipeline/results.py`
- Test: `tests/test_schemas_run_result.py`, `tests/test_results.py`

**Interfaces:**
- Consumes: `extract_mpo` (existing, Task 4). `BessScenario`/`BessMode`/`BessUnit` from `app.schemas.bess`. `bess_scenario_to_params` from `app.pipeline.case_builder` (test only).
- Produces: `RunResult.bess_path: str | None = None`, `RunResult.bess_summary: dict[str, float] | None = None`. `extract_bess(model, mpo: dict) -> pd.DataFrame` with columns `unit, datetime, charge, discharge, soc, revenue, cost`. `save_results` writes `bess_results-{date}-{type}.csv` and populates both new `RunResult` fields when `case.bess_scenario is not None`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_schemas_run_result.py
def test_run_result_bess_fields_default_none():
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    r = RunResult(case=case, ok=True)
    assert r.bess_path is None
    assert r.bess_summary is None


def test_run_result_bess_fields_settable():
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    r = RunResult(
        case=case, ok=True, bess_path="bess.csv",
        bess_summary={"bess_net_revenue": 100.0},
    )
    assert r.bess_path == "bess.csv"
    assert r.bess_summary["bess_net_revenue"] == 100.0
```

```python
# add to tests/test_results.py
import pandas as pd

from app.pipeline.case_builder import bess_scenario_to_params
from app.schemas.bess import BessMode, BessScenario, BessUnit


def _bess_case_and_model():
    ts = [1]
    set_data = {
        "G": [], "I": ["A"], "T": ts, "combined_cycle": [],
        "excluded_resource": {}, "gen_on": [], "gen_off": [],
    }
    param_data = {
        "Pmin": {("A", 1): 0.0},
        "Pmax": {("A", 1): 100.0},
        "max_min_op": 0, "ramp_up": {}, "ramp_down": {},
        "beta": {"A": 50.0}, "cold_start": {},
        "demand": {1: 80.0}, "TMG": {}, "Ton": {}, "z_on_t0_minus_1": {},
    }
    scenario = BessScenario(
        mode=BessMode.arbitrage, penetration_level="test",
        units=[BessUnit(
            name="B1", mwh_nom=40.0, hours_to_deplete=4.0, initial_soc=0.5,
            min_soc=0.1, max_soc=0.9, efficiency=0.9,
            charge_bid=5.0, discharge_bid=45.0,
        )],
    )
    bess_names, bess_params = bess_scenario_to_params(scenario)
    set_data["BESS"] = bess_names
    param_data.update(bess_params)

    case = DispatchCase(
        dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal,
        bess_scenario=scenario, solver="cbc",
    )
    m = UnitCommitmentModel(case=case)
    m.create_model(set_data=set_data, param_data=param_data)
    m.solve(solver="cbc")
    return m, case


def test_save_results_writes_bess_csv_and_summary(tmp_path):
    m, case = _bess_case_and_model()
    result = save_results(m, case, out=str(tmp_path))

    assert result.bess_path is not None
    bess_csv = tmp_path / f"bess_results-{case.dispatch_date}-{case.level.value}.csv"
    assert bess_csv.exists()

    df = pd.read_csv(bess_csv)
    assert set(df.columns) == {"unit", "datetime", "charge", "discharge", "soc", "revenue", "cost"}

    mpo = extract_mpo(m)
    price = list(mpo.values())[0]
    row = df.iloc[0]
    assert abs(row["revenue"] - row["discharge"] * price * 1000.0) < 1e-6
    assert abs(row["cost"] - row["charge"] * price * 1000.0) < 1e-6

    assert result.bess_summary is not None
    assert abs(result.bess_summary["bess_net_revenue"] - (df["revenue"] - df["cost"]).sum()) < 1e-6


def test_save_results_without_bess_scenario_has_no_bess_fields(tmp_path):
    m, case = _toy_model()
    result = save_results(m, case, out=str(tmp_path))
    assert result.bess_path is None
    assert result.bess_summary is None
```

This reuses the module's existing `_toy_model` helper (defined earlier in
`tests/test_results.py`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_schemas_run_result.py tests/test_results.py -v`
Expected: FAIL — `RunResult` has no field `bess_path`/`bess_summary` (pydantic will reject unknown kwargs), and `extract_mpo`/`save_results` import paths for the new test's imports resolve but assertions on `result.bess_path` fail with `AttributeError`.

- [ ] **Step 3: Write the implementation**

```python
# app/schemas/run_result.py
from pydantic import BaseModel

from app.schemas.case import DispatchCase


class RunResult(BaseModel):
    case: DispatchCase
    ok: bool
    dispatch_path: str | None = None
    price_path: str | None = None
    bess_path: str | None = None
    bess_summary: dict[str, float] | None = None
    metrics_path: str | None = None
    metrics: dict[str, float] | None = None
    error: str | None = None
```

```python
# app/pipeline/results.py — add extract_bess and wire it into save_results
def extract_bess(model, mpo: dict) -> pd.DataFrame:
    """Per-unit x hour BESS activity, settled at the system marginal price
    (MPO), not at the unit's own bid: the bid is an optimization input, and
    grid_asset units often have no bids at all. revenue/cost are in COP
    (energy in MWh x price in COP/kWh x 1000 kWh/MWh)."""
    charge = {(b, t): pyo.value(v) for (b, t), v in model._model.bess_charge.items()}
    discharge = {(b, t): pyo.value(v) for (b, t), v in model._model.bess_discharge.items()}
    soc = {(b, t): pyo.value(v) for (b, t), v in model._model.soc_bess.items()}

    rows = []
    for key in sorted(charge.keys()):
        b, t = key
        price = mpo.get(t, 0.0)
        c, d = charge[key], discharge[key]
        rows.append({
            "unit": b,
            "datetime": t,
            "charge": c,
            "discharge": d,
            "soc": soc[key],
            "revenue": d * price * 1000.0,
            "cost": c * price * 1000.0,
        })
    return pd.DataFrame(rows)


def _bess_summary(bess_df: pd.DataFrame) -> dict[str, float]:
    return {
        "bess_charge_mwh": float(bess_df["charge"].sum()),
        "bess_discharge_mwh": float(bess_df["discharge"].sum()),
        "bess_avg_soc": float(bess_df["soc"].mean()),
        "bess_net_revenue": float((bess_df["revenue"] - bess_df["cost"]).sum()),
    }


def save_results(model, case: DispatchCase, out: str = "data/results") -> RunResult:
    storage = get_storage(out)
    t = case.level.value

    dispatch = extract_dispatch(model)
    dispatch_name = f"dispatch_by_gen-{case.dispatch_date}-{t}.csv"
    with storage.open(dispatch_name, "w") as f:
        dispatch.to_csv(f, sep=",", index=False)

    mpo = extract_mpo(model)
    price_name = f"marginal_price-{case.dispatch_date}-{t}.csv"
    with storage.open(price_name, "w") as f:
        pd.DataFrame(
            data=mpo.values(), index=mpo.keys(), columns=["ideal_marginal_price"]
        ).reset_index(drop=False, names=["datetime"]).to_csv(f, sep=",", index=False)

    bess_path = None
    bess_summary = None
    if case.bess_scenario is not None:
        bess_df = extract_bess(model, mpo)
        bess_name = f"bess_results-{case.dispatch_date}-{t}.csv"
        with storage.open(bess_name, "w") as f:
            bess_df.to_csv(f, sep=",", index=False)
        bess_path = f"{out}/{bess_name}"
        bess_summary = _bess_summary(bess_df)

    return RunResult(
        case=case,
        ok=True,
        dispatch_path=f"{out}/{dispatch_name}",
        price_path=f"{out}/{price_name}",
        bess_path=bess_path,
        bess_summary=bess_summary,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_schemas_run_result.py tests/test_results.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add app/schemas/run_result.py app/pipeline/results.py tests/test_schemas_run_result.py tests/test_results.py
git commit -m "feat: persist BESS charge/discharge/soc/revenue results"
```

---

### Task 9: Consolidated summary (`run_many` — every-`ok`-row, `scenario` column, BESS columns)

**Files:**
- Modify: `app/pipeline/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `RunResult.bess_summary`, `RunResult.bess_path` from Task 8. `bess_scenario_to_params`, `BessScenario`, `BessMode`, `BessUnit` (test only).
- Produces: `run_many` writes `metrics-summary.csv` with one row per `r.ok` (not `r.ok and r.metrics`), columns `date, type, scenario, <metrics...>, <bess_summary...>`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_runner.py
import pandas as pd

from app.pipeline.case_builder import bess_scenario_to_params
from app.schemas.bess import BessMode, BessScenario, BessUnit


def test_summary_includes_every_ok_row_even_without_metrics(monkeypatch, tmp_path):
    good = date(2024, 4, 18)

    def fake_build(case, inputs, **kw):
        return _toy_case()

    monkeypatch.setattr(runner, "build_case", fake_build)
    case = DispatchCase(dispatch_date=good, level=DispatchLevel.preideal, solver="cbc")
    runner.run_many([case], evaluate=False, out=str(tmp_path))

    summary = pd.read_csv(tmp_path / "metrics-summary.csv")
    assert len(summary) == 1
    assert summary.iloc[0]["scenario"] == "baseline"


def test_summary_scenario_column_and_bess_totals(monkeypatch, tmp_path):
    date_ = date(2024, 4, 18)
    scenario = BessScenario(
        mode=BessMode.arbitrage, penetration_level="10pct",
        units=[BessUnit(
            name="B1", mwh_nom=40.0, hours_to_deplete=4.0, initial_soc=0.5,
            min_soc=0.1, max_soc=0.9, efficiency=0.9,
            charge_bid=5.0, discharge_bid=45.0,
        )],
    )

    def fake_build(case, inputs, **kw):
        set_data = {
            "G": [], "I": ["A"], "T": [1], "combined_cycle": [],
            "excluded_resource": {}, "gen_on": [], "gen_off": [],
        }
        param_data = {
            "Pmin": {("A", 1): 0.0}, "Pmax": {("A", 1): 100.0},
            "max_min_op": 0, "ramp_up": {}, "ramp_down": {},
            "beta": {"A": 50.0}, "cold_start": {},
            "demand": {1: 80.0}, "TMG": {}, "Ton": {}, "z_on_t0_minus_1": {},
        }
        bess_names, bess_params = bess_scenario_to_params(scenario)
        set_data["BESS"] = bess_names
        param_data.update(bess_params)
        return set_data, param_data, {}

    monkeypatch.setattr(runner, "build_case", fake_build)
    case = DispatchCase(
        dispatch_date=date_, level=DispatchLevel.preideal,
        bess_scenario=scenario, solver="cbc",
    )
    runner.run_many([case], evaluate=False, out=str(tmp_path))

    summary = pd.read_csv(tmp_path / "metrics-summary.csv")
    assert summary.iloc[0]["scenario"] == "10pct"
    assert "bess_net_revenue" in summary.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_runner.py -v`
Expected: FAIL — `test_summary_includes_every_ok_row_even_without_metrics` fails because current code only emits rows `if r.ok and r.metrics` (no metrics here, so `metrics-summary.csv` is never written); `test_summary_scenario_column_and_bess_totals` fails for the same reason plus missing `scenario`/`bess_net_revenue` columns.

- [ ] **Step 3: Write the implementation**

```python
# app/pipeline/runner.py — replace the row-building block in run_many
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
        {
            "date": r.case.dispatch_date,
            "type": r.case.level.value,
            "scenario": (
                r.case.bess_scenario.penetration_level
                if r.case.bess_scenario is not None
                else "baseline"
            ),
            **(r.metrics or {}),
            **(r.bess_summary or {}),
        }
        for r in results
        if r.ok
    ]
    if rows:
        from pathlib import Path

        Path(out).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(f"{out}/metrics-summary.csv", index=False)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_runner.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/runner.py tests/test_runner.py
git commit -m "feat: emit a summary row for every ok run, with scenario and BESS totals"
```

---

### Task 10: `fetch` command

**Files:**
- Modify: `app/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ensure_data_for_date` from `app.data.download` (imported at module level in `cli.py` so it's monkeypatchable as `cli.ensure_data_for_date`).
- Produces: `fetch` Typer command; `_enumerate_dates(token: str) -> list[date]` helper (not shared with `parse_dates_arg`, which filters against an `available` list that has no meaning for not-yet-downloaded dates).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_cli.py
def test_fetch_calls_ensure_data_for_date_for_each_date_in_range(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli, "ensure_data_for_date", lambda d, data_dir: calls.append(d)
    )
    result = runner.invoke(cli.app, ["fetch", "2024-04-18:2024-04-19"])
    assert result.exit_code == 0
    assert calls == [date(2024, 4, 18), date(2024, 4, 19)]


def test_fetch_single_date(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli, "ensure_data_for_date", lambda d, data_dir: calls.append(d)
    )
    result = runner.invoke(cli.app, ["fetch", "2024-04-18"])
    assert result.exit_code == 0
    assert calls == [date(2024, 4, 18)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_cli.py::test_fetch_calls_ensure_data_for_date_for_each_date_in_range -v`
Expected: FAIL — `Error: No such command 'fetch'`

- [ ] **Step 3: Write the implementation**

```python
# app/cli.py — add imports at top
import calendar
from datetime import timedelta

from app.data.download import ensure_data_for_date
```

```python
# app/cli.py — add helper and command
def _enumerate_dates(token: str) -> list[date]:
    """Enumerate every date in `token`, with no filtering against what's
    already on disk (unlike parse_dates_arg) — fetch's whole point is to
    reach dates that aren't available locally yet."""
    token = token.strip()
    if ":" in token:
        lo, hi = (datetime.strptime(p.strip(), "%Y-%m-%d").date() for p in token.split(":", 1))
        return [lo + timedelta(days=i) for i in range((hi - lo).days + 1)]
    parts = token.split("-")
    if len(parts) == 2:
        year, month = int(parts[0]), int(parts[1])
        days_in_month = calendar.monthrange(year, month)[1]
        return [date(year, month, d) for d in range(1, days_in_month + 1)]
    return [datetime.strptime(token, "%Y-%m-%d").date()]


@app.command()
def fetch(
    dates: str = typer.Argument(..., help="YYYY-MM-DD | range a:b | YYYY-MM"),
    data_dir: str = typer.Option("data", help="input data directory"),
):
    """Download raw XM inputs for the given date(s) without running the model."""
    selected = _enumerate_dates(dates)
    for d in selected:
        typer.echo(f"==> fetching {d}")
        ensure_data_for_date(d, data_dir=data_dir)
    typer.echo(f"Done: fetched {len(selected)} date(s).")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_cli.py -v`
Expected: all pass, including the two new tests

- [ ] **Step 5: Commit**

```bash
git add app/cli.py tests/test_cli.py
git commit -m "feat: add fetch command"
```

---

### Task 11: `evaluate` command (post-hoc re-scoring)

**Files:**
- Create: `app/pipeline/evaluate.py`
- Modify: `app/cli.py`
- Test: `tests/test_evaluate.py` (new), `tests/test_cli.py`

**Interfaces:**
- Consumes: `get_storage` from Task 1, `load_actual_price` from `app.data.actuals`, `price_metrics` from `app.utils.metrics`, `DispatchLevel` from `app.schemas`.
- Produces: `evaluate_saved_run(dispatch_date: date, level: DispatchLevel, *, out: str = "data/results", data_dir: str = "data") -> dict[str, float]` — raises `FileNotFoundError` if no saved price CSV exists for that date/level. Writes `metrics-{date}-{type}.csv` as a side effect, same as the inline `run --eval` path. `evaluate` Typer command.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate.py
from datetime import date

import pandas as pd
import pytest

import app.pipeline.runner as runner
from app.pipeline.evaluate import evaluate_saved_run
from app.schemas import DispatchCase, DispatchLevel


def test_evaluate_saved_run_writes_metrics_csv(tmp_path):
    price_df = pd.DataFrame({
        "datetime": pd.date_range("2024-04-18", periods=24, freq="1h"),
        "ideal_marginal_price": [float(i) for i in range(24)],
    })
    price_df.to_csv(tmp_path / "marginal_price-2024-04-18-preideal.csv", index=False)

    actuals_dir = tmp_path / "preideal_price"
    actuals_dir.mkdir()
    row = "MPO," + ",".join(str(float(i)) for i in range(24))
    (actuals_dir / "2024-04-18.txt").write_text(row + "\n")

    metrics = evaluate_saved_run(
        date(2024, 4, 18), DispatchLevel.preideal, out=str(tmp_path), data_dir=str(tmp_path)
    )
    assert metrics["mae"] == 0.0
    assert (tmp_path / "metrics-2024-04-18-preideal.csv").exists()


def test_evaluate_saved_run_missing_price_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_saved_run(
            date(2024, 4, 18), DispatchLevel.preideal, out=str(tmp_path), data_dir=str(tmp_path)
        )


def test_evaluate_matches_inline_eval_exactly(monkeypatch, tmp_path):
    """Acceptance test: evaluate must reproduce `run --eval`'s numbers,
    including the sort-by-datetime the inline path applies via
    extract_mpo_sorted (the saved CSV is written in dual-iteration order,
    which is not guaranteed sorted)."""
    ts = [pd.Timestamp("2024-04-18 00:00"), pd.Timestamp("2024-04-18 01:00")]

    def fake_build(case, inputs, **kw):
        set_data = {
            "G": [], "I": ["A", "B"], "T": ts, "combined_cycle": [],
            "excluded_resource": {}, "gen_on": [], "gen_off": [],
        }
        param_data = {
            "Pmin": {("A", t): 0.0 for t in ts} | {("B", t): 0.0 for t in ts},
            "Pmax": {("A", t): 100.0 for t in ts} | {("B", t): 100.0 for t in ts},
            "max_min_op": 0, "ramp_up": {}, "ramp_down": {},
            "beta": {"A": 10.0, "B": 50.0}, "cold_start": {},
            "demand": {t: 150.0 for t in ts}, "TMG": {}, "Ton": {}, "z_on_t0_minus_1": {},
        }
        return set_data, param_data, {}

    monkeypatch.setattr(runner, "build_case", fake_build)

    actuals_dir = tmp_path / "preideal_price"
    actuals_dir.mkdir()
    row = "MPO," + ",".join(str(float(i)) for i in range(24))
    (actuals_dir / "2024-04-18.txt").write_text(row + "\n")

    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal, solver="cbc")
    inline_result = runner.run_case(case, evaluate=True, out=str(tmp_path), data_dir=str(tmp_path))
    assert inline_result.ok is True
    assert inline_result.metrics is not None

    post_hoc_metrics = evaluate_saved_run(
        date(2024, 4, 18), DispatchLevel.preideal, out=str(tmp_path), data_dir=str(tmp_path)
    )
    for key, value in inline_result.metrics.items():
        assert abs(value - post_hoc_metrics[key]) < 1e-9, f"{key}: inline={value} post_hoc={post_hoc_metrics[key]}"
```

```python
# add to tests/test_cli.py
def test_evaluate_command(monkeypatch):
    _stub_dates(monkeypatch)
    monkeypatch.setattr(cli, "evaluate_saved_run", lambda d, lvl, **k: {"mae": 1.0})
    result = runner.invoke(cli.app, ["evaluate", "2024-04-18", "-t", "preideal"])
    assert result.exit_code == 0
    assert "1 run(s) evaluated" in result.output


def test_evaluate_command_reports_missing_runs(monkeypatch):
    _stub_dates(monkeypatch)

    def _raise(d, lvl, **k):
        raise FileNotFoundError("no saved price CSV")

    monkeypatch.setattr(cli, "evaluate_saved_run", _raise)
    result = runner.invoke(cli.app, ["evaluate", "2024-04-18", "-t", "preideal"])
    assert result.exit_code == 1
    assert "No runs evaluated" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.evaluate'`

- [ ] **Step 3: Write the implementation**

```python
# app/pipeline/evaluate.py
"""Post-hoc evaluation: re-score a saved run's price CSV against XM actuals
without re-solving the model. Must reproduce the numbers `run --eval`
computes inline, including the datetime sort the inline path applies via
extract_mpo_sorted — the saved CSV is written in dual-iteration order, which
is not guaranteed sorted."""

from datetime import date

import pandas as pd

from app.data.actuals import load_actual_price
from app.schemas import DispatchLevel
from app.storage import get_storage
from app.utils.metrics import price_metrics


def evaluate_saved_run(
    dispatch_date: date,
    level: DispatchLevel,
    *,
    out: str = "data/results",
    data_dir: str = "data",
) -> dict[str, float]:
    storage = get_storage(out)
    t = level.value
    price_name = f"marginal_price-{dispatch_date}-{t}.csv"
    if not storage.exists(price_name):
        raise FileNotFoundError(f"no saved price CSV for {dispatch_date} [{t}] in {out}")

    with storage.open(price_name) as f:
        price_df = pd.read_csv(f, parse_dates=["datetime"]).sort_values("datetime")
    model_mpo = price_df["ideal_marginal_price"].to_numpy()

    xm = load_actual_price(dispatch_date, data_dir=data_dir)
    n = min(len(xm), len(model_mpo))
    metrics = price_metrics(xm[:n], model_mpo[:n])

    metrics_name = f"metrics-{dispatch_date}-{t}.csv"
    with storage.open(metrics_name, "w") as f:
        pd.DataFrame([metrics]).to_csv(f, index=False)
    return metrics
```

```python
# app/cli.py — add import
from app.pipeline.evaluate import evaluate_saved_run
```

```python
# app/cli.py — add command
@app.command()
def evaluate(
    dates: str = typer.Argument(
        None, help="YYYY-MM-DD | range a:b | YYYY-MM | omit = all available"
    ),
    type: list[str] = typer.Option(
        ["preideal"], "--type", "-t", help="dispatch level, repeatable, or 'all'"
    ),
    out: str = typer.Option("data/results", help="results directory"),
    data_dir: str = typer.Option("data", help="input data directory"),
):
    """Re-score saved runs against XM actuals without re-solving the model."""
    avail = _available_dates(data_dir)
    selected = parse_dates_arg(dates, avail)
    levels = list(DispatchLevel) if "all" in type else [DispatchLevel(t) for t in type]

    evaluated = 0
    for d in selected:
        for lvl in levels:
            try:
                evaluate_saved_run(d, lvl, out=out, data_dir=data_dir)
            except FileNotFoundError as e:
                typer.echo(f"  ! {d} [{lvl.value}]: {e}")
                continue
            typer.echo(f"==> evaluated {d} [{lvl.value}]")
            evaluated += 1

    if evaluated == 0:
        typer.echo("No runs evaluated.")
        raise typer.Exit(code=1)
    typer.echo(f"Done: {evaluated} run(s) evaluated.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_evaluate.py tests/test_cli.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/evaluate.py app/cli.py tests/test_evaluate.py tests/test_cli.py
git commit -m "feat: add evaluate command for post-hoc metric re-scoring"
```

---

### Task 12: `compare` command

**Files:**
- Modify: `app/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `get_storage` from Task 1.
- Produces: `compare` Typer command; `_read_summary(out: str) -> pd.DataFrame` helper in `cli.py`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_cli.py
import pandas as pd


def test_compare_outer_joins_summaries_on_date_type_scenario(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    pd.DataFrame([
        {"date": "2024-04-18", "type": "preideal", "scenario": "baseline", "mae": 1.0},
    ]).to_csv(a / "metrics-summary.csv", index=False)
    pd.DataFrame([
        {"date": "2024-04-18", "type": "preideal", "scenario": "baseline", "mae": 2.0},
        {"date": "2024-04-19", "type": "preideal", "scenario": "baseline", "mae": 3.0},
    ]).to_csv(b / "metrics-summary.csv", index=False)

    result = runner.invoke(cli.app, ["compare", str(a), str(b)])
    assert result.exit_code == 0
    assert "2024-04-19" in result.output
    assert "NaN" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_cli.py::test_compare_outer_joins_summaries_on_date_type_scenario -v`
Expected: FAIL — `Error: No such command 'compare'`

- [ ] **Step 3: Write the implementation**

```python
# app/cli.py — add import
import pandas as pd
from app.storage import get_storage
```

```python
# app/cli.py — add helper and command
def _read_summary(out: str) -> pd.DataFrame:
    storage = get_storage(out)
    with storage.open("metrics-summary.csv") as f:
        return pd.read_csv(f)


@app.command()
def compare(
    out_a: str = typer.Argument(..., help="first run's results directory"),
    out_b: str = typer.Argument(..., help="second run's results directory"),
):
    """Outer-join two runs' metrics-summary.csv on (date, type, scenario)."""
    df_a = _read_summary(out_a)
    df_b = _read_summary(out_b)
    merged = df_a.merge(
        df_b, on=["date", "type", "scenario"], how="outer", suffixes=("_a", "_b")
    )
    typer.echo(merged.to_string(index=False))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/test_cli.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add app/cli.py tests/test_cli.py
git commit -m "feat: add compare command"
```

---

## Final verification

- [ ] Run the full suite: `~/.local/share/virtualenvs/dam-worker-optimizer-*/bin/python -m pytest tests/ -v`
- [ ] Expected: all tests pass (baseline 41 + new tests from this plan).
- [ ] Manually smoke-test the CLI help: `python -m app --help` shows `run`, `fetch`, `evaluate`, `compare`.
