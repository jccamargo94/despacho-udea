# Fase 3: Backend API y persistencia — plan de implementacion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a FastAPI backend + Supabase-Postgres persistence + a DB-queue worker around the existing `app/` domain library, so runs can be launched/tracked/queried over HTTP instead of only via CLI, without touching the already-validated Pyomo/`case_builder`/`runner` pipeline.

**Architecture:** `app/db/` adds SQLAlchemy models + query helpers shared by two new thin services: `services/api/` (FastAPI, verifies Supabase JWTs, reads/writes the DB) and `services/worker/` (plain Python polling loop, claims pending `Run` rows via `FOR UPDATE SKIP LOCKED` on Postgres, calls `app.pipeline.runner.run_case` unmodified). Time-series results stay as CSV via the existing `Storage` abstraction; only status/metadata/scalar metrics live in the DB.

**Tech Stack:** FastAPI 0.141.1, SQLAlchemy 2.0.51 (2.0-style `select()`, not legacy `Query`), Alembic 1.19.0, `psycopg[binary]` 3.3.4, PyJWT 2.13.0 (`[crypto]` extra), Uvicorn 0.52.1. SQLite for tests, Supabase Postgres for dev/prod. No Celery, no Redis.

## Global Constraints

- Python 3.12, `uv`-managed. New runtime deps, pinned exact (verified resolvable together against the existing lockfile in this session): `fastapi==0.141.1`, `uvicorn[standard]==0.52.1`, `sqlalchemy==2.0.51`, `psycopg[binary]==3.3.4`, `alembic==1.19.0`, `pyjwt[crypto]==2.13.0`. New dev-only dep: `httpx==0.28.1` (required by FastAPI's `TestClient`).
- No Celery, no Redis, no message broker of any kind — the worker is a DB-polling loop only (spec decision, `AGENTS.md` explicitly excludes Celery/Redis from this repo's stack).
- `app/pipeline/case_builder.py`, `app/pipeline/runner.py`, `app/pipeline/results.py`, and everything under `app/model/` stay **unmodified**. Only `app/pipeline/evaluate.py` gets the scoped upsert fix (Task 6).
- Postgres-only SQL (`FOR UPDATE SKIP LOCKED`) is isolated to `app/db/claim.py`, gated on `session.bind.dialect.name == "postgresql"`. Every other DB-touching module must behave identically on SQLite (tests) and Postgres (dev/prod).
- No login/signup/password-reset endpoints in this service. Supabase Auth (used by the Fase 4 frontend, out of scope here) issues the JWTs; this backend only verifies them.
- Time-series results (dispatch, prices, BESS activity) are **not** duplicated into the DB — they stay as CSV via the existing `Storage`/`LocalStorage` abstraction, unchanged. Only `Run`/`Case`/`Scenario`/`MetricSet` (status, metadata, scalar metrics) go into Postgres.
- `services/api` and `services/worker` Docker images build with `context: .` (repo root) and an explicit `dockerfile:` path — not `build: ./services/api` — because both need `app/` at build time.
- Tests follow this repo's existing conventions: plain `pytest`, `tmp_path`/`monkeypatch`, fixtures live in `tests/conftest.py`. No new test framework beyond FastAPI's own `TestClient` (which itself needs `httpx`).
- All work happens on branch `fase3-api-persistencia` (already created off `develop`, holds the spec commit). Never commit directly to `develop` — PR at the end, per `AGENTS.md`.
- Run `uv run ruff check --fix <files>` and `uv run ruff format <files>` before each commit (pre-commit already enforces this; running it manually avoids a red pre-commit hook mid-task).

---

## Task 1: Add API/DB/worker dependencies

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `fastapi`, `uvicorn[standard]`, `sqlalchemy`, `psycopg[binary]`, `alembic`, `pyjwt[crypto]` importable from any module in the project; `httpx` importable in tests.

- [ ] **Step 1: Add the runtime dependencies**

In `pyproject.toml`, in the `dependencies` list (after `"pydataxm==0.3.6",`), add:

```toml
    "fastapi==0.141.1",
    "uvicorn[standard]==0.52.1",
    "sqlalchemy==2.0.51",
    "psycopg[binary]==3.3.4",
    "alembic==1.19.0",
    "pyjwt[crypto]==2.13.0",
```

- [ ] **Step 2: Add the dev-only dependency**

In `pyproject.toml`, in `[dependency-groups] dev`, add `"httpx==0.28.1",` (after `"pytest",`).

- [ ] **Step 3: Sync and lock**

Run: `uv sync --group dev`
Expected: resolves cleanly, installs the packages listed above (and their transitive deps: `anyio`, `starlette`, `cffi`, `click`, `cryptography`, `greenlet`, `h11`, `httpcore`, `httptools`, `mako`, `markupsafe`, `psycopg-binary`, `pycparser`, `python-dotenv`, `uvloop`, `watchfiles`, `websockets`). No version conflicts.

- [ ] **Step 4: Confirm the existing suite is still green**

Run: `uv run pytest -q`
Expected: `88 passed` (same count as before this task — nothing here touches `app/` yet).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add FastAPI/SQLAlchemy/Alembic/PyJWT deps for Fase 3"
```

---

## Task 2: DB models

**Files:**
- Create: `app/db/__init__.py`
- Create: `app/db/models.py`
- Test: `tests/test_db_models.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `app.db.models.Base` (SQLAlchemy `DeclarativeBase`), `app.db.models.Scenario`, `app.db.models.Case`, `app.db.models.Run`, `app.db.models.MetricSet` — all later tasks import these.

- [ ] **Step 1: Write the failing test**

Create `tests/test_db_models.py`:

```python
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.models import Base, Case, MetricSet, Run, Scenario


def _memory_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def test_scenario_round_trip():
    engine = _memory_engine()
    with Session(engine) as session:
        scenario = Scenario(
            mode="arbitrage",
            penetration_level="low",
            units=[{"name": "B1", "mwh_nom": 10.0}],
        )
        session.add(scenario)
        session.commit()
        session.refresh(scenario)
        assert scenario.id
        fetched = session.get(Scenario, scenario.id)
        assert fetched.units == [{"name": "B1", "mwh_nom": 10.0}]


def test_case_run_metric_set_round_trip():
    engine = _memory_engine()
    with Session(engine) as session:
        case = Case(dispatch_date=date(2024, 4, 18), level="preideal")
        session.add(case)
        session.flush()

        run = Run(case_id=case.id, user_id="user-1", status="pending")
        session.add(run)
        session.flush()

        metric_set = MetricSet(run_id=run.id, mae=1.0, rmse=2.0)
        session.add(metric_set)
        session.commit()

        fetched_run = session.get(Run, run.id)
        assert fetched_run.case_id == case.id
        assert fetched_run.status == "pending"

        fetched_metrics = session.get(MetricSet, metric_set.id)
        assert fetched_metrics.run_id == run.id
        assert fetched_metrics.mae == 1.0


def test_case_scenario_id_defaults_to_none():
    engine = _memory_engine()
    with Session(engine) as session:
        case = Case(dispatch_date=date(2024, 4, 18), level="ideal")
        session.add(case)
        session.commit()
        session.refresh(case)
        assert case.scenario_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.db'`

- [ ] **Step 3: Write the implementation**

Create `app/db/__init__.py` (empty file).

Create `app/db/models.py`:

```python
import uuid
from datetime import date as date_
from datetime import datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, JSON, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _new_id() -> str:
    return uuid.uuid4().hex


class Scenario(Base):
    __tablename__ = "scenarios"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    penetration_level: Mapped[str] = mapped_column(String, nullable=False)
    units: Mapped[list] = mapped_column(JSON, nullable=False)
    created_by: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    dispatch_date: Mapped[date_] = mapped_column(Date, nullable=False)
    level: Mapped[str] = mapped_column(String, nullable=False)
    solver: Mapped[str] = mapped_column(String, default="cbc")
    compute_prices: Mapped[bool] = mapped_column(Boolean, default=True)
    scenario_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("scenarios.id"), nullable=True
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    case_id: Mapped[str] = mapped_column(String, ForeignKey("cases.id"), nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    out_dir: Mapped[str | None] = mapped_column(String, nullable=True)
    dispatch_path: Mapped[str | None] = mapped_column(String, nullable=True)
    price_path: Mapped[str | None] = mapped_column(String, nullable=True)
    bess_path: Mapped[str | None] = mapped_column(String, nullable=True)


class MetricSet(Base):
    __tablename__ = "metric_sets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    run_id: Mapped[str] = mapped_column(
        String, ForeignKey("runs.id"), unique=True, nullable=False
    )
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    bias: Mapped[float | None] = mapped_column(Float, nullable=True)
    wape: Mapped[float | None] = mapped_column(Float, nullable=True)
    smape: Mapped[float | None] = mapped_column(Float, nullable=True)
    r2: Mapped[float | None] = mapped_column(Float, nullable=True)
    bess_charge_mwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    bess_discharge_mwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    bess_avg_soc_mwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    bess_net_revenue: Mapped[float | None] = mapped_column(Float, nullable=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db_models.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add app/db/__init__.py app/db/models.py tests/test_db_models.py
git commit -m "feat: add SQLAlchemy models for scenarios/cases/runs/metric_sets"
```

---

## Task 3: DB session helper

**Files:**
- Create: `app/db/session.py`
- Test: `tests/test_db_session.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `app.db.session.get_engine(database_url: str | None = None) -> Engine` (reads `DATABASE_URL` env var if `database_url` is omitted), `app.db.session.get_sessionmaker(engine) -> sessionmaker`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_db_session.py`:

```python
from sqlalchemy.engine import Engine

from app.db.session import get_engine, get_sessionmaker


def test_get_engine_from_arg():
    engine = get_engine("sqlite:///:memory:")
    assert isinstance(engine, Engine)


def test_get_engine_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    engine = get_engine()
    assert isinstance(engine, Engine)


def test_get_sessionmaker_binds_engine():
    engine = get_engine("sqlite:///:memory:")
    session_factory = get_sessionmaker(engine)
    session = session_factory()
    assert session.bind is engine
    session.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.db.session'`

- [ ] **Step 3: Write the implementation**

Create `app/db/session.py`:

```python
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


def get_engine(database_url: str | None = None):
    url = database_url or os.environ["DATABASE_URL"]
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def get_sessionmaker(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db_session.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add app/db/session.py tests/test_db_session.py
git commit -m "feat: add DB engine/sessionmaker factory reading DATABASE_URL"
```

---

## Task 4: DB queries

**Files:**
- Create: `app/db/queries.py`
- Test: `tests/test_db_queries.py`

**Interfaces:**
- Consumes: `app.db.models.{Scenario,Case,Run,MetricSet}` (Task 2), `app.schemas.{BessScenario,RunResult}` (existing).
- Produces:
  - `create_scenario(session, scenario: BessScenario, created_by: str) -> Scenario`
  - `get_scenario(session, scenario_id: str) -> Scenario | None`
  - `create_case_and_run(session, *, dispatch_date: date, level: str, solver: str, compute_prices: bool, scenario_id: str | None, user_id: str) -> Run`
  - `get_run(session, run_id: str) -> Run | None`
  - `get_case(session, case_id: str) -> Case | None`
  - `list_runs_for_user(session, user_id: str) -> list[Run]`
  - `get_metric_set(session, run_id: str) -> MetricSet | None`
  - `finish_run_ok(session, run: Run, result: RunResult, out_dir: str) -> None`
  - `finish_run_failed(session, run: Run, error: str) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_db_queries.py`:

```python
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db import queries
from app.db.models import Base
from app.schemas import (
    BessMode,
    BessScenario,
    BessUnit,
    DispatchCase,
    DispatchLevel,
    RunResult,
)


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_create_scenario_persists_units_as_dicts():
    session = _session()
    scenario = BessScenario(
        mode=BessMode.generator,
        penetration_level="low",
        units=[
            BessUnit(
                name="B1",
                mwh_nom=10,
                hours_to_deplete=2,
                initial_soc=5,
                min_soc=0,
                max_soc=10,
                efficiency=0.9,
                discharge_bid=100.0,
            )
        ],
    )
    row = queries.create_scenario(session, scenario, created_by="user-1")
    assert row.id
    fetched = queries.get_scenario(session, row.id)
    assert fetched.units[0]["name"] == "B1"
    assert fetched.mode == "generator"


def test_create_case_and_run_defaults_to_pending():
    session = _session()
    run = queries.create_case_and_run(
        session,
        dispatch_date=date(2024, 4, 18),
        level="preideal",
        solver="cbc",
        compute_prices=True,
        scenario_id=None,
        user_id="user-1",
    )
    assert run.status == "pending"
    case = queries.get_case(session, run.case_id)
    assert case.level == "preideal"
    assert case.dispatch_date == date(2024, 4, 18)


def test_list_runs_for_user_orders_newest_first():
    session = _session()
    r1 = queries.create_case_and_run(
        session,
        dispatch_date=date(2024, 4, 18),
        level="preideal",
        solver="cbc",
        compute_prices=True,
        scenario_id=None,
        user_id="user-1",
    )
    r2 = queries.create_case_and_run(
        session,
        dispatch_date=date(2024, 4, 19),
        level="preideal",
        solver="cbc",
        compute_prices=True,
        scenario_id=None,
        user_id="user-1",
    )
    runs = queries.list_runs_for_user(session, "user-1")
    assert [r.id for r in runs] == [r2.id, r1.id]


def test_list_runs_for_user_excludes_other_users():
    session = _session()
    queries.create_case_and_run(
        session,
        dispatch_date=date(2024, 4, 18),
        level="preideal",
        solver="cbc",
        compute_prices=True,
        scenario_id=None,
        user_id="user-1",
    )
    queries.create_case_and_run(
        session,
        dispatch_date=date(2024, 4, 18),
        level="preideal",
        solver="cbc",
        compute_prices=True,
        scenario_id=None,
        user_id="user-2",
    )
    runs = queries.list_runs_for_user(session, "user-1")
    assert len(runs) == 1


def test_finish_run_ok_writes_metric_set():
    session = _session()
    run = queries.create_case_and_run(
        session,
        dispatch_date=date(2024, 4, 18),
        level="preideal",
        solver="cbc",
        compute_prices=True,
        scenario_id=None,
        user_id="user-1",
    )
    case = queries.get_case(session, run.case_id)
    dispatch_case = DispatchCase(dispatch_date=case.dispatch_date, level=DispatchLevel.preideal)
    result = RunResult(
        case=dispatch_case,
        ok=True,
        dispatch_path="data/results/x/d.csv",
        price_path="data/results/x/p.csv",
        metrics={"mae": 1.0, "rmse": 2.0, "bias": 0.1, "wape": 0.2, "smape": 0.3, "r2": 0.9},
    )
    queries.finish_run_ok(session, run, result, out_dir="data/results/x")

    updated = queries.get_run(session, run.id)
    assert updated.status == "done"
    assert updated.dispatch_path == "data/results/x/d.csv"

    metric_set = queries.get_metric_set(session, run.id)
    assert metric_set.mae == 1.0
    assert metric_set.rmse == 2.0


def test_finish_run_ok_without_metrics_skips_metric_set():
    session = _session()
    run = queries.create_case_and_run(
        session,
        dispatch_date=date(2024, 4, 18),
        level="preideal",
        solver="cbc",
        compute_prices=True,
        scenario_id=None,
        user_id="user-1",
    )
    case = queries.get_case(session, run.case_id)
    dispatch_case = DispatchCase(dispatch_date=case.dispatch_date, level=DispatchLevel.preideal)
    result = RunResult(case=dispatch_case, ok=True, dispatch_path="d.csv", price_path="p.csv")
    queries.finish_run_ok(session, run, result, out_dir="data/results/x")

    assert queries.get_metric_set(session, run.id) is None


def test_finish_run_failed_sets_error():
    session = _session()
    run = queries.create_case_and_run(
        session,
        dispatch_date=date(2024, 4, 18),
        level="preideal",
        solver="cbc",
        compute_prices=True,
        scenario_id=None,
        user_id="user-1",
    )
    queries.finish_run_failed(session, run, "boom")
    updated = queries.get_run(session, run.id)
    assert updated.status == "failed"
    assert updated.error == "boom"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_queries.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.db.queries'`

- [ ] **Step 3: Write the implementation**

Create `app/db/queries.py`:

```python
from datetime import date as date_
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Case, MetricSet, Run, Scenario
from app.schemas import BessScenario, RunResult


def create_scenario(session: Session, scenario: BessScenario, created_by: str) -> Scenario:
    row = Scenario(
        mode=scenario.mode.value,
        penetration_level=scenario.penetration_level,
        units=[u.model_dump() for u in scenario.units],
        created_by=created_by,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_scenario(session: Session, scenario_id: str) -> Scenario | None:
    return session.get(Scenario, scenario_id)


def create_case_and_run(
    session: Session,
    *,
    dispatch_date: date_,
    level: str,
    solver: str,
    compute_prices: bool,
    scenario_id: str | None,
    user_id: str,
) -> Run:
    case = Case(
        dispatch_date=dispatch_date,
        level=level,
        solver=solver,
        compute_prices=compute_prices,
        scenario_id=scenario_id,
    )
    session.add(case)
    session.flush()  # populate case.id before Run references it

    run = Run(case_id=case.id, user_id=user_id, status="pending")
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def get_run(session: Session, run_id: str) -> Run | None:
    return session.get(Run, run_id)


def get_case(session: Session, case_id: str) -> Case | None:
    return session.get(Case, case_id)


def list_runs_for_user(session: Session, user_id: str) -> list[Run]:
    stmt = select(Run).where(Run.user_id == user_id).order_by(Run.created_at.desc())
    return list(session.scalars(stmt))


def get_metric_set(session: Session, run_id: str) -> MetricSet | None:
    stmt = select(MetricSet).where(MetricSet.run_id == run_id)
    return session.scalars(stmt).first()


def finish_run_ok(session: Session, run: Run, result: RunResult, out_dir: str) -> None:
    run.status = "done"
    run.finished_at = datetime.now(timezone.utc)
    run.out_dir = out_dir
    run.dispatch_path = result.dispatch_path
    run.price_path = result.price_path
    run.bess_path = result.bess_path
    session.add(run)

    if result.metrics is not None or result.bess_summary is not None:
        metrics = result.metrics or {}
        bess = result.bess_summary or {}
        session.add(
            MetricSet(
                run_id=run.id,
                rmse=metrics.get("rmse"),
                mae=metrics.get("mae"),
                bias=metrics.get("bias"),
                wape=metrics.get("wape"),
                smape=metrics.get("smape"),
                r2=metrics.get("r2"),
                bess_charge_mwh=bess.get("bess_charge_mwh"),
                bess_discharge_mwh=bess.get("bess_discharge_mwh"),
                bess_avg_soc_mwh=bess.get("bess_avg_soc_mwh"),
                bess_net_revenue=bess.get("bess_net_revenue"),
            )
        )
    session.commit()


def finish_run_failed(session: Session, run: Run, error: str) -> None:
    run.status = "failed"
    run.finished_at = datetime.now(timezone.utc)
    run.error = error
    session.add(run)
    session.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db_queries.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add app/db/queries.py tests/test_db_queries.py
git commit -m "feat: add DB query helpers for scenarios/cases/runs/metric_sets"
```

---

## Task 5: DB claim (Postgres-only locking, isolated)

**Files:**
- Create: `app/db/claim.py`
- Test: `tests/test_db_claim.py`

**Interfaces:**
- Consumes: `app.db.models.Run` (Task 2), `app.db.queries.create_case_and_run` (Task 4, test-only).
- Produces: `claim_next_pending_run(session) -> Run | None` — atomically claims the oldest `pending` run and marks it `running`. Later consumed by `services/worker/main.py` (Task 12).

- [ ] **Step 1: Write the failing test**

Create `tests/test_db_claim.py`:

```python
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db import queries
from app.db.claim import claim_next_pending_run
from app.db.models import Base


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _make_pending_run(session, user_id="user-1"):
    return queries.create_case_and_run(
        session,
        dispatch_date=date(2024, 4, 18),
        level="preideal",
        solver="cbc",
        compute_prices=True,
        scenario_id=None,
        user_id=user_id,
    )


def test_claim_returns_none_when_no_pending_runs():
    session = _session()
    assert claim_next_pending_run(session) is None


def test_claim_marks_run_running_and_sets_started_at():
    session = _session()
    run = _make_pending_run(session)

    claimed = claim_next_pending_run(session)

    assert claimed.id == run.id
    assert claimed.status == "running"
    assert claimed.started_at is not None


def test_claim_does_not_reclaim_an_already_running_run():
    session = _session()
    _make_pending_run(session)

    first = claim_next_pending_run(session)
    second = claim_next_pending_run(session)

    assert first is not None
    assert second is None


def test_claim_picks_oldest_pending_run_first():
    session = _session()
    older = _make_pending_run(session)
    _make_pending_run(session)

    claimed = claim_next_pending_run(session)

    assert claimed.id == older.id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_claim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.db.claim'`

- [ ] **Step 3: Write the implementation**

Create `app/db/claim.py`:

```python
"""Postgres-only locking lives here, and only here: `FOR UPDATE SKIP LOCKED`
is what lets more than one worker replica claim rows safely without
stepping on each other. On SQLite (tests, single process) the clause is
simply not added -- SQLite serializes writers on its own, so there's
nothing to skip-lock."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Run


def claim_next_pending_run(session: Session) -> Run | None:
    stmt = select(Run).where(Run.status == "pending").order_by(Run.created_at)
    if session.bind.dialect.name == "postgresql":
        stmt = stmt.with_for_update(skip_locked=True)
    run = session.scalars(stmt).first()
    if run is None:
        return None
    run.status = "running"
    run.started_at = datetime.now(timezone.utc)
    session.commit()
    return run
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db_claim.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add app/db/claim.py tests/test_db_claim.py
git commit -m "feat: add worker claim (FOR UPDATE SKIP LOCKED, Postgres-only)"
```

---

## Task 6: Fix the `evaluate`/`compare` metrics-summary gap

**Files:**
- Modify: `app/pipeline/evaluate.py`
- Test: `tests/test_evaluate.py` (extend existing file)

**Interfaces:**
- Consumes: nothing new (uses existing `app.storage.get_storage`, `app.utils.metrics.price_metrics`).
- Produces: `evaluate_saved_run(...)` behavior change only — same signature and return value as before, now also upserts `metrics-summary.csv`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluate.py`:

```python
def test_evaluate_saved_run_upserts_metrics_summary_csv(tmp_path):
    price_df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-04-18", periods=24, freq="1h"),
            "ideal_marginal_price": [float(i) for i in range(24)],
        }
    )
    price_df.to_csv(tmp_path / "marginal_price-2024-04-18-preideal.csv", index=False)

    actuals_dir = tmp_path / "preideal_price"
    actuals_dir.mkdir()
    row = "MPO," + ",".join(str(float(i)) for i in range(24))
    (actuals_dir / "2024-04-18.txt").write_text(row + "\n")

    # pre-existing row for a different date must survive the upsert
    pd.DataFrame(
        [{"date": "2024-04-17", "type": "preideal", "scenario": "baseline", "mae": 99.0}]
    ).to_csv(tmp_path / "metrics-summary.csv", index=False)

    evaluate_saved_run(
        date(2024, 4, 18), DispatchLevel.preideal, out=str(tmp_path), data_dir=str(tmp_path)
    )

    summary = pd.read_csv(tmp_path / "metrics-summary.csv")
    assert len(summary) == 2
    new_row = summary[summary["date"] == "2024-04-18"].iloc[0]
    assert new_row["type"] == "preideal"
    assert new_row["scenario"] == "baseline"
    assert new_row["mae"] == 0.0
    old_row = summary[summary["date"] == "2024-04-17"].iloc[0]
    assert old_row["mae"] == 99.0


def test_evaluate_saved_run_replaces_stale_row_for_same_key(tmp_path):
    price_df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-04-18", periods=24, freq="1h"),
            "ideal_marginal_price": [float(i) for i in range(24)],
        }
    )
    price_df.to_csv(tmp_path / "marginal_price-2024-04-18-preideal.csv", index=False)

    actuals_dir = tmp_path / "preideal_price"
    actuals_dir.mkdir()
    row = "MPO," + ",".join(str(float(i)) for i in range(24))
    (actuals_dir / "2024-04-18.txt").write_text(row + "\n")

    # stale row for the SAME key must be replaced, not duplicated
    pd.DataFrame(
        [{"date": "2024-04-18", "type": "preideal", "scenario": "baseline", "mae": 999.0}]
    ).to_csv(tmp_path / "metrics-summary.csv", index=False)

    evaluate_saved_run(
        date(2024, 4, 18), DispatchLevel.preideal, out=str(tmp_path), data_dir=str(tmp_path)
    )

    summary = pd.read_csv(tmp_path / "metrics-summary.csv")
    assert len(summary) == 1
    assert summary.iloc[0]["mae"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluate.py -v -k upserts_metrics_summary or replaces_stale_row`
Expected: FAIL — `metrics-summary.csv` is never written/updated by `evaluate_saved_run` today, so `pd.read_csv(tmp_path / "metrics-summary.csv")` after the call reads back the untouched pre-seeded file (1 row, `mae == 99.0` / `999.0`), not the upserted one.

- [ ] **Step 3: Write the implementation**

In `app/pipeline/evaluate.py`, add a helper and call it from `evaluate_saved_run`:

```python
def _upsert_metrics_summary(
    storage, dispatch_date: date, level: DispatchLevel, metrics: dict[str, float]
) -> None:
    """Upsert this run's row into metrics-summary.csv, keyed on
    (date, type, scenario). Without this, a `run --no-eval` -> `evaluate`
    -> `compare` flow never sees post-hoc metrics: `run_many` only writes
    metrics-summary.csv once, at the end of a batch, and evaluate_saved_run
    previously only wrote the per-run metrics-{date}-{level}.csv file."""
    row = {"date": str(dispatch_date), "type": level.value, "scenario": "baseline", **metrics}
    summary_path = "metrics-summary.csv"
    if storage.exists(summary_path):
        with storage.open(summary_path) as f:
            summary = pd.read_csv(f)
        stale = (
            (summary["date"] == row["date"])
            & (summary["type"] == row["type"])
            & (summary["scenario"] == row["scenario"])
        )
        summary = pd.concat([summary[~stale], pd.DataFrame([row])], ignore_index=True)
    else:
        summary = pd.DataFrame([row])
    with storage.open(summary_path, "w") as f:
        summary.to_csv(f, index=False)
```

Then, in `evaluate_saved_run`, right after the existing block that writes `metrics-{dispatch_date}-{t}.csv`, add:

```python
    _upsert_metrics_summary(storage, dispatch_date, level, metrics)
```

(This goes after the `metrics_name`/`storage.open(metrics_name, "w")` block, before `return metrics`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: all tests in the file pass (existing 4 + 2 new = 6).

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/evaluate.py tests/test_evaluate.py
git commit -m "fix: evaluate_saved_run upserts metrics-summary.csv (compare gap)"
```

---

## Task 7: Alembic migrations

**Files:**
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/versions/0001_initial.py`
- Test: `tests/test_db_migrations.py`

**Interfaces:**
- Consumes: `app.db.models.Base` (Task 2).
- Produces: `alembic upgrade head` creates `scenarios`/`cases`/`runs`/`metric_sets` tables against any `sqlalchemy.url` (SQLite in tests, Supabase Postgres in dev/prod).

- [ ] **Step 1: Write the failing test**

Create `tests/test_db_migrations.py`:

```python
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def test_alembic_upgrade_head_creates_all_tables(tmp_path):
    db_path = tmp_path / "migration_smoke.db"
    database_url = f"sqlite:///{db_path}"

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")

    engine = create_engine(database_url)
    tables = set(inspect(engine).get_table_names())
    assert {"scenarios", "cases", "runs", "metric_sets"}.issubset(tables)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_migrations.py -v`
Expected: FAIL — `alembic.ini` doesn't exist yet (`FileNotFoundError` or Alembic config error).

- [ ] **Step 3: Write the implementation**

Create `alembic.ini` at the repo root:

```ini
[alembic]
script_location = alembic
sqlalchemy.url =

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

Create `alembic/env.py`:

```python
import sys
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.models import Base  # noqa: E402

config = context.config
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

Create `alembic/versions/0001_initial.py`:

```python
"""initial schema: scenarios, cases, runs, metric_sets

Revision ID: 0001
Revises:
Create Date: 2026-08-05
"""

import sqlalchemy as sa

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scenarios",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("penetration_level", sa.String(), nullable=False),
        sa.Column("units", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "cases",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("dispatch_date", sa.Date(), nullable=False),
        sa.Column("level", sa.String(), nullable=False),
        sa.Column("solver", sa.String(), nullable=False),
        sa.Column("compute_prices", sa.Boolean(), nullable=False),
        sa.Column("scenario_id", sa.String(), sa.ForeignKey("scenarios.id"), nullable=True),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("case_id", sa.String(), sa.ForeignKey("cases.id"), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("out_dir", sa.String(), nullable=True),
        sa.Column("dispatch_path", sa.String(), nullable=True),
        sa.Column("price_path", sa.String(), nullable=True),
        sa.Column("bess_path", sa.String(), nullable=True),
    )
    op.create_table(
        "metric_sets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("runs.id"), nullable=False, unique=True),
        sa.Column("rmse", sa.Float(), nullable=True),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("bias", sa.Float(), nullable=True),
        sa.Column("wape", sa.Float(), nullable=True),
        sa.Column("smape", sa.Float(), nullable=True),
        sa.Column("r2", sa.Float(), nullable=True),
        sa.Column("bess_charge_mwh", sa.Float(), nullable=True),
        sa.Column("bess_discharge_mwh", sa.Float(), nullable=True),
        sa.Column("bess_avg_soc_mwh", sa.Float(), nullable=True),
        sa.Column("bess_net_revenue", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("metric_sets")
    op.drop_table("runs")
    op.drop_table("cases")
    op.drop_table("scenarios")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db_migrations.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add alembic.ini alembic/env.py alembic/versions/0001_initial.py tests/test_db_migrations.py
git commit -m "feat: add Alembic migrations for the Fase 3 schema"
```

---

## Task 8: API JWT verification

**Files:**
- Create: `services/__init__.py`
- Create: `services/api/__init__.py`
- Create: `services/api/auth.py`
- Test: `tests/test_api_auth.py`

**Interfaces:**
- Consumes: nothing new (uses `pyjwt`, `cryptography` from Task 1).
- Produces: `decode_bearer_token(authorization: str, jwk_client) -> dict` (pure, injectable client — testable without network), `get_current_user_id(authorization: str = Header(...)) -> str` (FastAPI dependency, real entrypoint used by `services/api/main.py` in Task 9, overridden in API tests).

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_auth.py`:

```python
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from services.api.auth import decode_bearer_token


class _FakeSigningKey:
    def __init__(self, key):
        self.key = key


class _FakeJWKClient:
    def __init__(self, public_key):
        self._public_key = public_key

    def get_signing_key_from_jwt(self, token):
        return _FakeSigningKey(self._public_key)


@pytest.fixture
def rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _token(private_pem, **claims):
    payload = {"sub": "user-1", "aud": "authenticated", "exp": int(time.time()) + 3600, **claims}
    return jwt.encode(payload, private_pem, algorithm="RS256")


def test_decode_bearer_token_returns_payload_for_valid_token(rsa_keypair):
    private_pem, public_pem = rsa_keypair
    token = _token(private_pem)
    payload = decode_bearer_token(f"Bearer {token}", _FakeJWKClient(public_pem))
    assert payload["sub"] == "user-1"


def test_decode_bearer_token_rejects_missing_bearer_prefix(rsa_keypair):
    _, public_pem = rsa_keypair
    with pytest.raises(HTTPException) as exc_info:
        decode_bearer_token("not-a-bearer-token", _FakeJWKClient(public_pem))
    assert exc_info.value.status_code == 401


def test_decode_bearer_token_rejects_expired_token(rsa_keypair):
    private_pem, public_pem = rsa_keypair
    token = _token(private_pem, exp=int(time.time()) - 10)
    with pytest.raises(HTTPException) as exc_info:
        decode_bearer_token(f"Bearer {token}", _FakeJWKClient(public_pem))
    assert exc_info.value.status_code == 401


def test_decode_bearer_token_rejects_wrong_audience(rsa_keypair):
    private_pem, public_pem = rsa_keypair
    token = _token(private_pem, aud="something-else")
    with pytest.raises(HTTPException) as exc_info:
        decode_bearer_token(f"Bearer {token}", _FakeJWKClient(public_pem))
    assert exc_info.value.status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_auth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services'`

- [ ] **Step 3: Write the implementation**

Create `services/__init__.py` (empty file).
Create `services/api/__init__.py` (empty file).

Create `services/api/auth.py`:

```python
import os

import jwt
from fastapi import Header, HTTPException
from jwt import PyJWKClient

_jwk_client: PyJWKClient | None = None


def _get_jwk_client() -> PyJWKClient:
    global _jwk_client
    if _jwk_client is None:
        _jwk_client = PyJWKClient(os.environ["SUPABASE_JWKS_URL"])
    return _jwk_client


def decode_bearer_token(authorization: str, jwk_client: PyJWKClient) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ")
    try:
        signing_key = jwk_client.get_signing_key_from_jwt(token)
        return jwt.decode(token, signing_key.key, algorithms=["RS256"], audience="authenticated")
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"invalid token: {e}") from e


def get_current_user_id(authorization: str = Header(...)) -> str:
    payload = decode_bearer_token(authorization, _get_jwk_client())
    return payload["sub"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api_auth.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add services/__init__.py services/api/__init__.py services/api/auth.py tests/test_api_auth.py
git commit -m "feat: add Supabase JWT verification for the API"
```

---

## Task 9: API app skeleton + `POST /scenarios` + shared test fixture

**Files:**
- Create: `services/api/main.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_api_scenarios.py`

**Interfaces:**
- Consumes: `app.db.models.Base` (Task 2), `app.db.queries.create_scenario` (Task 4), `services.api.auth.get_current_user_id` (Task 8), `app.schemas.BessScenario` (existing).
- Produces: `services.api.main.app` (the FastAPI instance — Tasks 10/11/13 add routes to it and it's what `uvicorn` serves in prod), `services.api.main.get_session` (dependency, overridden in tests). Test fixture `api_client` in `tests/conftest.py`, available to every subsequent `test_api_*.py` file — yields a `TestClient` with an isolated in-memory SQLite DB and a fixed `user_id="user-1"`, exposes the raw session factory as `client.SessionLocal` for tests that need to seed/inspect DB state directly.

- [ ] **Step 1: Write the failing test**

Add to `tests/conftest.py` (after the existing `sys.path` line):

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient

    from app.db.models import Base
    from services.api.auth import get_current_user_id
    from services.api.main import app, get_session

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    test_session_factory = sessionmaker(bind=engine)

    def _override_session():
        session = test_session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    client = TestClient(app)
    client.SessionLocal = test_session_factory
    yield client
    app.dependency_overrides.clear()
```

Create `tests/test_api_scenarios.py`:

```python
def test_create_scenario_returns_id(api_client):
    body = {
        "mode": "generator",
        "penetration_level": "low",
        "units": [
            {
                "name": "B1",
                "mwh_nom": 10,
                "hours_to_deplete": 2,
                "initial_soc": 5,
                "min_soc": 0,
                "max_soc": 10,
                "efficiency": 0.9,
                "discharge_bid": 100.0,
            }
        ],
    }
    resp = api_client.post("/scenarios", json=body)
    assert resp.status_code == 200
    assert "id" in resp.json()


def test_create_scenario_rejects_arbitrage_without_charge_bid(api_client):
    body = {
        "mode": "arbitrage",
        "penetration_level": "low",
        "units": [
            {
                "name": "B1",
                "mwh_nom": 10,
                "hours_to_deplete": 2,
                "initial_soc": 5,
                "min_soc": 0,
                "max_soc": 10,
                "efficiency": 0.9,
                "discharge_bid": 100.0,
            }
        ],
    }
    resp = api_client.post("/scenarios", json=body)
    assert resp.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_scenarios.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.api.main'`

- [ ] **Step 3: Write the implementation**

Create `services/api/main.py`:

```python
from fastapi import Depends, FastAPI

from app.db import queries
from app.db.session import get_engine, get_sessionmaker
from app.schemas import BessScenario
from services.api.auth import get_current_user_id

app = FastAPI(title="despacho-udea API")

_engine = None
_session_local = None


def get_session():
    global _engine, _session_local
    if _session_local is None:
        _engine = get_engine()
        _session_local = get_sessionmaker(_engine)
    session = _session_local()
    try:
        yield session
    finally:
        session.close()


@app.post("/scenarios")
def create_scenario(
    scenario: BessScenario,
    user_id: str = Depends(get_current_user_id),
    session=Depends(get_session),
):
    row = queries.create_scenario(session, scenario, created_by=user_id)
    return {"id": row.id}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api_scenarios.py -v`
Expected: `2 passed`

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: all prior tests still pass, plus the new ones.

- [ ] **Step 6: Commit**

```bash
git add services/api/main.py tests/conftest.py tests/test_api_scenarios.py
git commit -m "feat: add FastAPI app skeleton and POST /scenarios"
```

---

## Task 10: `POST /runs`, `GET /runs`, `GET /runs/{id}`

**Files:**
- Modify: `services/api/main.py`
- Test: `tests/test_api_runs.py`

**Interfaces:**
- Consumes: `app.db.queries.{create_case_and_run,get_scenario,get_run,list_runs_for_user,get_metric_set}` (Task 4), `app.schemas.DispatchLevel` (existing), `services.api.main.app`/`get_session` (Task 9).
- Produces: `services.api.main._get_owned_run(session, run_id, user_id) -> Run` (module-private helper, raises 404 — reused by Task 11's artifact routes).

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_runs.py`:

```python
def test_create_run_returns_pending_status(api_client):
    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    assert "run_id" in body


def test_create_run_rejects_unknown_scenario_id(api_client):
    resp = api_client.post(
        "/runs",
        json={"dispatch_date": "2024-04-18", "level": "preideal", "scenario_id": "missing"},
    )
    assert resp.status_code == 404


def test_get_run_returns_404_for_unknown_id(api_client):
    resp = api_client.get("/runs/does-not-exist")
    assert resp.status_code == 404


def test_get_run_returns_status_and_null_metrics_before_worker_runs(api_client):
    create_resp = api_client.post(
        "/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"}
    )
    run_id = create_resp.json()["run_id"]

    resp = api_client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["status"] == "pending"
    assert body["metrics"] is None


def test_list_runs_returns_created_runs(api_client):
    api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    api_client.post("/runs", json={"dispatch_date": "2024-04-19", "level": "ideal"})
    resp = api_client.get("/runs")
    assert resp.status_code == 200
    assert len(resp.json()) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_runs.py -v`
Expected: FAIL with `404 Not Found` for `POST /runs` (route doesn't exist yet — FastAPI's default 404 for an unregistered path).

- [ ] **Step 3: Write the implementation**

In `services/api/main.py`, add imports and new routes:

```python
from datetime import date

from fastapi import HTTPException
from pydantic import BaseModel

from app.schemas import DispatchLevel
```

(merge with the existing `from fastapi import Depends, FastAPI` line — final import block should read `from fastapi import Depends, FastAPI, HTTPException`)

Then append to `services/api/main.py`:

```python
class RunCreateRequest(BaseModel):
    dispatch_date: date
    level: DispatchLevel
    solver: str = "cbc"
    compute_prices: bool = True
    scenario_id: str | None = None


def _run_summary(run) -> dict:
    return {
        "run_id": run.id,
        "status": run.status,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "error": run.error,
    }


def _get_owned_run(session, run_id: str, user_id: str):
    run = queries.get_run(session, run_id)
    if run is None or run.user_id != user_id:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.post("/runs")
def create_run(
    body: RunCreateRequest,
    user_id: str = Depends(get_current_user_id),
    session=Depends(get_session),
):
    if body.scenario_id is not None and queries.get_scenario(session, body.scenario_id) is None:
        raise HTTPException(status_code=404, detail="scenario not found")
    run = queries.create_case_and_run(
        session,
        dispatch_date=body.dispatch_date,
        level=body.level.value,
        solver=body.solver,
        compute_prices=body.compute_prices,
        scenario_id=body.scenario_id,
        user_id=user_id,
    )
    return {"run_id": run.id, "status": run.status}


@app.get("/runs")
def list_runs(user_id: str = Depends(get_current_user_id), session=Depends(get_session)):
    runs = queries.list_runs_for_user(session, user_id)
    return [_run_summary(r) for r in runs]


@app.get("/runs/{run_id}")
def get_run_detail(
    run_id: str, user_id: str = Depends(get_current_user_id), session=Depends(get_session)
):
    run = _get_owned_run(session, run_id, user_id)
    metric_set = queries.get_metric_set(session, run.id)
    out = _run_summary(run)
    out["metrics"] = (
        {
            "rmse": metric_set.rmse,
            "mae": metric_set.mae,
            "bias": metric_set.bias,
            "wape": metric_set.wape,
            "smape": metric_set.smape,
            "r2": metric_set.r2,
        }
        if metric_set
        else None
    )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api_runs.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add services/api/main.py tests/test_api_runs.py
git commit -m "feat: add POST /runs, GET /runs, GET /runs/{id}"
```

---

## Task 11: `GET /runs/{id}/{artifact}` and `GET /runs/{id}/download/{artifact}`

**Files:**
- Modify: `services/api/main.py`
- Test: `tests/test_api_results.py`

**Interfaces:**
- Consumes: `services.api.main._get_owned_run` (Task 10), `app.db.queries.{finish_run_ok,get_run}` (Task 4), `app.storage.get_storage` (existing), `app.schemas.{DispatchCase,DispatchLevel,RunResult}` (existing).
- Produces: nothing new consumed by later tasks — this is the last API route task.

`Run.dispatch_path`/`price_path`/`bess_path` are stored as full paths relative to the process's working directory (e.g. `"data/results/<run_id>/marginal_price-2024-04-18-preideal.csv"`, exactly what `save_results` returns) — so artifact routes read them via `get_storage(".")`, not `get_storage(run.out_dir)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_results.py`:

```python
from datetime import date

import pandas as pd

from app.db import queries
from app.schemas import DispatchCase, DispatchLevel, RunResult


def _seed_done_run_with_dispatch_csv(api_client, tmp_path):
    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    run_id = resp.json()["run_id"]

    out_dir = tmp_path / "results" / run_id
    out_dir.mkdir(parents=True)
    dispatch_csv = out_dir / "dispatch_by_gen-2024-04-18-preideal.csv"
    pd.DataFrame(
        [{"generador": "TERMO1", "datetime": "2024-04-18 00:00", "dispatch": 300.0}]
    ).to_csv(dispatch_csv, index=False)

    session = api_client.SessionLocal()
    run = queries.get_run(session, run_id)
    result = RunResult(
        case=DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal),
        ok=True,
        dispatch_path=str(dispatch_csv),
    )
    queries.finish_run_ok(session, run, result, out_dir=str(out_dir))
    session.close()

    return run_id


def test_get_run_dispatch_returns_json_rows(api_client, tmp_path):
    run_id = _seed_done_run_with_dispatch_csv(api_client, tmp_path)

    resp = api_client.get(f"/runs/{run_id}/dispatch")
    assert resp.status_code == 200
    rows = resp.json()
    assert rows[0]["generador"] == "TERMO1"
    assert rows[0]["dispatch"] == 300.0


def test_download_run_dispatch_returns_csv_file(api_client, tmp_path):
    run_id = _seed_done_run_with_dispatch_csv(api_client, tmp_path)

    resp = api_client.get(f"/runs/{run_id}/download/dispatch")
    assert resp.status_code == 200
    assert "TERMO1" in resp.text


def test_get_run_artifact_404_when_run_has_no_artifact_yet(api_client):
    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    run_id = resp.json()["run_id"]

    resp = api_client.get(f"/runs/{run_id}/dispatch")
    assert resp.status_code == 404


def test_get_run_artifact_404_for_unknown_artifact_name(api_client, tmp_path):
    run_id = _seed_done_run_with_dispatch_csv(api_client, tmp_path)

    resp = api_client.get(f"/runs/{run_id}/not-a-real-artifact")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_results.py -v`
Expected: FAIL with `404 Not Found` for `GET /runs/{id}/dispatch` (route doesn't exist yet).

- [ ] **Step 3: Write the implementation**

In `services/api/main.py`, add imports:

```python
import pandas as pd
from fastapi.responses import FileResponse

from app.storage import get_storage
```

Then append:

```python
_ARTIFACT_PATHS = {
    "dispatch": "dispatch_path",
    "prices": "price_path",
    "bess": "bess_path",
}


def _artifact_path(run, artifact: str) -> str:
    if artifact not in _ARTIFACT_PATHS:
        raise HTTPException(status_code=404, detail="unknown artifact")
    path = getattr(run, _ARTIFACT_PATHS[artifact])
    if path is None:
        raise HTTPException(status_code=404, detail=f"run has no {artifact} artifact yet")
    return path


@app.get("/runs/{run_id}/{artifact}")
def get_run_artifact(
    run_id: str,
    artifact: str,
    user_id: str = Depends(get_current_user_id),
    session=Depends(get_session),
):
    run = _get_owned_run(session, run_id, user_id)
    path = _artifact_path(run, artifact)
    storage = get_storage(".")
    if not storage.exists(path):
        raise HTTPException(status_code=404, detail="artifact file missing on disk")
    with storage.open(path) as f:
        df = pd.read_csv(f)
    return df.to_dict(orient="records")


@app.get("/runs/{run_id}/download/{artifact}")
def download_run_artifact(
    run_id: str,
    artifact: str,
    user_id: str = Depends(get_current_user_id),
    session=Depends(get_session),
):
    run = _get_owned_run(session, run_id, user_id)
    path = _artifact_path(run, artifact)
    return FileResponse(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api_results.py -v`
Expected: `4 passed`

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass (prior 88 + all new ones from Tasks 2-11).

- [ ] **Step 6: Commit**

```bash
git add services/api/main.py tests/test_api_results.py
git commit -m "feat: add GET /runs/{id}/{artifact} and download endpoints"
```

---

## Task 12: Worker

**Files:**
- Create: `services/worker/__init__.py`
- Create: `services/worker/main.py`
- Test: `tests/test_worker_main.py`

**Interfaces:**
- Consumes: `app.db.claim.claim_next_pending_run` (Task 5), `app.db.queries.{get_case,get_scenario,finish_run_ok,finish_run_failed}` (Task 4), `app.pipeline.runner.run_case` (existing, unmodified), `app.schemas.{BessScenario,DispatchCase,DispatchLevel}` (existing).
- Produces: `process_once(session, *, data_dir="data", results_root="data/results") -> bool` (claims and runs one pending run; returns `False` if there was nothing to claim), `main()` (the poll-forever entrypoint used by the Docker `CMD`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_worker_main.py`:

```python
from datetime import date
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db import queries
from app.db.models import Base
from app.schemas import DispatchCase, DispatchLevel, RunResult
from services.worker.main import process_once

DD = str(Path(__file__).parent / "fixtures" / "xm_smoke")
FECHA = date(2024, 4, 18)


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_process_once_returns_false_when_no_pending_runs():
    session = _session()
    assert process_once(session, data_dir=DD, results_root="data/results") is False


def test_process_once_solves_pending_run_end_to_end(tmp_path, monkeypatch):
    def _no_network(*a, **kw):
        raise AssertionError(f"unexpected network call: {a} {kw}")

    monkeypatch.setattr("app.data.download.requests.get", _no_network)

    session = _session()
    run = queries.create_case_and_run(
        session,
        dispatch_date=FECHA,
        level="preideal",
        solver="cbc",
        compute_prices=True,
        scenario_id=None,
        user_id="user-1",
    )

    results_root = str(tmp_path / "results")
    processed = process_once(session, data_dir=DD, results_root=results_root)
    assert processed is True

    updated = queries.get_run(session, run.id)
    assert updated.status == "done", updated.error
    assert updated.price_path is not None
    assert Path(updated.price_path).exists()

    # xm_smoke fixture has no preideal_price actuals -> evaluate is skipped,
    # matching tests/test_xm_smoke_run.py's own assertion
    assert queries.get_metric_set(session, run.id) is None


def test_process_once_marks_run_failed_when_run_case_reports_failure(tmp_path, monkeypatch):
    session = _session()
    run = queries.create_case_and_run(
        session,
        dispatch_date=FECHA,
        level="preideal",
        solver="cbc",
        compute_prices=True,
        scenario_id=None,
        user_id="user-1",
    )

    case = DispatchCase(dispatch_date=FECHA, level=DispatchLevel.preideal, solver="cbc")
    fake_result = RunResult(case=case, ok=False, error="boom")
    monkeypatch.setattr("services.worker.main.run_case", lambda *a, **kw: fake_result)

    processed = process_once(session, data_dir=DD, results_root=str(tmp_path / "results"))
    assert processed is True

    updated = queries.get_run(session, run.id)
    assert updated.status == "failed"
    assert updated.error == "boom"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_worker_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.worker'`

- [ ] **Step 3: Write the implementation**

Create `services/worker/__init__.py` (empty file).

Create `services/worker/main.py`:

```python
import time

from sqlalchemy.orm import Session

from app.db import queries
from app.db.claim import claim_next_pending_run
from app.db.session import get_engine, get_sessionmaker
from app.pipeline.runner import run_case
from app.schemas import BessScenario, DispatchCase, DispatchLevel

POLL_INTERVAL_SECONDS = 5


def _build_case(session: Session, case_row) -> DispatchCase:
    scenario = None
    if case_row.scenario_id is not None:
        scenario_row = queries.get_scenario(session, case_row.scenario_id)
        scenario = BessScenario(
            mode=scenario_row.mode,
            penetration_level=scenario_row.penetration_level,
            units=scenario_row.units,
        )
    return DispatchCase(
        dispatch_date=case_row.dispatch_date,
        level=DispatchLevel(case_row.level),
        solver=case_row.solver,
        compute_prices=case_row.compute_prices,
        bess_scenario=scenario,
    )


def process_once(
    session: Session, *, data_dir: str = "data", results_root: str = "data/results"
) -> bool:
    run = claim_next_pending_run(session)
    if run is None:
        return False

    case_row = queries.get_case(session, run.case_id)
    case = _build_case(session, case_row)
    out_dir = f"{results_root}/{run.id}"

    result = run_case(case, evaluate=True, out=out_dir, data_dir=data_dir)

    if result.ok:
        queries.finish_run_ok(session, run, result, out_dir=out_dir)
    else:
        queries.finish_run_failed(session, run, result.error or "unknown error")
    return True


def main() -> None:
    engine = get_engine()
    session_factory = get_sessionmaker(engine)
    while True:
        with session_factory() as session:
            processed = process_once(session)
        if not processed:
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_worker_main.py -v`
Expected: `3 passed` (the end-to-end one takes longer — it runs a real `cbc` solve, same as `tests/test_xm_smoke_run.py`).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add services/worker/__init__.py services/worker/main.py tests/test_worker_main.py
git commit -m "feat: add worker polling loop (claim -> run_case -> finish)"
```

---

## Task 13: Dockerfiles + docker-compose wiring

**Files:**
- Create: `services/api/Dockerfile`
- Create: `services/worker/Dockerfile`
- Create: `.env.example`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: `app/` (unchanged), `services/api/`, `services/worker/` (Tasks 8-12).
- Produces: buildable `api`/`worker` images; `docker compose --profile backend up` starts both against a real `DATABASE_URL`/`SUPABASE_JWKS_URL` supplied via `.env` (not committed).

- [ ] **Step 1: Create `services/api/Dockerfile`**

```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY app ./app
COPY services/api ./services/api

ENTRYPOINT ["uv", "run", "--no-sync", "uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create `services/worker/Dockerfile`**

```dockerfile
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY app ./app
COPY services/worker ./services/worker

ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "services.worker.main"]
```

(`coinor-cbc` is required here — unlike the API, the worker calls `run_case`, which solves with `cbc`. The API never solves anything, so it doesn't need the solver.)

- [ ] **Step 3: Update `docker-compose.yml`**

Replace the `api`/`worker` service blocks (currently `build: ./services/api` / `build: ./services/worker` under `profiles: ["future"]`) with:

```yaml
  api:
    build:
      context: .
      dockerfile: services/api/Dockerfile
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SUPABASE_JWKS_URL=${SUPABASE_JWKS_URL}
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    profiles: ["backend"]

  worker:
    build:
      context: .
      dockerfile: services/worker/Dockerfile
    environment:
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - ./data:/app/data
    profiles: ["backend"]
```

(`context: .` instead of `./services/api` — both images need `app/` from the repo root, which a subdirectory build context can't see. Profile renamed `future` -> `backend`: these are no longer placeholders, but they still shouldn't start on a bare `docker compose up` for anyone who hasn't set `DATABASE_URL`/`SUPABASE_JWKS_URL` yet.)

- [ ] **Step 4: Create `.env.example`**

```
DATABASE_URL=postgresql+psycopg://postgres:password@db.<project-ref>.supabase.co:5432/postgres
SUPABASE_JWKS_URL=https://<project-ref>.supabase.co/auth/v1/.well-known/jwks.json
```

- [ ] **Step 5: Verify compose config and image builds**

Run: `docker compose config --profile backend`
Expected: resolves cleanly, prints the full `api`/`worker` service definitions with the new `context`/`dockerfile` fields.

Run: `docker build -f services/api/Dockerfile -t despacho-udea-api:latest .`
Expected: builds successfully (no `DATABASE_URL` needed at build time — only `uv sync`, which doesn't touch the DB).

Run: `docker build -f services/worker/Dockerfile -t despacho-udea-worker:latest .`
Expected: builds successfully.

Note: running `docker compose --profile backend up` end-to-end against a live database is **not** verifiable in this environment without real Supabase credentials — that's a manual follow-up once a Supabase project exists, same kind of gap the "Brechas conocidas" sections in this repo's other specs already track.

- [ ] **Step 6: Commit**

```bash
git add services/api/Dockerfile services/worker/Dockerfile .env.example docker-compose.yml
git commit -m "build: add api/worker Dockerfiles, wire docker-compose (backend profile)"
```

---

## Task 14: Docs

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed by other tasks — documentation only.

- [ ] **Step 1: Update `README.md`**

In section "11. Brechas conocidas", remove the two lines that are no longer true:
- `- No existe backend HTTP ni frontend.` -> replace with `- No existe frontend (backend HTTP ya existe desde Fase 3).`
- `- No existe persistencia de ejecuciones, metadatos, artefactos y logs.` -> delete this line (superseded by Fase 3's DB).

Add a new short section (after the existing CLI usage section, matching the doc's existing style) documenting:
- How to run the API/worker locally: `uv run uvicorn services.api.main:app --reload` and `uv run python -m services.worker.main`, both requiring `DATABASE_URL` (and `SUPABASE_JWKS_URL` for the API) in the environment.
- How to run via Docker: `docker compose --profile backend up --build`, requiring a `.env` file (copy `.env.example`).
- How to apply migrations: `uv run alembic upgrade head` (needs `DATABASE_URL` set).

- [ ] **Step 2: Update `AGENTS.md`**

In the table row `este repo no usa FastAPI, Celery, Redis ni Thori`, correct it to reflect Fase 3's actual state — FastAPI is now used (`services/api/`), Celery/Redis are still explicitly not used (the worker is a DB-polling loop, `app/db/claim.py`). Reword the sentence so it stops warning against FastAPI itself and instead warns against copying Celery/Redis/Thori conventions, since those remain genuinely absent.

- [ ] **Step 3: Confirm the full suite is still green**

Run: `uv run pytest -q`
Expected: all tests pass (documentation-only task, but this is the final checkpoint before the PR).

- [ ] **Step 4: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: document Fase 3 API/worker/migrations usage"
```

---

## After all tasks: PR

Once all 14 tasks are committed on `fase3-api-persistencia` and `uv run pytest -q` is green:

```bash
git push -u origin fase3-api-persistencia
gh pr create --base develop --title "Fase 3: Backend API y persistencia" --body "..."
```

(Follow this repo's existing PR pattern — see PRs #3/#4/#6 for style. No AI co-authorship line, per `AGENTS.md`.)
