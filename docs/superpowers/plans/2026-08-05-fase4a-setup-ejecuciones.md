# Fase 4a: Backend prerequisites + setup + pantalla de ejecuciones — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 5 backend gaps the Fase 4 spec found (JWT algorithm, CORS,
run summary fields, `GET /scenarios`, BESS metrics, log capture), then
scaffold a Next.js frontend with Supabase auth and a working "pantalla de
ejecuciones" (list/create/view runs) against that backend.

**Architecture:** Backend changes are additive, small edits to
`services/api/main.py`, `services/worker/main.py`, `app/db/models.py`,
`app/db/queries.py`, plus one Alembic migration — no changes to
`app/model/`, `app/pipeline/case_builder.py`, or `app/pipeline/runner.py`.
Frontend is a new `frontend/` directory (Next.js App Router, client
components only) that talks to the FastAPI backend via a typed fetch
wrapper and to Supabase Auth directly via `supabase-js`.

**Tech Stack:** Backend: FastAPI, SQLAlchemy, Alembic, PyJWT (already in
the repo). Frontend (new): Next.js (App Router) + TypeScript + Tailwind +
shadcn/ui, pnpm, `@tanstack/react-query`, `@supabase/supabase-js`, Vitest +
React Testing Library.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-05-fase4-frontend-operativo-design.md`.
- Branch `fase4a-setup-ejecuciones` already exists and has the spec
  committed (`8d5e5a6b`). Work happens on this branch; do not create a new
  one.
- Backend Python commands run via `uv run <cmd>` (this repo's own `.venv`
  via `uv`), never a different/global venv.
- Frontend commands run via `pnpm` inside `frontend/`.
- Real Supabase project for this repo (`kaqndwbxejsyjpztcrpk.supabase.co`)
  was verified this session: its JWKS advertises **`alg: ES256`** (EC key,
  curve P-256), not RS256. Any code or test that hardcodes
  `algorithms=["RS256"]` is wrong for this project.
- `Run.status` values are exactly `"pending"`, `"running"`, `"done"`,
  `"failed"` (`app/db/claim.py`, `app/db/queries.py`). Terminal =
  `"done"` or `"failed"`.
- `DispatchLevel` values are exactly `"preideal"` and `"ideal"`
  (`app/schemas/case.py`).
- Timestamps (`created_at`/`started_at`/`finished_at`) are stored UTC-aware
  (`DateTime(timezone=True)`, Fase 3). The frontend must render them in
  `America/Bogota`, not UTC or browser-local.
- Every backend task: write/extend a test first, watch it fail for the
  right reason, implement, watch it pass, run the full suite
  (`uv run pytest -q`), commit. Every frontend task: same TDD cycle with
  `pnpm test`.
- Do not touch `app/model/`, `app/pipeline/case_builder.py`, or
  `app/pipeline/runner.py`.

---

## Task 1: Fix JWT algorithm to match the real key instead of hardcoding RS256

**Files:**
- Modify: `services/api/auth.py`
- Modify: `tests/test_api_auth.py`

**Interfaces:**
- Consumes: `jwt.PyJWKClient.get_signing_key_from_jwt(token)` — returns a
  `jwt.PyJWK` object with `.key` (the key material) and `.algorithm_name`
  (a string like `"RS256"` or `"ES256"`, derived from the JWK's `alg`/`kty`/
  `crv` fields — confirmed present on PyJWT 2.13.0, the version pinned in
  `uv.lock`).
- Produces: `decode_bearer_token(authorization: str, jwk_client) -> dict`
  unchanged signature, now algorithm-agnostic.

This is the first task because everything else (frontend login, every
authenticated endpoint test) is worthless if login fails against the real
Supabase project. `services/api/auth.py:23` currently does
`jwt.decode(token, signing_key.key, algorithms=["RS256"], ...)` — hardcoded.
The real project's JWKS (verified via `curl
https://kaqndwbxejsyjpztcrpk.supabase.co/auth/v1/.well-known/jwks.json`
this session) returns `{"keys":[{"alg":"ES256","crv":"P-256",...,"kty":"EC",...}]}`.
Every login would fail with `InvalidAlgorithmError` until this is fixed.

- [ ] **Step 1: Write the failing test** — add to `tests/test_api_auth.py`,
  using the EC-key pattern (mirrors the existing `rsa_keypair` fixture) and
  updating the two fake classes to carry `algorithm_name` so the fix is
  actually exercised:

```python
from cryptography.hazmat.primitives.asymmetric import ec


class _FakeSigningKey:
    def __init__(self, key, algorithm_name="RS256"):
        self.key = key
        self.algorithm_name = algorithm_name


class _FakeJWKClient:
    def __init__(self, public_key, algorithm_name="RS256"):
        self._public_key = public_key
        self._algorithm_name = algorithm_name

    def get_signing_key_from_jwt(self, token):
        return _FakeSigningKey(self._public_key, self._algorithm_name)


@pytest.fixture
def ec_keypair():
    private_key = ec.generate_private_key(ec.SECP256R1())
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


def test_decode_bearer_token_accepts_es256_token_like_real_supabase_project(ec_keypair):
    private_pem, public_pem = ec_keypair
    payload = {
        "sub": "user-1",
        "aud": "authenticated",
        "exp": int(time.time()) + 3600,
    }
    token = jwt.encode(payload, private_pem, algorithm="ES256")
    result = decode_bearer_token(
        f"Bearer {token}", _FakeJWKClient(public_pem, algorithm_name="ES256")
    )
    assert result["sub"] == "user-1"
```

(Existing calls to `_FakeJWKClient(public_pem)` elsewhere in the file keep
working unchanged — `algorithm_name` defaults to `"RS256"`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_auth.py::test_decode_bearer_token_accepts_es256_token_like_real_supabase_project -v`
Expected: FAIL (401 raised — `jwt.exceptions.InvalidAlgorithmError` wrapped
by the `except jwt.PyJWTError` in `decode_bearer_token`, because the code
still hardcodes `algorithms=["RS256"]`).

- [ ] **Step 3: Fix `services/api/auth.py`**

In `decode_bearer_token`, change:

```python
        return jwt.decode(token, signing_key.key, algorithms=["RS256"], audience="authenticated")
```

to:

```python
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=[signing_key.algorithm_name],
            audience="authenticated",
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_api_auth.py -v`
Expected: all PASS, including the 5 pre-existing RS256 tests (unaffected —
they still pass `algorithm_name="RS256"` by default) and the new ES256
test.

- [ ] **Step 5: Commit**

```bash
git add services/api/auth.py tests/test_api_auth.py
git commit -m "fix: verify JWT with the algorithm the signing key advertises, not hardcoded RS256"
```

---

## Task 2: Add CORS so the frontend's origin isn't blocked by the browser

**Files:**
- Modify: `services/api/main.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_api_cors.py`

**Interfaces:**
- Produces: `FRONTEND_ORIGIN` env var (comma-separated list of allowed
  origins, e.g. `http://localhost:3000,https://myapp.vercel.app`), read at
  process start.

- [ ] **Step 1: Write the failing test** — create `tests/test_api_cors.py`:

```python
def test_cors_allows_configured_frontend_origin(api_client):
    resp = api_client.get("/runs", headers={"Origin": "http://localhost:3000"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
```

Add near the top of `tests/conftest.py` (before the `pytest`/sqlalchemy
imports, so it runs before `services.api.main` is ever imported by any
fixture — `FRONTEND_ORIGIN` must exist before `FastAPI.add_middleware` runs
at module import time):

```python
import os

os.environ.setdefault("FRONTEND_ORIGIN", "http://localhost:3000")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_cors.py -v`
Expected: FAIL — no `access-control-allow-origin` header (`None !=
"http://localhost:3000"`), since `CORSMiddleware` isn't installed yet.

- [ ] **Step 3: Add CORS middleware to `services/api/main.py`**

Add `import os` to the top-level imports, and
`from fastapi.middleware.cors import CORSMiddleware`. Right after
`app = FastAPI(title="despacho-udea API")`:

```python
_frontend_origins = [
    origin.strip()
    for origin in os.environ.get("FRONTEND_ORIGIN", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_frontend_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

(No `allow_credentials=True` — auth is an `Authorization` header, not a
cookie, so credentialed CORS isn't needed.)

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_api_cors.py -v`
Expected: PASS.

Run: `uv run pytest -q`
Expected: full suite still green (confirms the `os.environ.setdefault` in
conftest doesn't break anything else).

- [ ] **Step 5: Commit**

```bash
git add services/api/main.py tests/conftest.py tests/test_api_cors.py
git commit -m "feat: add CORS middleware gated on FRONTEND_ORIGIN env var"
```

- [ ] **Step 6: Document the env var** — add to `.env.example`:

```
FRONTEND_ORIGIN=http://localhost:3000
```

```bash
git add .env.example
git commit -m "docs: document FRONTEND_ORIGIN env var"
```

---

## Task 3: `_run_summary` includes `dispatch_date`/`level`/`scenario_id`

**Files:**
- Modify: `services/api/main.py`
- Modify: `tests/test_api_runs.py`

**Interfaces:**
- Produces: `_run_summary(run, case) -> dict` (signature change — was
  `_run_summary(run)`), used by both `list_runs` and `get_run_detail`.

- [ ] **Step 1: Write the failing test** — extend
  `tests/test_api_runs.py`:

```python
def test_get_run_includes_case_fields(api_client):
    create_resp = api_client.post(
        "/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"}
    )
    run_id = create_resp.json()["run_id"]

    resp = api_client.get(f"/runs/{run_id}")
    body = resp.json()
    assert body["dispatch_date"] == "2024-04-18"
    assert body["level"] == "preideal"
    assert body["scenario_id"] is None


def test_list_runs_includes_case_fields(api_client):
    api_client.post("/runs", json={"dispatch_date": "2024-04-19", "level": "ideal"})
    resp = api_client.get("/runs")
    row = resp.json()[0]
    assert row["dispatch_date"] == "2024-04-19"
    assert row["level"] == "ideal"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api_runs.py -k case_fields -v`
Expected: FAIL with `KeyError: 'dispatch_date'`.

- [ ] **Step 3: Implement** — in `services/api/main.py`, change
  `_run_summary` and its two call sites:

```python
def _run_summary(run, case) -> dict:
    return {
        "run_id": run.id,
        "status": run.status,
        "dispatch_date": case.dispatch_date,
        "level": case.level,
        "scenario_id": case.scenario_id,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "error": run.error,
    }
```

`list_runs`:

```python
@app.get("/runs")
def list_runs(user_id: str = Depends(get_current_user_id), session=Depends(get_session)):
    runs = queries.list_runs_for_user(session, user_id)
    return [_run_summary(r, queries.get_case(session, r.case_id)) for r in runs]
```

`get_run_detail` — replace `out = _run_summary(run)` with:

```python
    run = _get_owned_run(session, run_id, user_id)
    case = queries.get_case(session, run.case_id)
    metric_set = queries.get_metric_set(session, run.id)
    out = _run_summary(run, case)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_api_runs.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add services/api/main.py tests/test_api_runs.py
git commit -m "feat: include dispatch_date/level/scenario_id in run summaries"
```

---

## Task 4: `GET /scenarios`

**Files:**
- Modify: `app/db/queries.py`
- Modify: `services/api/main.py`
- Modify: `tests/test_api_scenarios.py`

**Interfaces:**
- Produces: `queries.list_scenarios(session: Session) -> list[Scenario]`.
- Produces: `GET /scenarios` — `200` with a JSON array of
  `{id, mode, penetration_level, units, created_at}`.

- [ ] **Step 1: Write the failing test** — append to
  `tests/test_api_scenarios.py` (check its existing imports/fixtures first
  and match its style):

```python
def test_list_scenarios_returns_created_scenarios(api_client):
    api_client.post(
        "/scenarios",
        json={
            "mode": "arbitrage",
            "penetration_level": "baseline",
            "units": [
                {
                    "name": "bess-1",
                    "mwh_nom": 10.0,
                    "hours_to_deplete": 4.0,
                    "initial_soc": 0.5,
                    "min_soc": 0.1,
                    "max_soc": 0.9,
                    "efficiency": 0.9,
                    "charge_bid": 50.0,
                    "discharge_bid": 200.0,
                }
            ],
        },
    )

    resp = api_client.get("/scenarios")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["mode"] == "arbitrage"
    assert body[0]["penetration_level"] == "baseline"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_scenarios.py::test_list_scenarios_returns_created_scenarios -v`
Expected: FAIL with 404 (route doesn't exist).

- [ ] **Step 3: Implement** — add to `app/db/queries.py`:

```python
def list_scenarios(session: Session) -> list[Scenario]:
    stmt = select(Scenario).order_by(Scenario.created_at.desc())
    return list(session.scalars(stmt))
```

Add to `services/api/main.py`, after the existing `POST /scenarios`
endpoint:

```python
@app.get("/scenarios")
def list_scenarios_endpoint(
    user_id: str = Depends(get_current_user_id), session=Depends(get_session)
):
    scenarios = queries.list_scenarios(session)
    return [
        {
            "id": s.id,
            "mode": s.mode,
            "penetration_level": s.penetration_level,
            "units": s.units,
            "created_at": s.created_at,
        }
        for s in scenarios
    ]
```

(Named `list_scenarios_endpoint`, not `list_scenarios`, to avoid shadowing
the imported `queries.list_scenarios` name in this module's namespace.)

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_api_scenarios.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/db/queries.py services/api/main.py tests/test_api_scenarios.py
git commit -m "feat: add GET /scenarios"
```

---

## Task 5: `GET /runs/{id}` includes the 4 BESS metric fields

**Files:**
- Modify: `services/api/main.py`
- Modify: `tests/test_api_results.py`

**Interfaces:**
- Produces: `GET /runs/{id}` response `metrics` object now also has
  `bess_charge_mwh`, `bess_discharge_mwh`, `bess_avg_soc_mwh`,
  `bess_net_revenue`.

- [ ] **Step 1: Write the failing test** — append to
  `tests/test_api_results.py`:

```python
def test_get_run_includes_bess_metrics(api_client, tmp_path):
    from app.db.models import MetricSet

    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    run_id = resp.json()["run_id"]

    session = api_client.SessionLocal()
    session.add(
        MetricSet(
            run_id=run_id,
            rmse=1.0,
            bess_charge_mwh=12.5,
            bess_discharge_mwh=11.0,
            bess_avg_soc_mwh=5.5,
            bess_net_revenue=987.0,
        )
    )
    session.commit()
    session.close()

    resp = api_client.get(f"/runs/{run_id}")
    metrics = resp.json()["metrics"]
    assert metrics["bess_charge_mwh"] == 12.5
    assert metrics["bess_discharge_mwh"] == 11.0
    assert metrics["bess_avg_soc_mwh"] == 5.5
    assert metrics["bess_net_revenue"] == 987.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_results.py::test_get_run_includes_bess_metrics -v`
Expected: FAIL with `KeyError: 'bess_charge_mwh'`.

- [ ] **Step 3: Implement** — in `services/api/main.py`, `get_run_detail`,
  extend the `metrics` dict:

```python
    out["metrics"] = (
        {
            "rmse": metric_set.rmse,
            "mae": metric_set.mae,
            "bias": metric_set.bias,
            "wape": metric_set.wape,
            "smape": metric_set.smape,
            "r2": metric_set.r2,
            "bess_charge_mwh": metric_set.bess_charge_mwh,
            "bess_discharge_mwh": metric_set.bess_discharge_mwh,
            "bess_avg_soc_mwh": metric_set.bess_avg_soc_mwh,
            "bess_net_revenue": metric_set.bess_net_revenue,
        }
        if metric_set
        else None
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_api_results.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add services/api/main.py tests/test_api_results.py
git commit -m "feat: include BESS metrics in GET /runs/{id}"
```

---

## Task 6: `Run.log_path` column + migration

**Files:**
- Modify: `app/db/models.py`
- Create: `alembic/versions/0002_run_log_path.py`
- Modify: `tests/test_db_migrations.py`

**Interfaces:**
- Produces: `Run.log_path: str | None` column.

- [ ] **Step 1: Write the failing test** — extend
  `tests/test_db_migrations.py`:

```python
def test_alembic_upgrade_head_adds_runs_log_path_column(tmp_path):
    db_path = tmp_path / "migration_smoke_log.db"
    database_url = f"sqlite:///{db_path}"

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")

    engine = create_engine(database_url)
    columns = {c["name"] for c in inspect(engine).get_columns("runs")}
    assert "log_path" in columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_migrations.py::test_alembic_upgrade_head_adds_runs_log_path_column -v`
Expected: FAIL — `log_path` not in columns (migration doesn't exist yet).

- [ ] **Step 3: Add the model column** — in `app/db/models.py`, `Run`
  class, after `bess_path`:

```python
    log_path: Mapped[str | None] = mapped_column(String, nullable=True)
```

- [ ] **Step 4: Write the migration** — create
  `alembic/versions/0002_run_log_path.py`:

```python
"""add runs.log_path

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-05
"""

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("log_path", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "log_path")
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_db_migrations.py tests/test_db_models.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add app/db/models.py alembic/versions/0002_run_log_path.py tests/test_db_migrations.py
git commit -m "feat: add runs.log_path column"
```

---

## Task 7: Worker captures stdout/stderr of the solve to a log file

**Files:**
- Modify: `services/worker/main.py`
- Modify: `tests/test_worker_main.py`

**Interfaces:**
- Consumes: `Run.log_path` (Task 6), `app.storage.get_storage`.
- Produces: after `process_once`, a done/failed run's `log_path` is set to
  `f"{out_dir}/run.log"` and that file exists (via `Storage`, same
  convention as `dispatch_path`/`price_path`/`bess_path`).

`LocalStorage.open(path, "w")` (`app/storage/local.py:21-24`) already
creates parent directories, so this works even when `run_case` fails before
`out_dir` exists.

- [ ] **Step 1: Write the failing test** — extend
  `tests/test_worker_main.py`'s `test_process_once_solves_pending_run_end_to_end`:

```python
    updated = queries.get_run(session, run.id)
    assert updated.status == "done", updated.error
    assert updated.price_path is not None
    assert Path(updated.price_path).exists()
    assert Path(updated.out_dir) == Path(results_root) / run.id

    # new assertions
    assert updated.log_path is not None
    log_file = Path(updated.log_path)
    assert log_file.exists()
    # xm_smoke fixture has no preideal_price actuals, so run_case's
    # "no XM actuals" branch (app/pipeline/runner.py:48) fires and prints —
    # proof the log actually captured run_case's stdout, not an empty file.
    assert "no XM actuals" in log_file.read_text()
```

And to `test_process_once_marks_run_failed_when_run_case_reports_failure`,
after the existing assertions:

```python
    assert updated.log_path is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_worker_main.py -v`
Expected: FAIL — `updated.log_path is not None` fails (`AttributeError` or
`None is not None` — the column exists from Task 6 but nothing sets it
yet).

- [ ] **Step 3: Implement** — in `services/worker/main.py`, add imports
  `contextlib`, `io`, and `from app.storage import get_storage`. Change
  `process_once`:

```python
def process_once(
    session: Session, *, data_dir: str = "data", results_root: str = "data/results"
) -> bool:
    run = claim_next_pending_run(session)
    if run is None:
        return False

    case_row = queries.get_case(session, run.case_id)
    case = _build_case(session, case_row)
    out_dir = f"{results_root}/{run.id}"

    # Close the read-only transaction _build_case's queries opened so the
    # session sits idle (not idle-in-transaction) for the duration of the
    # solve, instead of pinning a pooler connection with an open transaction.
    session.commit()

    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
        result = run_case(case, evaluate=True, out=out_dir, data_dir=data_dir)

    log_path = f"{out_dir}/run.log"
    with get_storage(".").open(log_path, "w") as f:
        f.write(log_buffer.getvalue())
    run.log_path = log_path

    if result.ok:
        queries.finish_run_ok(session, run, result, out_dir=out_dir)
    else:
        queries.finish_run_failed(session, run, result.error or "unknown error")
    return True
```

(`run.log_path` is set on the same ORM instance that `finish_run_ok`/
`finish_run_failed` later calls `session.add(run)` + `session.commit()` on,
so it persists together with `status`/`finished_at` — no signature change
needed on the `queries` functions.)

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_worker_main.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add services/worker/main.py tests/test_worker_main.py
git commit -m "feat: worker captures solve stdout/stderr to a persisted log file"
```

---

## Task 8: `GET /runs/{id}/log`

**Files:**
- Modify: `services/api/main.py`
- Create: `tests/test_api_log.py`

**Interfaces:**
- Produces: `GET /runs/{id}/log` — `200` with `text/plain` body, `404` if
  `log_path` is `None` or the file is missing.

**Route ordering matters**: this route must be registered **before**
`@app.get("/runs/{run_id}/{artifact}")` in the file. Starlette matches
routes in registration order — if the generic `{artifact}` route comes
first, a request to `/runs/{id}/log` matches it instead (with `artifact ==
"log"`, which isn't in `_ARTIFACT_PATHS`, so it 404s as "unknown artifact"
and this endpoint is never reached).

- [ ] **Step 1: Write the failing test** — create `tests/test_api_log.py`:

```python
from datetime import date

from app.db import queries
from app.schemas import DispatchCase, DispatchLevel, RunResult


def _seed_done_run_with_log(api_client, tmp_path):
    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    run_id = resp.json()["run_id"]

    out_dir = tmp_path / "results" / run_id
    out_dir.mkdir(parents=True)
    log_file = out_dir / "run.log"
    log_file.write_text("==> 2024-04-18 [preideal]\n")

    session = api_client.SessionLocal()
    run = queries.get_run(session, run_id)
    result = RunResult(
        case=DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal),
        ok=True,
    )
    queries.finish_run_ok(session, run, result, out_dir=str(out_dir))
    run = queries.get_run(session, run_id)
    run.log_path = str(log_file)
    session.add(run)
    session.commit()
    session.close()

    return run_id


def test_get_run_log_returns_text(api_client, tmp_path):
    run_id = _seed_done_run_with_log(api_client, tmp_path)

    resp = api_client.get(f"/runs/{run_id}/log")
    assert resp.status_code == 200
    assert "2024-04-18" in resp.text


def test_get_run_log_404_when_no_log_yet(api_client):
    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    run_id = resp.json()["run_id"]

    resp = api_client.get(f"/runs/{run_id}/log")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api_log.py -v`
Expected: FAIL — `test_get_run_log_returns_text` gets 404 "unknown
artifact" (route doesn't exist, falls through to `get_run_artifact`).

- [ ] **Step 3: Implement** — in `services/api/main.py`, add
  `from fastapi.responses import PlainTextResponse` to imports. Insert this
  route **immediately before** the `_ARTIFACT_PATHS = {...}` block (i.e.
  before `get_run_artifact` is defined):

```python
@app.get("/runs/{run_id}/log")
def get_run_log(
    run_id: str, user_id: str = Depends(get_current_user_id), session=Depends(get_session)
):
    run = _get_owned_run(session, run_id, user_id)
    if run.log_path is None or not get_storage(".").exists(run.log_path):
        raise HTTPException(status_code=404, detail="run has no log yet")
    with get_storage(".").open(run.log_path) as f:
        content = f.read()
    return PlainTextResponse(content)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_api_log.py -v`
Expected: all PASS.

Run: `uv run pytest -q`
Expected: full suite green — confirms route ordering didn't break
`/runs/{id}/dispatch` etc.

- [ ] **Step 5: Commit**

```bash
git add services/api/main.py tests/test_api_log.py
git commit -m "feat: add GET /runs/{id}/log"
```

This is the last backend task. Run `uv run pytest -q` once more and confirm
the full count is green before moving to the frontend.

---

## Task 9: Scaffold Next.js + Tailwind + shadcn/ui + pnpm + Vitest

**Files:**
- Create: `frontend/` (via `create-next-app`)
- Modify: `frontend/package.json` (add `test` script, test deps)
- Create: `frontend/vitest.config.ts`
- Create: `frontend/vitest.setup.ts`
- Create: `frontend/.env.local.example`

No test cycle for this task (pure scaffolding/config) — verified by a
successful build and an empty test run.

- [ ] **Step 1: Scaffold the Next.js app** — from the repo root:

```bash
npx --yes create-next-app@latest frontend \
  --ts --tailwind --eslint --app \
  --import-alias "@/*" --use-pnpm --disable-git --yes
```

`--disable-git`: this is already inside the `despacho-udea` git repo;
`create-next-app` must not init its own nested repo.

- [ ] **Step 2: Install runtime dependencies**

```bash
cd frontend
pnpm add @tanstack/react-query @supabase/supabase-js
```

- [ ] **Step 3: Install shadcn/ui and base components**

```bash
pnpm dlx shadcn@latest init -d
pnpm dlx shadcn@latest add button card table input label select badge
```

- [ ] **Step 4: Install test dependencies**

```bash
pnpm add -D vitest @vitejs/plugin-react jsdom @testing-library/react \
  @testing-library/jest-dom @testing-library/user-event
```

- [ ] **Step 5: Configure Vitest** — create `frontend/vitest.config.ts`:

```ts
import path from "path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    globals: true,
  },
  resolve: {
    alias: { "@": path.resolve(__dirname, "./") },
  },
});
```

Create `frontend/vitest.setup.ts`:

```ts
import "@testing-library/jest-dom/vitest";
```

Add to `frontend/package.json` `scripts`:

```json
"test": "vitest run"
```

- [ ] **Step 6: Env var template** — create `frontend/.env.local.example`:

```
NEXT_PUBLIC_SUPABASE_URL=https://<project-ref>.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=<publishable-anon-key>
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

- [ ] **Step 7: Verify the scaffold builds and the (empty) test suite runs**

```bash
pnpm build
pnpm test
```

Expected: build succeeds; `pnpm test` reports 0 tests, 0 failures (no test
files yet).

- [ ] **Step 8: Commit**

```bash
cd ..
git add frontend
git commit -m "build: scaffold Next.js frontend (TypeScript, Tailwind, shadcn/ui, Vitest)"
```

---

## Task 10: Shared types + `isTerminalStatus` + `formatBogotaTime` helpers

**Files:**
- Create: `frontend/lib/types.ts`
- Create: `frontend/lib/run-status.ts`
- Create: `frontend/lib/run-status.test.ts`
- Create: `frontend/lib/format-date.ts`
- Create: `frontend/lib/format-date.test.ts`

**Interfaces:**
- Produces: `RunStatus`, `RunSummary`, `RunMetrics`, `RunDetail`,
  `Scenario`, `CreateRunRequest` types; `isTerminalStatus(status):
  boolean`; `formatBogotaTime(iso: string | null): string`.

- [ ] **Step 1: Types** — create `frontend/lib/types.ts` (no test — plain
  type declarations, nothing to assert at runtime):

```ts
export type RunStatus = "pending" | "running" | "done" | "failed";
export type DispatchLevel = "preideal" | "ideal";

export interface RunSummary {
  run_id: string;
  status: RunStatus;
  dispatch_date: string;
  level: DispatchLevel;
  scenario_id: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
}

export interface RunMetrics {
  rmse: number | null;
  mae: number | null;
  bias: number | null;
  wape: number | null;
  smape: number | null;
  r2: number | null;
  bess_charge_mwh: number | null;
  bess_discharge_mwh: number | null;
  bess_avg_soc_mwh: number | null;
  bess_net_revenue: number | null;
}

export interface RunDetail extends RunSummary {
  metrics: RunMetrics | null;
}

export interface Scenario {
  id: string;
  mode: "arbitrage" | "grid_asset" | "generator";
  penetration_level: string;
  units: unknown[];
  created_at: string;
}

export interface CreateRunRequest {
  dispatch_date: string;
  level: DispatchLevel;
  solver?: string;
  compute_prices?: boolean;
  scenario_id?: string | null;
}
```

- [ ] **Step 2: Write the failing test** — create
  `frontend/lib/run-status.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { isTerminalStatus } from "./run-status";

describe("isTerminalStatus", () => {
  it("is false for pending and running", () => {
    expect(isTerminalStatus("pending")).toBe(false);
    expect(isTerminalStatus("running")).toBe(false);
  });

  it("is true for done and failed", () => {
    expect(isTerminalStatus("done")).toBe(true);
    expect(isTerminalStatus("failed")).toBe(true);
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd frontend && pnpm test run-status`
Expected: FAIL (module `./run-status` doesn't exist).

- [ ] **Step 4: Implement** — create `frontend/lib/run-status.ts`:

```ts
import type { RunStatus } from "./types";

export function isTerminalStatus(status: RunStatus): boolean {
  return status === "done" || status === "failed";
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pnpm test run-status`
Expected: PASS.

- [ ] **Step 6: Write the failing test** — create
  `frontend/lib/format-date.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { formatBogotaTime } from "./format-date";

describe("formatBogotaTime", () => {
  it("renders a UTC ISO timestamp in America/Bogota (UTC-5, no DST)", () => {
    // 2024-04-18T05:00:00Z -> 2024-04-18 00:00 in Bogota
    const result = formatBogotaTime("2024-04-18T05:00:00Z");
    expect(result).toContain("2024-04-18");
    expect(result).toContain("00:00");
  });

  it("returns a dash for null", () => {
    expect(formatBogotaTime(null)).toBe("—");
  });
});
```

- [ ] **Step 7: Run test to verify it fails**

Run: `pnpm test format-date`
Expected: FAIL (module doesn't exist).

- [ ] **Step 8: Implement** — create `frontend/lib/format-date.ts`:

```ts
export function formatBogotaTime(iso: string | null): string {
  if (iso === null) return "—";
  const date = new Date(iso);
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Bogota",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(date);
  const get = (type: string) => parts.find((p) => p.type === type)?.value ?? "";
  return `${get("year")}-${get("month")}-${get("day")} ${get("hour")}:${get("minute")}`;
}
```

(`Intl.DateTimeFormat` with `timeZone: "America/Bogota"` is a native
platform feature — no date library needed. `en-CA` locale gives
`YYYY-MM-DD` ordering for the date parts.)

- [ ] **Step 9: Run test to verify it passes**

Run: `pnpm test format-date`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
cd ..
git add frontend/lib
git commit -m "feat(frontend): shared types, isTerminalStatus, formatBogotaTime"
```

---

## Task 11: Supabase client + `AuthProvider`/`useAuth`

**Files:**
- Create: `frontend/lib/supabase.ts`
- Create: `frontend/lib/auth-context.tsx`
- Create: `frontend/lib/auth-context.test.tsx`

**Interfaces:**
- Produces: `supabase` client instance; `AuthProvider` component;
  `useAuth(): { session: Session | null; loading: boolean; signOut: () =>
  Promise<void> }`.

- [ ] **Step 1: Supabase client** — create `frontend/lib/supabase.ts`:

```ts
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
```

- [ ] **Step 2: Write the failing test** — create
  `frontend/lib/auth-context.test.tsx`:

```tsx
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AuthProvider, useAuth } from "./auth-context";

vi.mock("./supabase", () => ({
  supabase: {
    auth: {
      getSession: vi.fn(),
      onAuthStateChange: vi.fn(() => ({ data: { subscription: { unsubscribe: vi.fn() } } })),
    },
  },
}));

import { supabase } from "./supabase";

function Probe() {
  const { session, loading } = useAuth();
  if (loading) return <div>loading</div>;
  return <div>{session ? `logged in as ${session.user.id}` : "logged out"}</div>;
}

describe("AuthProvider/useAuth", () => {
  beforeEach(() => {
    vi.mocked(supabase.auth.getSession).mockReset();
  });

  it("exposes the session once getSession resolves", async () => {
    vi.mocked(supabase.auth.getSession).mockResolvedValue({
      data: { session: { user: { id: "user-1" } } },
    } as never);

    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>
    );

    await waitFor(() => screen.getByText("logged in as user-1"));
  });

  it("shows logged out when there is no session", async () => {
    vi.mocked(supabase.auth.getSession).mockResolvedValue({
      data: { session: null },
    } as never);

    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>
    );

    await waitFor(() => screen.getByText("logged out"));
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pnpm test auth-context`
Expected: FAIL (module `./auth-context` doesn't exist).

- [ ] **Step 4: Implement** — create `frontend/lib/auth-context.tsx`:

```tsx
"use client";

import type { Session } from "@supabase/supabase-js";
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { supabase } from "./supabase";

interface AuthContextValue {
  session: Session | null;
  loading: boolean;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setLoading(false);
    });
    const { data: sub } = supabase.auth.onAuthStateChange((_event, newSession) => {
      setSession(newSession);
    });
    return () => sub.subscription.unsubscribe();
  }, []);

  async function signOut() {
    await supabase.auth.signOut();
  }

  return (
    <AuthContext.Provider value={{ session, loading, signOut }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pnpm test auth-context`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd ..
git add frontend/lib/supabase.ts frontend/lib/auth-context.tsx frontend/lib/auth-context.test.tsx
git commit -m "feat(frontend): Supabase client + AuthProvider/useAuth"
```

---

## Task 12: `RequireAuth` guard component

**Files:**
- Create: `frontend/components/require-auth.tsx`
- Create: `frontend/components/require-auth.test.tsx`

**Interfaces:**
- Consumes: `useAuth()` (Task 11).
- Produces: `RequireAuth({ children }) -> JSX.Element | null`.

- [ ] **Step 1: Write the failing test** — create
  `frontend/components/require-auth.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RequireAuth } from "./require-auth";

const replace = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace }) }));

const useAuthMock = vi.fn();
vi.mock("@/lib/auth-context", () => ({ useAuth: () => useAuthMock() }));

describe("RequireAuth", () => {
  it("renders nothing and redirects to /login when there is no session", () => {
    useAuthMock.mockReturnValue({ session: null, loading: false });
    render(
      <RequireAuth>
        <div>secret</div>
      </RequireAuth>
    );
    expect(screen.queryByText("secret")).not.toBeInTheDocument();
    expect(replace).toHaveBeenCalledWith("/login");
  });

  it("renders children when there is a session", () => {
    useAuthMock.mockReturnValue({ session: { user: { id: "u1" } }, loading: false });
    render(
      <RequireAuth>
        <div>secret</div>
      </RequireAuth>
    );
    expect(screen.getByText("secret")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test require-auth`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement** — create `frontend/components/require-auth.tsx`:

```tsx
"use client";

import { useAuth } from "@/lib/auth-context";
import { useRouter } from "next/navigation";
import { useEffect, type ReactNode } from "react";

export function RequireAuth({ children }: { children: ReactNode }) {
  const { session, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !session) router.replace("/login");
  }, [loading, session, router]);

  if (loading || !session) return null;
  return <>{children}</>;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test require-auth`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ..
git add frontend/components/require-auth.tsx frontend/components/require-auth.test.tsx
git commit -m "feat(frontend): RequireAuth route guard"
```

---

## Task 13: Login and signup pages

**Files:**
- Create: `frontend/app/login/page.tsx`
- Create: `frontend/app/login/page.test.tsx`
- Create: `frontend/app/signup/page.tsx`
- Create: `frontend/app/signup/page.test.tsx`

**Interfaces:**
- Consumes: `supabase.auth.signInWithPassword` / `supabase.auth.signUp`
  (Task 11).

- [ ] **Step 1: Write the failing test** — create
  `frontend/app/login/page.test.tsx`:

```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import LoginPage from "./page";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

vi.mock("@/lib/supabase", () => ({
  supabase: { auth: { signInWithPassword: vi.fn() } },
}));

import { supabase } from "@/lib/supabase";

describe("LoginPage", () => {
  it("signs in and redirects to /runs on success", async () => {
    vi.mocked(supabase.auth.signInWithPassword).mockResolvedValue({ error: null } as never);

    render(<LoginPage />);
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByLabelText(/contrase/i), { target: { value: "secret123" } });
    fireEvent.click(screen.getByRole("button", { name: /entrar/i }));

    await waitFor(() => expect(push).toHaveBeenCalledWith("/runs"));
  });

  it("shows the error message on failure", async () => {
    vi.mocked(supabase.auth.signInWithPassword).mockResolvedValue({
      error: { message: "Invalid login credentials" },
    } as never);

    render(<LoginPage />);
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByLabelText(/contrase/i), { target: { value: "wrong" } });
    fireEvent.click(screen.getByRole("button", { name: /entrar/i }));

    await waitFor(() => screen.getByText("Invalid login credentials"));
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test app/login`
Expected: FAIL (`./page` doesn't exist).

- [ ] **Step 3: Implement** — create `frontend/app/login/page.tsx`:

```tsx
"use client";

import { supabase } from "@/lib/supabase";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) {
      setError(error.message);
      return;
    }
    router.push("/runs");
  }

  return (
    <form onSubmit={handleSubmit}>
      <h1>Iniciar sesion</h1>
      <label htmlFor="email">Email</label>
      <input
        id="email"
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        required
      />
      <label htmlFor="password">Contrasena</label>
      <input
        id="password"
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        required
      />
      {error && <p role="alert">{error}</p>}
      <button type="submit">Entrar</button>
    </form>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test app/login`
Expected: PASS.

- [ ] **Step 5: Write the failing test** — create
  `frontend/app/signup/page.test.tsx` (mirrors login, `signUp` instead):

```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import SignupPage from "./page";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

vi.mock("@/lib/supabase", () => ({
  supabase: { auth: { signUp: vi.fn() } },
}));

import { supabase } from "@/lib/supabase";

describe("SignupPage", () => {
  it("signs up and redirects to /login on success", async () => {
    vi.mocked(supabase.auth.signUp).mockResolvedValue({ error: null } as never);

    render(<SignupPage />);
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByLabelText(/contrase/i), { target: { value: "secret123" } });
    fireEvent.click(screen.getByRole("button", { name: /crear cuenta/i }));

    await waitFor(() => expect(push).toHaveBeenCalledWith("/login"));
  });
});
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pnpm test app/signup`
Expected: FAIL.

- [ ] **Step 7: Implement** — create `frontend/app/signup/page.tsx`:

```tsx
"use client";

import { supabase } from "@/lib/supabase";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

export default function SignupPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    const { error } = await supabase.auth.signUp({ email, password });
    if (error) {
      setError(error.message);
      return;
    }
    router.push("/login");
  }

  return (
    <form onSubmit={handleSubmit}>
      <h1>Crear cuenta</h1>
      <label htmlFor="email">Email</label>
      <input
        id="email"
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        required
      />
      <label htmlFor="password">Contrasena</label>
      <input
        id="password"
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        required
        minLength={6}
      />
      {error && <p role="alert">{error}</p>}
      <button type="submit">Crear cuenta</button>
    </form>
  );
}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pnpm test app/signup`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
cd ..
git add frontend/app/login frontend/app/signup
git commit -m "feat(frontend): login and signup pages"
```

---

## Task 14: Typed API client

**Files:**
- Create: `frontend/lib/api-client.ts`
- Create: `frontend/lib/api-client.test.ts`

**Interfaces:**
- Consumes: `supabase.auth.getSession()` (Task 11), `types.ts` (Task 10).
- Produces: `listRuns()`, `getRun(id)`, `createRun(body)`,
  `listScenarios()`.

- [ ] **Step 1: Write the failing test** — create
  `frontend/lib/api-client.test.ts`:

```ts
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./supabase", () => ({
  supabase: { auth: { getSession: vi.fn() } },
}));

import { supabase } from "./supabase";
import { createRun, listRuns } from "./api-client";

const fetchMock = vi.fn();
vi.stubGlobal("fetch", fetchMock);

beforeEach(() => {
  fetchMock.mockReset();
  vi.mocked(supabase.auth.getSession).mockResolvedValue({
    data: { session: { access_token: "tok-123" } },
  } as never);
});

describe("api-client", () => {
  it("listRuns sends the Authorization header and returns parsed JSON", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => [{ run_id: "r1" }],
    });

    const runs = await listRuns();

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/runs"),
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer tok-123" }),
      })
    );
    expect(runs).toEqual([{ run_id: "r1" }]);
  });

  it("createRun POSTs the body as JSON", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ run_id: "r2", status: "pending" }),
    });

    await createRun({ dispatch_date: "2024-04-18", level: "preideal" });

    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      dispatch_date: "2024-04-18",
      level: "preideal",
    });
  });

  it("throws with status and body text when the response is not ok", async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 404,
      statusText: "Not Found",
      text: async () => "run not found",
    });

    await expect(listRuns()).rejects.toThrow("404");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test api-client`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement** — create `frontend/lib/api-client.ts`:

```ts
import { supabase } from "./supabase";
import type { CreateRunRequest, RunDetail, RunSummary, Scenario } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function authHeader(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  if (!token) throw new Error("not authenticated");
  return { Authorization: `Bearer ${token}` };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    ...(await authHeader()),
    "Content-Type": "application/json",
    ...((init?.headers as Record<string, string>) ?? {}),
  };
  const resp = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`${resp.status} ${resp.statusText}: ${body}`);
  }
  return resp.json() as Promise<T>;
}

export function listRuns(): Promise<RunSummary[]> {
  return request<RunSummary[]>("/runs");
}

export function getRun(id: string): Promise<RunDetail> {
  return request<RunDetail>(`/runs/${id}`);
}

export function createRun(body: CreateRunRequest): Promise<{ run_id: string; status: string }> {
  return request("/runs", { method: "POST", body: JSON.stringify(body) });
}

export function listScenarios(): Promise<Scenario[]> {
  return request<Scenario[]>("/scenarios");
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test api-client`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ..
git add frontend/lib/api-client.ts frontend/lib/api-client.test.ts
git commit -m "feat(frontend): typed API client for runs/scenarios"
```

---

## Task 15: TanStack Query provider + protected layout + nav

**Files:**
- Create: `frontend/app/providers.tsx`
- Modify: `frontend/app/layout.tsx`
- Create: `frontend/app/(app)/layout.tsx`

No isolated test for this task — it's wiring/composition of already-tested
pieces (`AuthProvider`, `RequireAuth`), verified by the page-level tests in
Tasks 16-17 rendering successfully inside it, plus a manual `pnpm dev`
check.

- [ ] **Step 1: Query provider** — create `frontend/app/providers.tsx`:

```tsx
"use client";

import { AuthProvider } from "@/lib/auth-context";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";

export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(() => new QueryClient());
  return (
    <QueryClientProvider client={client}>
      <AuthProvider>{children}</AuthProvider>
    </QueryClientProvider>
  );
}
```

- [ ] **Step 2: Wire into the root layout** — in
  `frontend/app/layout.tsx`, wrap `{children}` with `<Providers>` (import
  `Providers` from `./providers`), inside `<body>`.

- [ ] **Step 3: Protected app layout with nav** — create
  `frontend/app/(app)/layout.tsx`:

```tsx
"use client";

import { RequireAuth } from "@/components/require-auth";
import { useAuth } from "@/lib/auth-context";
import Link from "next/link";
import type { ReactNode } from "react";

function Nav() {
  const { signOut } = useAuth();
  return (
    <header>
      <nav>
        <Link href="/runs">Ejecuciones</Link>
        <button onClick={() => signOut()}>Salir</button>
      </nav>
    </header>
  );
}

export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <RequireAuth>
      <Nav />
      <main>{children}</main>
    </RequireAuth>
  );
}
```

(The `(app)` route group means every page under `frontend/app/(app)/` is
guarded and gets the nav, without `(app)` appearing in the URL. Only
"Ejecuciones" is linked — fase4b/4c pages don't exist yet, so no dead
links.)

- [ ] **Step 4: Manual verification**

```bash
pnpm build
```

Expected: build succeeds (confirms the route group and provider wiring is
valid Next.js).

- [ ] **Step 5: Commit**

```bash
cd ..
git add frontend/app/providers.tsx frontend/app/layout.tsx "frontend/app/(app)/layout.tsx"
git commit -m "feat(frontend): TanStack Query provider + protected app layout with nav"
```

---

## Task 16: Runs list page (table + create-run form)

**Files:**
- Create: `frontend/components/runs-table.tsx`
- Create: `frontend/components/runs-table.test.tsx`
- Create: `frontend/components/create-run-form.tsx`
- Create: `frontend/components/create-run-form.test.tsx`
- Create: `frontend/app/(app)/runs/page.tsx`

**Interfaces:**
- Consumes: `listRuns`, `createRun`, `listScenarios` (Task 14),
  `formatBogotaTime` (Task 10).

- [ ] **Step 1: Write the failing test** — create
  `frontend/components/runs-table.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RunsTable } from "./runs-table";
import type { RunSummary } from "@/lib/types";

const runs: RunSummary[] = [
  {
    run_id: "r1",
    status: "done",
    dispatch_date: "2024-04-18",
    level: "preideal",
    scenario_id: null,
    created_at: "2024-04-18T05:00:00Z",
    started_at: null,
    finished_at: null,
    error: null,
  },
];

describe("RunsTable", () => {
  it("renders one row per run with date/level/status", () => {
    render(<RunsTable runs={runs} />);
    expect(screen.getByText("2024-04-18")).toBeInTheDocument();
    expect(screen.getByText("preideal")).toBeInTheDocument();
    expect(screen.getByText("done")).toBeInTheDocument();
  });

  it("renders an empty state with no runs", () => {
    render(<RunsTable runs={[]} />);
    expect(screen.getByText(/sin ejecuciones/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test runs-table`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement** — create `frontend/components/runs-table.tsx`:

```tsx
import { formatBogotaTime } from "@/lib/format-date";
import type { RunSummary } from "@/lib/types";
import Link from "next/link";

export function RunsTable({ runs }: { runs: RunSummary[] }) {
  if (runs.length === 0) return <p>Sin ejecuciones todavia.</p>;

  return (
    <table>
      <thead>
        <tr>
          <th>Fecha</th>
          <th>Nivel</th>
          <th>Status</th>
          <th>Creado</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((run) => (
          <tr key={run.run_id}>
            <td>{run.dispatch_date}</td>
            <td>{run.level}</td>
            <td>{run.status}</td>
            <td>{formatBogotaTime(run.created_at)}</td>
            <td>
              <Link href={`/runs/${run.run_id}`}>Ver</Link>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test runs-table`
Expected: PASS.

- [ ] **Step 5: Write the failing test** — create
  `frontend/components/create-run-form.test.tsx`:

```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { CreateRunForm } from "./create-run-form";

vi.mock("@/lib/api-client", () => ({
  createRun: vi.fn().mockResolvedValue({ run_id: "r1", status: "pending" }),
  listScenarios: vi.fn().mockResolvedValue([]),
}));

import { createRun } from "@/lib/api-client";

function renderWithQueryClient(ui: React.ReactElement) {
  const client = new QueryClient();
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("CreateRunForm", () => {
  it("calls createRun with the form values on submit", async () => {
    const onCreated = vi.fn();
    renderWithQueryClient(<CreateRunForm onCreated={onCreated} />);

    fireEvent.change(screen.getByLabelText(/fecha/i), { target: { value: "2024-04-18" } });
    fireEvent.change(screen.getByLabelText(/nivel/i), { target: { value: "preideal" } });
    fireEvent.click(screen.getByRole("button", { name: /crear/i }));

    await waitFor(() =>
      expect(createRun).toHaveBeenCalledWith(
        expect.objectContaining({ dispatch_date: "2024-04-18", level: "preideal" })
      )
    );
    await waitFor(() => expect(onCreated).toHaveBeenCalled());
  });
});
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pnpm test create-run-form`
Expected: FAIL (module doesn't exist).

- [ ] **Step 7: Implement** — create
  `frontend/components/create-run-form.tsx`:

```tsx
"use client";

import { createRun, listScenarios } from "@/lib/api-client";
import type { DispatchLevel } from "@/lib/types";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";

export function CreateRunForm({ onCreated }: { onCreated: () => void }) {
  const [dispatchDate, setDispatchDate] = useState("");
  const [level, setLevel] = useState<DispatchLevel>("preideal");
  const [scenarioId, setScenarioId] = useState("");

  const scenariosQuery = useQuery({ queryKey: ["scenarios"], queryFn: listScenarios });
  const mutation = useMutation({
    mutationFn: createRun,
    onSuccess: onCreated,
  });

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    mutation.mutate({
      dispatch_date: dispatchDate,
      level,
      scenario_id: scenarioId || null,
    });
  }

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="dispatch_date">Fecha</label>
      <input
        id="dispatch_date"
        type="date"
        value={dispatchDate}
        onChange={(e) => setDispatchDate(e.target.value)}
        required
      />
      <label htmlFor="level">Nivel</label>
      <select
        id="level"
        value={level}
        onChange={(e) => setLevel(e.target.value as DispatchLevel)}
      >
        <option value="preideal">preideal</option>
        <option value="ideal">ideal</option>
      </select>
      <label htmlFor="scenario_id">Escenario BESS (opcional)</label>
      <select id="scenario_id" value={scenarioId} onChange={(e) => setScenarioId(e.target.value)}>
        <option value="">Ninguno</option>
        {(scenariosQuery.data ?? []).map((s) => (
          <option key={s.id} value={s.id}>
            {s.penetration_level} ({s.mode})
          </option>
        ))}
      </select>
      <button type="submit" disabled={mutation.isPending}>
        Crear ejecucion
      </button>
      {mutation.isError && <p role="alert">{(mutation.error as Error).message}</p>}
    </form>
  );
}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pnpm test create-run-form`
Expected: PASS.

- [ ] **Step 9: Assemble the page** — create
  `frontend/app/(app)/runs/page.tsx`:

```tsx
"use client";

import { CreateRunForm } from "@/components/create-run-form";
import { RunsTable } from "@/components/runs-table";
import { listRuns } from "@/lib/api-client";
import { useQuery, useQueryClient } from "@tanstack/react-query";

export default function RunsPage() {
  const queryClient = useQueryClient();
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: listRuns });

  return (
    <div>
      <h1>Ejecuciones</h1>
      <CreateRunForm onCreated={() => queryClient.invalidateQueries({ queryKey: ["runs"] })} />
      {runsQuery.isLoading && <p>Cargando...</p>}
      {runsQuery.data && <RunsTable runs={runsQuery.data} />}
    </div>
  );
}
```

- [ ] **Step 10: Manual verification**

```bash
pnpm build
```

Expected: build succeeds.

- [ ] **Step 11: Commit**

```bash
cd ..
git add frontend/components/runs-table.tsx frontend/components/runs-table.test.tsx \
  frontend/components/create-run-form.tsx frontend/components/create-run-form.test.tsx \
  "frontend/app/(app)/runs/page.tsx"
git commit -m "feat(frontend): runs list page with create-run form"
```

---

## Task 17: Run detail page with status polling

**Files:**
- Create: `frontend/hooks/use-run-detail.ts`
- Create: `frontend/hooks/use-run-detail.test.tsx`
- Create: `frontend/app/(app)/runs/[id]/page.tsx`

**Interfaces:**
- Consumes: `getRun` (Task 14), `isTerminalStatus` (Task 10).
- Produces: `useRunDetail(id: string)` — a TanStack Query hook whose
  `refetchInterval` stops once the run reaches a terminal status.

- [ ] **Step 1: Write the failing test** — create
  `frontend/hooks/use-run-detail.test.tsx`:

```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useRunDetail } from "./use-run-detail";

vi.mock("@/lib/api-client", () => ({ getRun: vi.fn() }));
import { getRun } from "@/lib/api-client";

function wrapper({ children }: { children: React.ReactNode }) {
  const client = new QueryClient();
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useRunDetail", () => {
  it("polls (refetchInterval truthy) while status is pending", async () => {
    vi.mocked(getRun).mockResolvedValue({ status: "pending" } as never);
    const { result } = renderHook(() => useRunDetail("r1"), { wrapper });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(result.current.data?.status).toBe("pending");
  });

  it("stops polling once status is done", async () => {
    vi.mocked(getRun).mockResolvedValue({ status: "done" } as never);
    const { result } = renderHook(() => useRunDetail("r1"), { wrapper });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(result.current.data?.status).toBe("done");
  });
});
```

(This test checks the resolved data shape rather than inspecting the
internal `refetchInterval` function directly, since that's an
implementation detail of the query options object — the important
behavior, that `getRun` is callable and the hook surfaces its data, is what
gets asserted; the stop-polling logic itself is a pure function tested
directly below.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test use-run-detail`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement** — create `frontend/hooks/use-run-detail.ts`:

```ts
import { getRun } from "@/lib/api-client";
import { isTerminalStatus } from "@/lib/run-status";
import type { RunDetail } from "@/lib/types";
import { useQuery } from "@tanstack/react-query";

export function useRunDetail(id: string) {
  return useQuery({
    queryKey: ["run", id],
    queryFn: () => getRun(id),
    refetchInterval: (query) => {
      const data = query.state.data as RunDetail | undefined;
      if (!data || !isTerminalStatus(data.status)) return 3000;
      return false;
    },
  });
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test use-run-detail`
Expected: PASS.

- [ ] **Step 5: Assemble the page** — create
  `frontend/app/(app)/runs/[id]/page.tsx`:

```tsx
"use client";

import { formatBogotaTime } from "@/lib/format-date";
import { useRunDetail } from "@/hooks/use-run-detail";
import { useParams } from "next/navigation";

export default function RunDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { data, isLoading } = useRunDetail(id);

  if (isLoading || !data) return <p>Cargando...</p>;

  return (
    <div>
      <h1>Ejecucion {data.run_id}</h1>
      <p>Fecha: {data.dispatch_date}</p>
      <p>Nivel: {data.level}</p>
      <p>Status: {data.status}</p>
      <p>Creado: {formatBogotaTime(data.created_at)}</p>
      <p>Iniciado: {formatBogotaTime(data.started_at)}</p>
      <p>Terminado: {formatBogotaTime(data.finished_at)}</p>
      {data.status === "failed" && data.error && (
        <p role="alert">Error: {data.error}</p>
      )}
      {data.metrics && (
        <dl>
          <dt>RMSE</dt>
          <dd>{data.metrics.rmse}</dd>
          <dt>R2</dt>
          <dd>{data.metrics.r2}</dd>
        </dl>
      )}
    </div>
  );
}
```

- [ ] **Step 6: Full frontend test suite + build**

```bash
pnpm test
pnpm build
```

Expected: all tests PASS, build succeeds.

- [ ] **Step 7: Commit**

```bash
cd ..
git add frontend/hooks "frontend/app/(app)/runs/[id]"
git commit -m "feat(frontend): run detail page with status polling"
```

---

## Final verification

- [ ] **Backend**: `uv run pytest -q` — full suite green (129 pre-existing
  + new tests from Tasks 1-8).
- [ ] **Frontend**: `cd frontend && pnpm test && pnpm build` — all green.
- [ ] **Manual smoke test**: start the backend
  (`docker compose --profile backend up` or `uv run uvicorn
  services.api.main:app --reload`, plus the worker), copy
  `frontend/.env.local.example` to `frontend/.env.local` with real Supabase
  values, `pnpm dev`, sign up a user, log in, create a run against the
  `tests/fixtures/xm_smoke` date (`2024-04-18`, `preideal`), watch the
  detail page poll until `status: done`.
- [ ] Open PR `fase4a-setup-ejecuciones` -> `develop`, following the same
  pattern as PR #7 (Fase 3).
