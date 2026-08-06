# Ingesta/Storage XM — Fundacion (manifest + fixes Storage) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two unblocking pieces of `docs/superpowers/specs/2026-08-06-ingesta-storage-xm-design.md` that have zero open research questions: the `input_datasets` manifest table (Postgres/Supabase, via the existing SQLAlchemy+Alembic stack) and closing the two `Storage`-bypass gaps in `case_builder.py` (GitHub issues #24 and #25).

**Architecture:** Extend the existing `app/db/models.py`/`alembic/versions/` stack with one new table, following the exact patterns already used for `Scenario`/`Case`/`Run`/`MetricSet`. Extend the `Storage` protocol (`app/storage/base.py`) with an optional `encoding` parameter so `case_builder.py`'s two remaining plain `open()` calls (`ramps.json`, `preideal_dispatch_map.json`) can route through `storage.open()` like every other file read in that module already does.

**Tech Stack:** SQLAlchemy 2.0 (declarative `Mapped`/`mapped_column`), Alembic, pytest, SQLite in-memory for tests (Postgres in prod, per existing `app/db/session.py`).

## Global Constraints

- Match `app/db/models.py`'s exact style: `class Base(DeclarativeBase)`, string-uuid primary keys via `_new_id()` (`uuid.uuid4().hex`), `Mapped[...]`/`mapped_column(...)`.
- Query functions are plain functions taking a `Session` as first arg (see `app/db/queries.py`) — no repository/DAO classes.
- Alembic revisions are sequential zero-padded ids (`"0001"`, `"0002"`, next is `"0003"`), one file per revision in `alembic/versions/`, `down_revision` chains to the previous one.
- Tests use `sqlite:///:memory:` (see `tests/test_db_models.py`, `tests/test_db_queries.py`) or `command.upgrade(cfg, "head")` against a `tmp_path` sqlite file (see `tests/test_db_migrations.py`) — never a real Postgres connection in tests.
- **Never add an AI/model co-authorship line to any commit message** (`CLAUDE.md`, project-wide rule, no exceptions) — every commit step in this plan uses a plain message.
- Do not touch `GcsStorage` — it stays `NotImplementedError` (out of scope, tracked separately).
- Run `uv run pytest -q` (not bare `pytest`) — this repo's env is managed by `uv`.

---

## Task 1: `Storage.open` accepts an optional `encoding` parameter

**Files:**
- Modify: `app/storage/base.py`
- Modify: `app/storage/local.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Produces: `Storage.open(path: str, mode: str = "r", encoding: str | None = None) -> ContextManager[IO]` — the signature `case_builder.py` (Task 2) calls with `encoding="utf-8"` for one file and no `encoding` (defaults to `None`) for the other.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_storage.py`:

```python
def test_open_read_with_explicit_encoding(tmp_path):
    (tmp_path / "c.txt").write_text("café", encoding="utf-8")
    storage = LocalStorage(str(tmp_path))
    with storage.open("c.txt", "r", encoding="utf-8") as f:
        assert f.read() == "café"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_storage.py::test_open_read_with_explicit_encoding -v`
Expected: FAIL with `TypeError: open() got an unexpected keyword argument 'encoding'`

- [ ] **Step 3: Update the `Storage` protocol**

In `app/storage/base.py`, change:

```python
    def open(self, path: str, mode: str = "r") -> ContextManager[IO]: ...
```

to:

```python
    def open(self, path: str, mode: str = "r", encoding: str | None = None) -> ContextManager[IO]: ...
```

- [ ] **Step 4: Update `LocalStorage.open`**

In `app/storage/local.py`, change:

```python
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
```

to:

```python
    @contextmanager
    def open(self, path: str, mode: str = "r", encoding: str | None = None) -> Iterator[IO]:
        p = self._resolve(path)
        if "w" in mode or "a" in mode:
            p.parent.mkdir(parents=True, exist_ok=True)
        f = open(p, mode, encoding=encoding)
        try:
            yield f
        finally:
            f.close()
```

(`encoding=None` is safe to pass even in binary modes — Python's `open()` only rejects a *non-`None`* encoding when the mode is binary.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_storage.py -v`
Expected: all pass, including the new test.

- [ ] **Step 6: Commit**

```bash
git add app/storage/base.py app/storage/local.py tests/test_storage.py
git commit -m "feat: Storage.open accepts an optional encoding parameter"
```

---

## Task 2: Route `case_builder.py`'s two remaining plain `open()` calls through `Storage`

**Files:**
- Modify: `app/pipeline/case_builder.py`
- Test: `tests/test_case_builder_storage.py` (new file)

**Interfaces:**
- Consumes: `Storage.open(path, mode="r", encoding=None)` from Task 1; `app.storage.get_storage(root: str) -> Storage` (already exists, `app/storage/factory.py`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_case_builder_storage.py`:

```python
"""Regression test: case_builder.py must read every input file through
Storage, not plain open() — otherwise a GCS-backed data_dir silently
can't find these files. See docs/superpowers/specs/2026-08-06-ingesta-storage-xm-design.md
section 5."""

from datetime import date
from pathlib import Path

from app.pipeline.case_builder import build_case
from app.schemas.case import DispatchCase, DispatchLevel
from app.schemas.input_pack import InputPack, InputSource
from app.storage.local import LocalStorage

DD = str(Path(__file__).parent / "fixtures" / "xm_smoke")
FECHA = date(2024, 4, 18)


def test_build_case_reads_ramps_and_preideal_map_through_storage(monkeypatch):
    calls = []
    original_open = LocalStorage.open

    def spy_open(self, path, mode="r", encoding=None):
        calls.append(path)
        return original_open(self, path, mode, encoding)

    monkeypatch.setattr(LocalStorage, "open", spy_open)

    case = DispatchCase(dispatch_date=FECHA, level=DispatchLevel.preideal, solver="cbc")
    inputs = InputPack(dispatch_date=FECHA, source=InputSource.historical, data_dir=DD)
    build_case(case, inputs, ders=None)

    assert "ramps.json" in calls
    assert "preideal_dispatch_map.json" in calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_case_builder_storage.py -v`
Expected: FAIL — `calls` is missing `"ramps.json"` and `"preideal_dispatch_map.json"` (they're currently read via plain `open(f"{dd}/...")`, which never touches `LocalStorage.open`).

- [ ] **Step 3: Add the `get_storage` import and instantiate `storage`**

In `app/pipeline/case_builder.py`, add to the imports:

```python
from app.storage import get_storage
```

Find the line `dd = inputs.data_dir` (inside `build_case`) and add immediately after it:

```python
    storage = get_storage(dd)
```

- [ ] **Step 4: Replace the two plain `open()` calls**

Change:

```python
    with open(f"{dd}/preideal_dispatch_map.json", "r", encoding="utf-8") as file:
        preideal_dispatch_map = json.load(file)
```

to:

```python
    with storage.open("preideal_dispatch_map.json", "r", encoding="utf-8") as file:
        preideal_dispatch_map = json.load(file)
```

Change:

```python
    with open(f"{dd}/ramps.json", "r") as file:
        ramps = json.load(file)
```

to:

```python
    with storage.open("ramps.json", "r") as file:
        ramps = json.load(file)
```

- [ ] **Step 5: Run the new test and the full smoke suite**

Run: `uv run pytest tests/test_case_builder_storage.py tests/test_xm_smoke_build_case.py tests/test_xm_smoke_run.py -v`
Expected: all pass — the smoke tests prove behavior is unchanged (fixture's `ramps.json`/`preideal_dispatch_map.json` are both `{}`, so `build_case` output is identical), the new test proves the read now goes through `Storage`.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (140 + 1 new = 141).

- [ ] **Step 7: Commit**

```bash
git add app/pipeline/case_builder.py tests/test_case_builder_storage.py
git commit -m "fix: route ramps.json and preideal_dispatch_map.json reads through Storage"
```

---

## Task 3: `InputDataset` SQLAlchemy model

**Files:**
- Modify: `app/db/models.py`
- Test: `tests/test_db_models.py`

**Interfaces:**
- Produces: `InputDataset` model with columns `id: str`, `dataset: str`, `partition_key: str`, `source: str`, `checksum: str | None`, `row_count: int | None`, `fetched_at: datetime`. Unique constraint on `(dataset, partition_key)`.
- Consumed by: Task 4 (migration must match these columns exactly), Task 5 (`app/db/queries.py`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_db_models.py` (add `import pytest` and `from sqlalchemy.exc import IntegrityError` to the top of the file, alongside the existing imports):

```python
def test_input_dataset_round_trip():
    engine = _memory_engine()
    with Session(engine) as session:
        row = InputDataset(
            dataset="precio_bolsa",
            partition_key="2024",
            source="pydataxm:PrecBolsNaci",
            checksum="abc123",
            row_count=8784,
        )
        session.add(row)
        session.commit()
        session.refresh(row)

        assert row.id
        assert row.fetched_at is not None

        fetched = session.get(InputDataset, row.id)
        assert fetched.dataset == "precio_bolsa"
        assert fetched.partition_key == "2024"
        assert fetched.row_count == 8784


def test_input_dataset_unique_dataset_partition_key():
    engine = _memory_engine()
    with Session(engine) as session:
        session.add(
            InputDataset(dataset="ofertas", partition_key="2024", source="pydataxm:PrecOferDesp")
        )
        session.commit()

        session.add(
            InputDataset(dataset="ofertas", partition_key="2024", source="pydataxm:PrecOferDesp")
        )
        with pytest.raises(IntegrityError):
            session.commit()
```

Also add `InputDataset` to the `from app.db.models import ...` line at the top of the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_models.py::test_input_dataset_round_trip -v`
Expected: FAIL with `ImportError: cannot import name 'InputDataset'`

- [ ] **Step 3: Add the model**

In `app/db/models.py`, add `Integer` and `UniqueConstraint` to the existing `from sqlalchemy import ...` line, then append at the end of the file:

```python
class InputDataset(Base):
    __tablename__ = "input_datasets"
    __table_args__ = (
        UniqueConstraint("dataset", "partition_key", name="uq_input_datasets_dataset_partition_key"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    dataset: Mapped[str] = mapped_column(String, nullable=False)
    partition_key: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    checksum: Mapped[str | None] = mapped_column(String, nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db_models.py -v`
Expected: all pass, including the 2 new tests.

- [ ] **Step 5: Commit**

```bash
git add app/db/models.py tests/test_db_models.py
git commit -m "feat: add InputDataset manifest model"
```

---

## Task 4: Alembic migration for `input_datasets`

**Files:**
- Create: `alembic/versions/0003_input_datasets.py`
- Test: `tests/test_db_migrations.py`

**Interfaces:**
- Consumes: column set from Task 3's `InputDataset` model (must match exactly — Alembic migrations here are hand-written, not autogenerated, per `0001_initial.py`/`0002_run_log_path.py`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_db_migrations.py`:

```python
def test_alembic_upgrade_head_creates_input_datasets_table(tmp_path):
    db_path = tmp_path / "migration_smoke_input_datasets.db"
    database_url = f"sqlite:///{db_path}"

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")

    engine = create_engine(database_url)
    tables = set(inspect(engine).get_table_names())
    assert "input_datasets" in tables

    columns = {c["name"] for c in inspect(engine).get_columns("input_datasets")}
    assert {
        "id",
        "dataset",
        "partition_key",
        "source",
        "checksum",
        "row_count",
        "fetched_at",
    }.issubset(columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_migrations.py::test_alembic_upgrade_head_creates_input_datasets_table -v`
Expected: FAIL — `input_datasets` not in `tables` (migration `0003` doesn't exist yet, `head` is still `0002`).

- [ ] **Step 3: Write the migration**

Create `alembic/versions/0003_input_datasets.py`:

```python
"""add input_datasets manifest table

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-06
"""

import sqlalchemy as sa

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "input_datasets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("partition_key", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("checksum", sa.String(), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "dataset", "partition_key", name="uq_input_datasets_dataset_partition_key"
        ),
    )


def downgrade() -> None:
    op.drop_table("input_datasets")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db_migrations.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add alembic/versions/0003_input_datasets.py tests/test_db_migrations.py
git commit -m "feat: add 0003 migration for input_datasets table"
```

---

## Task 5: `upsert_input_dataset` / `get_input_dataset` query functions

**Files:**
- Modify: `app/db/queries.py`
- Test: `tests/test_db_queries.py`

**Interfaces:**
- Consumes: `InputDataset` model from Task 3.
- Produces: `upsert_input_dataset(session, *, dataset, partition_key, source, checksum=None, row_count=None) -> InputDataset` and `get_input_dataset(session, dataset, partition_key) -> InputDataset | None` — these are the two functions the future fetch-if-missing implementation (issues #20/#21/#23) will call against the manifest.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_db_queries.py`:

```python
def test_upsert_input_dataset_creates_new_row():
    session = _session()
    row = queries.upsert_input_dataset(
        session,
        dataset="precio_bolsa",
        partition_key="2024",
        source="pydataxm:PrecBolsNaci",
        row_count=8784,
    )
    assert row.id
    assert row.fetched_at is not None
    assert row.row_count == 8784


def test_upsert_input_dataset_updates_existing_row_in_place():
    session = _session()
    first = queries.upsert_input_dataset(
        session,
        dataset="precio_bolsa",
        partition_key="2024",
        source="pydataxm:PrecBolsNaci",
        row_count=8784,
    )
    second = queries.upsert_input_dataset(
        session,
        dataset="precio_bolsa",
        partition_key="2024",
        source="pydataxm:PrecBolsNaci",
        row_count=8785,
    )
    assert second.id == first.id
    assert second.row_count == 8785


def test_get_input_dataset_returns_none_when_missing():
    session = _session()
    assert queries.get_input_dataset(session, "precio_bolsa", "2024") is None


def test_get_input_dataset_returns_row_when_present():
    session = _session()
    queries.upsert_input_dataset(
        session, dataset="demaCome", partition_key="2024", source="pydataxm:DemaCome"
    )
    found = queries.get_input_dataset(session, "demaCome", "2024")
    assert found is not None
    assert found.source == "pydataxm:DemaCome"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_queries.py::test_upsert_input_dataset_creates_new_row -v`
Expected: FAIL with `AttributeError: module 'app.db.queries' has no attribute 'upsert_input_dataset'`

- [ ] **Step 3: Add the query functions**

In `app/db/queries.py`, add `InputDataset` to the `from app.db.models import ...` line, then append:

```python
def upsert_input_dataset(
    session: Session,
    *,
    dataset: str,
    partition_key: str,
    source: str,
    checksum: str | None = None,
    row_count: int | None = None,
) -> InputDataset:
    stmt = select(InputDataset).where(
        InputDataset.dataset == dataset, InputDataset.partition_key == partition_key
    )
    existing = session.scalars(stmt).first()
    if existing is not None:
        existing.source = source
        existing.checksum = checksum
        existing.row_count = row_count
        existing.fetched_at = datetime.now(timezone.utc)
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing

    row = InputDataset(
        dataset=dataset,
        partition_key=partition_key,
        source=source,
        checksum=checksum,
        row_count=row_count,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_input_dataset(session: Session, dataset: str, partition_key: str) -> InputDataset | None:
    stmt = select(InputDataset).where(
        InputDataset.dataset == dataset, InputDataset.partition_key == partition_key
    )
    return session.scalars(stmt).first()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db_queries.py -v`
Expected: all pass, including the 4 new tests.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (141 + 2 (Task3) + 4 (Task5) = 147).

- [ ] **Step 6: Commit**

```bash
git add app/db/queries.py tests/test_db_queries.py
git commit -m "feat: add upsert_input_dataset/get_input_dataset manifest queries"
```

---

## Done criteria

- `uv run pytest -q` passes with 147 tests (140 baseline + 7 new).
- `GET`-style manual check: `uv run alembic upgrade head` against a scratch sqlite DB creates `input_datasets` with the exact columns in section 2 of the design doc.
- `case_builder.py` has zero remaining plain `open()` calls for files under `data_dir` root — only the two pre-existing, documented exceptions (`dCondIniP`/`dCondIniU` via `resolve_input`) remain, per `2026-08-04-fase1-cli-completo-design.md`.
- GitHub issues #24 and #25 can be closed, referencing the commits from Task 3/4/5 and Task 1/2 respectively.
