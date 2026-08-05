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
