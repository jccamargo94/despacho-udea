"""Postgres-only locking lives here, and only here: `FOR UPDATE SKIP LOCKED`
is what lets more than one worker replica claim rows safely without
stepping on each other. SQLite's dialect doesn't support that clause at
all, so on SQLite (tests, single process) it's simply never added --
there's nothing broken by its absence since SQLite doesn't run concurrent
worker replicas anyway."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Run


def claim_next_pending_run(session: Session) -> Run | None:
    stmt = select(Run).where(Run.status == "pending").order_by(Run.created_at).limit(1)
    if session.bind.dialect.name == "postgresql":
        stmt = stmt.with_for_update(skip_locked=True)
    run = session.scalars(stmt).first()
    if run is None:
        return None
    run.status = "running"
    run.started_at = datetime.now(timezone.utc)
    session.commit()
    return run
