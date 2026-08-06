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
