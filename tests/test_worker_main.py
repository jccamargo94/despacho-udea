from datetime import date
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import queries
from app.db.models import Base, Run
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
    assert session.scalars(select(Run)).first() is None


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
    assert Path(updated.out_dir) == Path(results_root) / run.id

    # xm_smoke fixture has no preideal_price actuals -> evaluate is skipped,
    # matching tests/test_xm_smoke_run.py's own assertion
    assert queries.get_metric_set(session, run.id) is None

    # new assertions
    assert updated.log_path is not None
    log_file = Path(updated.log_path)
    assert log_file.exists()
    # xm_smoke fixture has no preideal_price actuals, so run_case's
    # "no XM actuals" branch (app/pipeline/runner.py:48) fires and prints —
    # proof the log actually captured run_case's stdout, not an empty file.
    assert "no XM actuals" in log_file.read_text()


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
    assert updated.log_path is not None
