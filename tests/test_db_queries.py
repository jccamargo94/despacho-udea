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
