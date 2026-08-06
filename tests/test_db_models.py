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
