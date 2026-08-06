from datetime import date as date_
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Case, InputDataset, MetricSet, Run, Scenario
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


def list_scenarios(session: Session) -> list[Scenario]:
    stmt = select(Scenario).order_by(Scenario.created_at.desc())
    return list(session.scalars(stmt))


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
