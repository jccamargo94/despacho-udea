import contextlib
import io
import time

from sqlalchemy.orm import Session

from app.db import queries
from app.db.claim import claim_next_pending_run
from app.db.session import get_engine, get_sessionmaker
from app.pipeline.runner import run_case
from app.schemas import BessScenario, DispatchCase, DispatchLevel
from app.storage import get_storage

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
