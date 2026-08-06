from datetime import date

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from app.db import queries
from app.db.session import get_engine, get_sessionmaker
from app.schemas import BessScenario, DispatchLevel
from services.api.auth import get_current_user_id

app = FastAPI(title="despacho-udea API")

_engine = None
_session_local = None


def get_session():
    global _engine, _session_local
    if _session_local is None:
        _engine = get_engine()
        _session_local = get_sessionmaker(_engine)
    session = _session_local()
    try:
        yield session
    finally:
        session.close()


@app.post("/scenarios")
def create_scenario(
    scenario: BessScenario,
    user_id: str = Depends(get_current_user_id),
    session=Depends(get_session),
):
    row = queries.create_scenario(session, scenario, created_by=user_id)
    return {"id": row.id}


class RunCreateRequest(BaseModel):
    dispatch_date: date
    level: DispatchLevel
    solver: str = "cbc"
    compute_prices: bool = True
    scenario_id: str | None = None


def _run_summary(run) -> dict:
    return {
        "run_id": run.id,
        "status": run.status,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "error": run.error,
    }


def _get_owned_run(session, run_id: str, user_id: str):
    run = queries.get_run(session, run_id)
    if run is None or run.user_id != user_id:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.post("/runs")
def create_run(
    body: RunCreateRequest,
    user_id: str = Depends(get_current_user_id),
    session=Depends(get_session),
):
    if body.scenario_id is not None and queries.get_scenario(session, body.scenario_id) is None:
        raise HTTPException(status_code=404, detail="scenario not found")
    run = queries.create_case_and_run(
        session,
        dispatch_date=body.dispatch_date,
        level=body.level.value,
        solver=body.solver,
        compute_prices=body.compute_prices,
        scenario_id=body.scenario_id,
        user_id=user_id,
    )
    return {"run_id": run.id, "status": run.status}


@app.get("/runs")
def list_runs(user_id: str = Depends(get_current_user_id), session=Depends(get_session)):
    runs = queries.list_runs_for_user(session, user_id)
    return [_run_summary(r) for r in runs]


@app.get("/runs/{run_id}")
def get_run_detail(
    run_id: str, user_id: str = Depends(get_current_user_id), session=Depends(get_session)
):
    run = _get_owned_run(session, run_id, user_id)
    metric_set = queries.get_metric_set(session, run.id)
    out = _run_summary(run)
    out["metrics"] = (
        {
            "rmse": metric_set.rmse,
            "mae": metric_set.mae,
            "bias": metric_set.bias,
            "wape": metric_set.wape,
            "smape": metric_set.smape,
            "r2": metric_set.r2,
        }
        if metric_set
        else None
    )
    return out
