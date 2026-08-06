import os
from datetime import date

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from app.db import queries
from app.db.session import get_engine, get_sessionmaker
from app.schemas import BessScenario, DispatchLevel
from app.storage import get_storage
from services.api.auth import get_current_user_id

app = FastAPI(title="despacho-udea API")

_frontend_origins = [
    origin.strip() for origin in os.environ.get("FRONTEND_ORIGIN", "").split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_frontend_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/scenarios")
def list_scenarios_endpoint(
    user_id: str = Depends(get_current_user_id), session=Depends(get_session)
):
    scenarios = queries.list_scenarios(session)
    return [
        {
            "id": s.id,
            "mode": s.mode,
            "penetration_level": s.penetration_level,
            "units": s.units,
            "created_at": s.created_at,
        }
        for s in scenarios
    ]


class RunCreateRequest(BaseModel):
    dispatch_date: date
    level: DispatchLevel
    solver: str = "cbc"
    compute_prices: bool = True
    scenario_id: str | None = None


def _run_summary(run, case) -> dict:
    return {
        "run_id": run.id,
        "status": run.status,
        "dispatch_date": case.dispatch_date,
        "level": case.level,
        "scenario_id": case.scenario_id,
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
    return [_run_summary(r, queries.get_case(session, r.case_id)) for r in runs]


@app.get("/runs/{run_id}")
def get_run_detail(
    run_id: str, user_id: str = Depends(get_current_user_id), session=Depends(get_session)
):
    run = _get_owned_run(session, run_id, user_id)
    case = queries.get_case(session, run.case_id)
    metric_set = queries.get_metric_set(session, run.id)
    out = _run_summary(run, case)
    out["metrics"] = (
        {
            "rmse": metric_set.rmse,
            "mae": metric_set.mae,
            "bias": metric_set.bias,
            "wape": metric_set.wape,
            "smape": metric_set.smape,
            "r2": metric_set.r2,
            "bess_charge_mwh": metric_set.bess_charge_mwh,
            "bess_discharge_mwh": metric_set.bess_discharge_mwh,
            "bess_avg_soc_mwh": metric_set.bess_avg_soc_mwh,
            "bess_net_revenue": metric_set.bess_net_revenue,
        }
        if metric_set
        else None
    )
    out["artifacts"] = {
        "dispatch": run.dispatch_path is not None,
        "prices": run.price_path is not None,
        "bess": run.bess_path is not None,
    }
    return out


@app.get("/runs/{run_id}/log")
def get_run_log(
    run_id: str, user_id: str = Depends(get_current_user_id), session=Depends(get_session)
):
    run = _get_owned_run(session, run_id, user_id)
    if run.log_path is None or not get_storage(".").exists(run.log_path):
        raise HTTPException(status_code=404, detail="run has no log yet")
    with get_storage(".").open(run.log_path) as f:
        content = f.read()
    return PlainTextResponse(content)


_ARTIFACT_PATHS = {
    "dispatch": "dispatch_path",
    "prices": "price_path",
    "bess": "bess_path",
}


def _artifact_path(run, artifact: str) -> str:
    if artifact not in _ARTIFACT_PATHS:
        raise HTTPException(status_code=404, detail="unknown artifact")
    path = getattr(run, _ARTIFACT_PATHS[artifact])
    if path is None:
        raise HTTPException(status_code=404, detail=f"run has no {artifact} artifact yet")
    if not get_storage(".").exists(path):
        raise HTTPException(status_code=404, detail="artifact file missing on disk")
    return path


@app.get("/runs/{run_id}/{artifact}")
def get_run_artifact(
    run_id: str,
    artifact: str,
    user_id: str = Depends(get_current_user_id),
    session=Depends(get_session),
):
    run = _get_owned_run(session, run_id, user_id)
    path = _artifact_path(run, artifact)
    with get_storage(".").open(path) as f:
        df = pd.read_csv(f)
    return df.to_dict(orient="records")


@app.get("/runs/{run_id}/download/{artifact}")
def download_run_artifact(
    run_id: str,
    artifact: str,
    user_id: str = Depends(get_current_user_id),
    session=Depends(get_session),
):
    run = _get_owned_run(session, run_id, user_id)
    path = _artifact_path(run, artifact)
    return FileResponse(path)
