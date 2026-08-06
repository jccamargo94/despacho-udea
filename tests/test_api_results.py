from datetime import date

import pandas as pd

from app.db import queries
from app.schemas import DispatchCase, DispatchLevel, RunResult


def _seed_done_run_with_dispatch_csv(api_client, tmp_path):
    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    run_id = resp.json()["run_id"]

    out_dir = tmp_path / "results" / run_id
    out_dir.mkdir(parents=True)
    dispatch_csv = out_dir / "dispatch_by_gen-2024-04-18-preideal.csv"
    pd.DataFrame(
        [{"generador": "TERMO1", "datetime": "2024-04-18 00:00", "dispatch": 300.0}]
    ).to_csv(dispatch_csv, index=False)

    session = api_client.SessionLocal()
    run = queries.get_run(session, run_id)
    result = RunResult(
        case=DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal),
        ok=True,
        dispatch_path=str(dispatch_csv),
    )
    queries.finish_run_ok(session, run, result, out_dir=str(out_dir))
    session.close()

    return run_id


def test_get_run_dispatch_returns_json_rows(api_client, tmp_path):
    run_id = _seed_done_run_with_dispatch_csv(api_client, tmp_path)

    resp = api_client.get(f"/runs/{run_id}/dispatch")
    assert resp.status_code == 200
    rows = resp.json()
    assert rows[0]["generador"] == "TERMO1"
    assert rows[0]["dispatch"] == 300.0


def test_download_run_dispatch_returns_csv_file(api_client, tmp_path):
    run_id = _seed_done_run_with_dispatch_csv(api_client, tmp_path)

    resp = api_client.get(f"/runs/{run_id}/download/dispatch")
    assert resp.status_code == 200
    assert "TERMO1" in resp.text


def test_get_run_artifact_404_when_run_has_no_artifact_yet(api_client):
    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    run_id = resp.json()["run_id"]

    resp = api_client.get(f"/runs/{run_id}/dispatch")
    assert resp.status_code == 404


def test_get_run_artifact_404_for_unknown_artifact_name(api_client, tmp_path):
    run_id = _seed_done_run_with_dispatch_csv(api_client, tmp_path)

    resp = api_client.get(f"/runs/{run_id}/not-a-real-artifact")
    assert resp.status_code == 404


def test_download_run_artifact_404_when_file_missing_on_disk(api_client, tmp_path):
    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    run_id = resp.json()["run_id"]

    out_dir = tmp_path / "results" / run_id
    out_dir.mkdir(parents=True)
    dispatch_csv = out_dir / "dispatch_by_gen-2024-04-18-preideal.csv"  # never written

    session = api_client.SessionLocal()
    run = queries.get_run(session, run_id)
    result = RunResult(
        case=DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal),
        ok=True,
        dispatch_path=str(dispatch_csv),
    )
    queries.finish_run_ok(session, run, result, out_dir=str(out_dir))
    session.close()

    resp = api_client.get(f"/runs/{run_id}/download/dispatch")
    assert resp.status_code == 404
