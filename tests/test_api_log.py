from datetime import date

from app.db import queries
from app.schemas import DispatchCase, DispatchLevel, RunResult


def _seed_done_run_with_log(api_client, tmp_path):
    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    run_id = resp.json()["run_id"]

    out_dir = tmp_path / "results" / run_id
    out_dir.mkdir(parents=True)
    log_file = out_dir / "run.log"
    log_file.write_text("==> 2024-04-18 [preideal]\n")

    session = api_client.SessionLocal()
    run = queries.get_run(session, run_id)
    result = RunResult(
        case=DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal),
        ok=True,
    )
    queries.finish_run_ok(session, run, result, out_dir=str(out_dir))
    run = queries.get_run(session, run_id)
    run.log_path = str(log_file)
    session.add(run)
    session.commit()
    session.close()

    return run_id


def test_get_run_log_returns_text(api_client, tmp_path):
    run_id = _seed_done_run_with_log(api_client, tmp_path)

    resp = api_client.get(f"/runs/{run_id}/log")
    assert resp.status_code == 200
    assert "2024-04-18" in resp.text


def test_get_run_log_404_when_no_log_yet(api_client):
    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    run_id = resp.json()["run_id"]

    resp = api_client.get(f"/runs/{run_id}/log")
    assert resp.status_code == 404
