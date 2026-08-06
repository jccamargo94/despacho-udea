def test_create_run_returns_pending_status(api_client):
    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    assert "run_id" in body


def test_create_run_rejects_unknown_scenario_id(api_client):
    resp = api_client.post(
        "/runs",
        json={"dispatch_date": "2024-04-18", "level": "preideal", "scenario_id": "missing"},
    )
    assert resp.status_code == 404


def test_get_run_returns_404_for_unknown_id(api_client):
    resp = api_client.get("/runs/does-not-exist")
    assert resp.status_code == 404


def test_get_run_returns_status_and_null_metrics_before_worker_runs(api_client):
    create_resp = api_client.post(
        "/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"}
    )
    run_id = create_resp.json()["run_id"]

    resp = api_client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["status"] == "pending"
    assert body["metrics"] is None


def test_get_run_returns_404_for_another_users_run(api_client):
    from datetime import date

    from app.db import queries

    session = api_client.SessionLocal()
    run = queries.create_case_and_run(
        session,
        dispatch_date=date(2024, 4, 18),
        level="preideal",
        solver="cbc",
        compute_prices=True,
        scenario_id=None,
        user_id="user-2",
    )
    run_id = run.id
    session.close()

    resp = api_client.get(f"/runs/{run_id}")
    assert resp.status_code == 404


def test_list_runs_returns_created_runs(api_client):
    api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    api_client.post("/runs", json={"dispatch_date": "2024-04-19", "level": "ideal"})
    resp = api_client.get("/runs")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_get_run_includes_case_fields(api_client):
    create_resp = api_client.post(
        "/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"}
    )
    run_id = create_resp.json()["run_id"]

    resp = api_client.get(f"/runs/{run_id}")
    body = resp.json()
    assert body["dispatch_date"] == "2024-04-18"
    assert body["level"] == "preideal"
    assert body["scenario_id"] is None


def test_list_runs_includes_case_fields(api_client):
    api_client.post("/runs", json={"dispatch_date": "2024-04-19", "level": "ideal"})
    resp = api_client.get("/runs")
    row = resp.json()[0]
    assert row["dispatch_date"] == "2024-04-19"
    assert row["level"] == "ideal"


def test_get_run_artifacts_all_false_before_worker_runs(api_client):
    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    run_id = resp.json()["run_id"]

    resp = api_client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["artifacts"] == {"dispatch": False, "prices": False, "bess": False}


def test_get_run_artifacts_reflects_available_paths(api_client, tmp_path):
    from datetime import date

    from app.db import queries
    from app.schemas import DispatchCase, DispatchLevel, RunResult

    resp = api_client.post("/runs", json={"dispatch_date": "2024-04-18", "level": "preideal"})
    run_id = resp.json()["run_id"]

    out_dir = tmp_path / "results" / run_id
    out_dir.mkdir(parents=True)

    session = api_client.SessionLocal()
    run = queries.get_run(session, run_id)
    result = RunResult(
        case=DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal),
        ok=True,
        dispatch_path=str(out_dir / "dispatch.csv"),
        price_path=str(out_dir / "price.csv"),
    )
    queries.finish_run_ok(session, run, result, out_dir=str(out_dir))
    session.close()

    resp = api_client.get(f"/runs/{run_id}")
    assert resp.json()["artifacts"] == {"dispatch": True, "prices": True, "bess": False}
