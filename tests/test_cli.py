from datetime import date

import pandas as pd
from typer.testing import CliRunner

import app.cli as cli
from app.schemas import DispatchCase, DispatchLevel, RunResult

runner = CliRunner()


def _stub_dates(monkeypatch):
    monkeypatch.setattr(cli, "_available_dates", lambda data_dir: [date(2024, 4, 18)])


def test_run_success(monkeypatch):
    _stub_dates(monkeypatch)
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    monkeypatch.setattr(
        cli, "run_many",
        lambda *a, **k: [RunResult(case=case, ok=True)],
    )
    result = runner.invoke(cli.app, ["run", "2024-04-18", "-t", "preideal"])
    assert result.exit_code == 0
    assert "1 ok, 0 failed" in result.output


def test_run_reports_failure(monkeypatch):
    _stub_dates(monkeypatch)
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    monkeypatch.setattr(
        cli, "run_many",
        lambda *a, **k: [RunResult(case=case, ok=False, error="X")],
    )
    result = runner.invoke(cli.app, ["run", "2024-04-18"])
    assert result.exit_code == 1
    assert "1 failed" in result.output


def test_no_dates_selected(monkeypatch):
    monkeypatch.setattr(cli, "_available_dates", lambda data_dir: [])
    result = runner.invoke(cli.app, ["run", "2024-05-01:2024-05-02"])
    assert result.exit_code == 1
    assert "No dates selected" in result.output


def test_available_dates_reads_condicion_inicial_dirs(tmp_path):
    root = tmp_path / "condicion_inicial"
    root.mkdir()
    (root / "2024-04-18").mkdir()
    (root / "2024-04-19").mkdir()
    (root / "notes.txt").write_text("not a date dir")
    dates = cli._available_dates(str(tmp_path))
    assert sorted(dates) == [date(2024, 4, 18), date(2024, 4, 19)]


def test_run_loads_bess_scenario(monkeypatch):
    _stub_dates(monkeypatch)
    captured = {}

    def fake_run_many(cases, **k):
        captured["cases"] = cases
        return [RunResult(case=c, ok=True) for c in cases]

    monkeypatch.setattr(cli, "run_many", fake_run_many)
    result = runner.invoke(
        cli.app, ["run", "2024-04-18", "-t", "preideal", "--bess-scenario", "20pct_arbitrage"]
    )
    assert result.exit_code == 0
    case = captured["cases"][0]
    assert case.bess_scenario is not None
    assert case.bess_scenario.penetration_level == "20pct"


def test_run_without_bess_scenario_flag_has_none(monkeypatch):
    _stub_dates(monkeypatch)
    captured = {}

    def fake_run_many(cases, **k):
        captured["cases"] = cases
        return [RunResult(case=c, ok=True) for c in cases]

    monkeypatch.setattr(cli, "run_many", fake_run_many)
    runner.invoke(cli.app, ["run", "2024-04-18", "-t", "preideal"])
    assert captured["cases"][0].bess_scenario is None


def test_fetch_calls_ensure_data_for_date_for_each_date_in_range(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli, "ensure_data_for_date", lambda d, data_dir: calls.append(d)
    )
    result = runner.invoke(cli.app, ["fetch", "2024-04-18:2024-04-19"])
    assert result.exit_code == 0
    assert calls == [date(2024, 4, 18), date(2024, 4, 19)]


def test_fetch_single_date(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli, "ensure_data_for_date", lambda d, data_dir: calls.append(d)
    )
    result = runner.invoke(cli.app, ["fetch", "2024-04-18"])
    assert result.exit_code == 0
    assert calls == [date(2024, 4, 18)]


def test_fetch_month_covers_every_day_including_leap_day(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli, "ensure_data_for_date", lambda d, data_dir: calls.append(d)
    )
    result = runner.invoke(cli.app, ["fetch", "2024-02"])
    assert result.exit_code == 0
    assert len(calls) == 29
    assert calls[0] == date(2024, 2, 1)
    assert calls[-1] == date(2024, 2, 29)


def test_fetch_isolates_per_date_failures(monkeypatch):
    calls = []

    def _fake(d, data_dir):
        calls.append(d)
        if d == date(2024, 4, 19):
            raise IndexError("XM hasn't published this file yet")

    monkeypatch.setattr(cli, "ensure_data_for_date", _fake)
    result = runner.invoke(cli.app, ["fetch", "2024-04-18:2024-04-20"])
    assert result.exit_code == 0
    assert calls == [date(2024, 4, 18), date(2024, 4, 19), date(2024, 4, 20)]
    assert "fetched 2/3 date(s), 1 failed" in result.output


def test_evaluate_command(monkeypatch):
    _stub_dates(monkeypatch)
    monkeypatch.setattr(cli, "evaluate_saved_run", lambda d, lvl, **k: {"mae": 1.0})
    result = runner.invoke(cli.app, ["evaluate", "2024-04-18", "-t", "preideal"])
    assert result.exit_code == 0
    assert "1 run(s) evaluated" in result.output


def test_evaluate_command_reports_missing_runs(monkeypatch):
    _stub_dates(monkeypatch)

    def _raise(d, lvl, **k):
        raise FileNotFoundError("no saved price CSV")

    monkeypatch.setattr(cli, "evaluate_saved_run", _raise)
    result = runner.invoke(cli.app, ["evaluate", "2024-04-18", "-t", "preideal"])
    assert result.exit_code == 1
    assert "No runs evaluated" in result.output


def test_compare_outer_joins_summaries_on_date_type_scenario(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    pd.DataFrame([
        {"date": "2024-04-18", "type": "preideal", "scenario": "baseline", "mae": 1.0},
    ]).to_csv(a / "metrics-summary.csv", index=False)
    pd.DataFrame([
        {"date": "2024-04-18", "type": "preideal", "scenario": "baseline", "mae": 2.0},
        {"date": "2024-04-19", "type": "preideal", "scenario": "baseline", "mae": 3.0},
    ]).to_csv(b / "metrics-summary.csv", index=False)

    result = runner.invoke(cli.app, ["compare", str(a), str(b)])
    assert result.exit_code == 0
    assert "2024-04-19" in result.output
    assert "NaN" in result.output
