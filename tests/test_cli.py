from datetime import date

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
