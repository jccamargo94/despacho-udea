from datetime import date

from app.schemas import DispatchCase, DispatchLevel, RunResult


def test_run_result_ok():
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    r = RunResult(case=case, ok=True, dispatch_path="a.csv", price_path="b.csv",
                   metrics={"mae": 1.2})
    assert r.ok is True
    assert r.metrics["mae"] == 1.2
    assert r.error is None


def test_run_result_failure():
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.ideal)
    r = RunResult(case=case, ok=False, error="RuntimeError: boom")
    assert r.ok is False
    assert r.dispatch_path is None


def test_run_result_bess_fields_default_none():
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    r = RunResult(case=case, ok=True)
    assert r.bess_path is None
    assert r.bess_summary is None


def test_run_result_bess_fields_settable():
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    r = RunResult(
        case=case, ok=True, bess_path="bess.csv",
        bess_summary={"bess_net_revenue": 100.0},
    )
    assert r.bess_path == "bess.csv"
    assert r.bess_summary["bess_net_revenue"] == 100.0
