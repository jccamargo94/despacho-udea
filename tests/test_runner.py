"""Runner orchestration: a failing case must not abort the batch."""

from datetime import date

import pandas as pd

import app.pipeline.runner as runner
from app.pipeline.case_builder import bess_scenario_to_params
from app.schemas import DispatchCase, DispatchLevel
from app.schemas.bess import BessMode, BessScenario, BessUnit


def _toy_case():
    set_data = {
        "G": [],
        "I": ["A", "B"],
        "T": [1],
        "combined_cycle": [],
        "excluded_resource": {},
        "gen_on": [],
        "gen_off": [],
    }
    param_data = {
        "Pmin": {("A", 1): 0.0, ("B", 1): 0.0},
        "Pmax": {("A", 1): 100.0, ("B", 1): 100.0},
        "max_min_op": 0,
        "ramp_up": {},
        "ramp_down": {},
        "beta": {"A": 10.0, "B": 50.0},
        "cold_start": {},
        "demand": {1: 150.0},
        "TMG": {},
        "Ton": {},
        "z_on_t0_minus_1": {},
    }
    return set_data, param_data, {}


def test_failure_isolated(monkeypatch, tmp_path):
    good, bad = date(2024, 4, 18), date(2024, 4, 19)

    def fake_build(case, inputs, **kw):
        if case.dispatch_date == bad:
            raise RuntimeError("boom")
        return _toy_case()

    monkeypatch.setattr(runner, "build_case", fake_build)

    cases = [
        DispatchCase(dispatch_date=good, level=DispatchLevel.preideal, solver="cbc"),
        DispatchCase(dispatch_date=bad, level=DispatchLevel.preideal, solver="cbc"),
    ]
    results = runner.run_many(cases, evaluate=False, out=str(tmp_path))
    assert len(results) == 2
    ok = {r.case.dispatch_date: r.ok for r in results}
    assert ok[good] is True
    assert ok[bad] is False
    bad_r = next(r for r in results if r.case.dispatch_date == bad)
    assert "boom" in bad_r.error


def test_summary_includes_every_ok_row_even_without_metrics(monkeypatch, tmp_path):
    good = date(2024, 4, 18)

    def fake_build(case, inputs, **kw):
        return _toy_case()

    monkeypatch.setattr(runner, "build_case", fake_build)
    case = DispatchCase(dispatch_date=good, level=DispatchLevel.preideal, solver="cbc")
    runner.run_many([case], evaluate=False, out=str(tmp_path))

    summary = pd.read_csv(tmp_path / "metrics-summary.csv")
    assert len(summary) == 1
    assert summary.iloc[0]["scenario"] == "baseline"


def test_summary_scenario_column_and_bess_totals(monkeypatch, tmp_path):
    date_ = date(2024, 4, 18)
    scenario = BessScenario(
        mode=BessMode.arbitrage,
        penetration_level="10pct",
        units=[
            BessUnit(
                name="B1",
                mwh_nom=40.0,
                hours_to_deplete=4.0,
                initial_soc=0.5,
                min_soc=0.1,
                max_soc=0.9,
                efficiency=0.9,
                charge_bid=5.0,
                discharge_bid=45.0,
            )
        ],
    )

    def fake_build(case, inputs, **kw):
        set_data = {
            "G": [],
            "I": ["A"],
            "T": [1],
            "combined_cycle": [],
            "excluded_resource": {},
            "gen_on": [],
            "gen_off": [],
        }
        param_data = {
            "Pmin": {("A", 1): 0.0},
            "Pmax": {("A", 1): 100.0},
            "max_min_op": 0,
            "ramp_up": {},
            "ramp_down": {},
            "beta": {"A": 50.0},
            "cold_start": {},
            "demand": {1: 80.0},
            "TMG": {},
            "Ton": {},
            "z_on_t0_minus_1": {},
        }
        bess_names, bess_params = bess_scenario_to_params(scenario)
        set_data["BESS"] = bess_names
        param_data.update(bess_params)
        return set_data, param_data, {}

    monkeypatch.setattr(runner, "build_case", fake_build)
    case = DispatchCase(
        dispatch_date=date_,
        level=DispatchLevel.preideal,
        bess_scenario=scenario,
        solver="cbc",
    )
    runner.run_many([case], evaluate=False, out=str(tmp_path))

    summary = pd.read_csv(tmp_path / "metrics-summary.csv")
    assert summary.iloc[0]["scenario"] == "10pct"
    assert "bess_net_revenue" in summary.columns
