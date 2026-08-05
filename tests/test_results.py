"""Results extraction tested against a tiny solvable model.

Cheap gen A (beta=10, Pmax=100), expensive B (beta=50). demand=150 -> A=100,
B=50; marginal unit is B so MPO should equal B's cost (50).
"""
from datetime import date

import pandas as pd

from app.model.model import UnitCommitmentModel
from app.pipeline.case_builder import bess_scenario_to_params
from app.schemas import DispatchCase, DispatchLevel
from app.schemas.bess import BessMode, BessScenario, BessUnit
from app.pipeline.results import extract_mpo, extract_dispatch, extract_bess, save_results


def _toy_model():
    set_data = {
        "G": [], "I": ["A", "B"], "T": [1], "combined_cycle": [],
        "excluded_resource": {}, "gen_on": [], "gen_off": [],
    }
    param_data = {
        "Pmin": {("A", 1): 0.0, ("B", 1): 0.0},
        "Pmax": {("A", 1): 100.0, ("B", 1): 100.0},
        "max_min_op": 0, "ramp_up": {}, "ramp_down": {},
        "beta": {"A": 10.0, "B": 50.0}, "cold_start": {},
        "demand": {1: 150.0}, "TMG": {}, "Ton": {}, "z_on_t0_minus_1": {},
    }
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    m = UnitCommitmentModel(case=case)
    m.create_model(set_data=set_data, param_data=param_data)
    m.solve(solver="cbc")
    return m, case


def test_extract_mpo_is_marginal_cost():
    m, _ = _toy_model()
    mpo = extract_mpo(m)
    assert len(mpo) == 1
    assert abs(list(mpo.values())[0] - 50.0) < 1e-6


def test_extract_dispatch_rows():
    m, _ = _toy_model()
    df = extract_dispatch(m)
    assert set(df.columns) == {"generador", "datetime", "dispatch"}
    by_gen = df.set_index("generador")["dispatch"].to_dict()
    assert abs(by_gen["A"] - 100.0) < 1e-6
    assert abs(by_gen["B"] - 50.0) < 1e-6


def test_save_results_writes_csvs(tmp_path):
    m, case = _toy_model()
    result = save_results(m, case, out=str(tmp_path))
    assert (tmp_path / f"dispatch_by_gen-{case.dispatch_date}-{case.level.value}.csv").exists()
    assert (tmp_path / f"marginal_price-{case.dispatch_date}-{case.level.value}.csv").exists()
    assert result.ok is True
    assert result.dispatch_path is not None


def _bess_case_and_model():
    ts = [1]
    set_data = {
        "G": [], "I": ["A"], "T": ts, "combined_cycle": [],
        "excluded_resource": {}, "gen_on": [], "gen_off": [],
    }
    param_data = {
        "Pmin": {("A", 1): 0.0},
        "Pmax": {("A", 1): 100.0},
        "max_min_op": 0, "ramp_up": {}, "ramp_down": {},
        "beta": {"A": 50.0}, "cold_start": {},
        "demand": {1: 80.0}, "TMG": {}, "Ton": {}, "z_on_t0_minus_1": {},
    }
    scenario = BessScenario(
        mode=BessMode.arbitrage, penetration_level="test",
        units=[BessUnit(
            name="B1", mwh_nom=40.0, hours_to_deplete=4.0, initial_soc=0.5,
            min_soc=0.1, max_soc=0.9, efficiency=0.9,
            charge_bid=5.0, discharge_bid=45.0,
        )],
    )
    bess_names, bess_params = bess_scenario_to_params(scenario)
    set_data["BESS"] = bess_names
    param_data.update(bess_params)

    case = DispatchCase(
        dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal,
        bess_scenario=scenario, solver="cbc",
    )
    m = UnitCommitmentModel(case=case)
    m.create_model(set_data=set_data, param_data=param_data)
    m.solve(solver="cbc")
    return m, case


def test_save_results_writes_bess_csv_and_summary(tmp_path):
    m, case = _bess_case_and_model()
    result = save_results(m, case, out=str(tmp_path))

    assert result.bess_path is not None
    bess_csv = tmp_path / f"bess_results-{case.dispatch_date}-{case.level.value}.csv"
    assert bess_csv.exists()

    df = pd.read_csv(bess_csv)
    assert set(df.columns) == {"unit", "datetime", "charge", "discharge", "soc", "revenue", "cost"}

    mpo = extract_mpo(m)
    price = list(mpo.values())[0]
    row = df.iloc[0]
    # toy scenario's optimum is discharge-only (demand=80 > gen A alone would
    # cover, BESS discharges to help meet it); pin that precondition so a
    # future change to demand/bids that flips it to charge=0/discharge=0
    # can't silently turn the formula checks below into 0==0 tautologies.
    assert row["discharge"] > 0
    assert row["charge"] == 0
    assert abs(row["revenue"] - row["discharge"] * price * 1000.0) < 1e-6
    assert abs(row["cost"] - row["charge"] * price * 1000.0) < 1e-6

    assert result.bess_summary is not None
    assert abs(result.bess_summary["bess_net_revenue"] - (df["revenue"] - df["cost"]).sum()) < 1e-6


def test_extract_bess_cost_formula_with_nonzero_charge():
    """test_save_results_writes_bess_csv_and_summary's toy LP always lands on
    charge=0, so its cost assertion never actually exercises the cost
    formula. Call extract_bess directly with a stubbed model where charge is
    nonzero to cover that branch of the formula for real."""
    class _Inner:
        bess_charge = {("B1", 1): 5.0}
        bess_discharge = {("B1", 1): 0.0}
        soc_bess = {("B1", 1): 20.0}

    class _Model:
        _model = _Inner()

    mpo = {1: 42.0}
    df = extract_bess(_Model(), mpo)
    row = df.iloc[0]
    assert row["charge"] == 5.0
    assert row["revenue"] == 0.0
    assert abs(row["cost"] - 5.0 * 42.0 * 1000.0) < 1e-6


def test_save_results_without_bess_scenario_has_no_bess_fields(tmp_path):
    m, case = _toy_model()
    result = save_results(m, case, out=str(tmp_path))
    assert result.bess_path is None
    assert result.bess_summary is None
