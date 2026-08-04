"""UnitCommitmentModel dispatch-case branching: level -> thermal constraints,
BESS mode -> objective choice / NotImplementedError guard.

Uses the same tiny 2-generator toy fixture as test_results.py/test_runner.py,
extended with one BESS unit for the BESS-mode tests.
"""
from datetime import date

import pytest
from pyomo.core.expr import identify_variables

from app.model.model import UnitCommitmentModel
from app.schemas.case import DispatchCase, DispatchLevel
from app.schemas.bess import BessMode, BessScenario, BessUnit


def _toy_sets_and_params():
    set_data = {
        "G": [], "I": ["A", "B"], "T": [1], "combined_cycle": [],
        "excluded_resource": {}, "gen_on": [], "gen_off": [],
    }
    param_data = {
        "Pmin": {("A", 1): 0.0, ("B", 1): 0.0},
        "Pmax": {("A", 1): 100.0, ("B", 1): 100.0},
        "max_min_op": 0, "ramp_up": {}, "ramp_down": {},
        "beta": {"A": 10.0, "B": 50.0}, "cold_start": {},
        "demand": {1: 130.0}, "TMG": {}, "Ton": {}, "z_on_t0_minus_1": {},
    }
    return set_data, param_data


def _toy_bess_params():
    return {
        "BESS": ["B1"],
        "bess_soc_0": {"B1": 50.0}, "bess_charge_bid": {"B1": 5.0},
        "bess_discharge_bid": {"B1": 60.0}, "bess_min_soc": {"B1": 10.0},
        "bess_max_soc": {"B1": 90.0}, "efficiency": {"B1": 0.9},
        "bess_max_charge": {"B1": 25.0}, "bess_max_discharge": {"B1": 25.0},
    }


def test_ideal_level_adds_thermal_constraints():
    set_data, param_data = _toy_sets_and_params()
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.ideal)
    m = UnitCommitmentModel(case=case)
    m.create_model(set_data=set_data, param_data=param_data)
    assert hasattr(m._model, "up_ramps_thermal_gen")


def test_preideal_level_skips_thermal_constraints():
    set_data, param_data = _toy_sets_and_params()
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    m = UnitCommitmentModel(case=case)
    m.create_model(set_data=set_data, param_data=param_data)
    assert not hasattr(m._model, "up_ramps_thermal_gen")


def test_bess_ideal_resource_still_gets_thermal_constraints():
    """Regression check for the fixed bug: BESS grid_asset + ideal must get
    thermal constraints, which the old string-matching logic skipped."""
    set_data, param_data = _toy_sets_and_params()
    set_data.update(BESS=["B1"])
    param_data.update(_toy_bess_params())
    scenario = BessScenario(
        mode=BessMode.grid_asset, penetration_level="10pct",
        units=[BessUnit(name="B1", mwh_nom=100.0, hours_to_deplete=4.0,
                         initial_soc=0.5, min_soc=0.1, max_soc=0.9, efficiency=0.9)],
    )
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.ideal, bess_scenario=scenario)
    m = UnitCommitmentModel(case=case)
    m.create_model(set_data=set_data, param_data=param_data)
    assert hasattr(m._model, "up_ramps_thermal_gen")


def test_grid_asset_mode_uses_resource_objective():
    set_data, param_data = _toy_sets_and_params()
    set_data.update(BESS=["B1"])
    param_data.update(_toy_bess_params())
    scenario = BessScenario(
        mode=BessMode.grid_asset, penetration_level="10pct",
        units=[BessUnit(name="B1", mwh_nom=100.0, hours_to_deplete=4.0,
                         initial_soc=0.5, min_soc=0.1, max_soc=0.9, efficiency=0.9)],
    )
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal, bess_scenario=scenario)
    m = UnitCommitmentModel(case=case)
    m.create_model(set_data=set_data, param_data=param_data)
    # maximize_social_welfare_as_resource drops the bess_charge/bess_discharge
    # bid terms from the objective (see app/model/constraints/bess/soc.py);
    # maximize_social_welfare includes them. Both docstrings render to the
    # identical "Maximize social welfare" text, so `.doc` cannot distinguish
    # the two rules -- check which variables actually entered the expression.
    objective_vars = {v.name for v in identify_variables(m._model.objective.expr)}
    assert "bess_charge[B1,1]" not in objective_vars
    assert "bess_discharge[B1,1]" not in objective_vars


def test_generator_mode_raises_not_implemented():
    set_data, param_data = _toy_sets_and_params()
    set_data.update(BESS=["B1"])
    param_data.update(_toy_bess_params())
    scenario = BessScenario(
        mode=BessMode.generator, penetration_level="10pct",
        units=[BessUnit(name="B1", mwh_nom=100.0, hours_to_deplete=4.0,
                         initial_soc=0.5, min_soc=0.1, max_soc=0.9, efficiency=0.9,
                         discharge_bid=60.0)],
    )
    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal, bess_scenario=scenario)
    m = UnitCommitmentModel(case=case)
    with pytest.raises(NotImplementedError, match="generator"):
        m.create_model(set_data=set_data, param_data=param_data)
