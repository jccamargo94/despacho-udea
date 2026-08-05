from datetime import date

from app.schemas.case import DispatchCase, DispatchLevel
from app.schemas.bess import BessMode, BessScenario, BessUnit


def test_case_without_bess():
    c = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal)
    assert c.bess_scenario is None
    assert c.solver == "cbc"
    assert c.compute_prices is True


def test_case_with_bess_scenario():
    scenario = BessScenario(
        mode=BessMode.grid_asset, penetration_level="10pct",
        units=[BessUnit(
            name="B1", mwh_nom=100.0, hours_to_deplete=4.0, initial_soc=0.5,
            min_soc=0.1, max_soc=0.9, efficiency=0.92,
        )],
    )
    c = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.ideal, bess_scenario=scenario)
    assert c.bess_scenario.mode == BessMode.grid_asset


def test_level_rejects_unknown_value():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DispatchCase(dispatch_date=date(2024, 4, 18), level="bess_ideal_resource")
