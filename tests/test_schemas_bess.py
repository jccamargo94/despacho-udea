import pytest
from pydantic import ValidationError

from app.schemas.bess import BessMode, BessScenario, BessUnit


def _unit(**overrides):
    base = dict(
        name="B1",
        mwh_nom=100.0,
        hours_to_deplete=4.0,
        initial_soc=0.5,
        min_soc=0.1,
        max_soc=0.9,
        efficiency=0.92,
    )
    base.update(overrides)
    return BessUnit(**base)


def test_arbitrage_requires_both_bids():
    with pytest.raises(ValidationError, match="charge_bid"):
        BessScenario(
            mode=BessMode.arbitrage,
            penetration_level="10pct",
            units=[_unit(discharge_bid=50.0)],
        )
    with pytest.raises(ValidationError, match="discharge_bid"):
        BessScenario(
            mode=BessMode.arbitrage,
            penetration_level="10pct",
            units=[_unit(charge_bid=20.0)],
        )


def test_arbitrage_with_both_bids_is_valid():
    s = BessScenario(
        mode=BessMode.arbitrage,
        penetration_level="10pct",
        units=[_unit(charge_bid=20.0, discharge_bid=50.0)],
    )
    assert s.units[0].charge_bid == 20.0


def test_generator_requires_discharge_bid_only():
    with pytest.raises(ValidationError, match="discharge_bid"):
        BessScenario(
            mode=BessMode.generator,
            penetration_level="10pct",
            units=[_unit()],
        )
    s = BessScenario(
        mode=BessMode.generator,
        penetration_level="10pct",
        units=[_unit(discharge_bid=50.0)],
    )
    assert s.units[0].discharge_bid == 50.0


def test_grid_asset_does_not_require_bids():
    s = BessScenario(
        mode=BessMode.grid_asset,
        penetration_level="10pct",
        units=[_unit()],
    )
    assert s.units[0].charge_bid is None
