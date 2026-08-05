from app.pipeline.case_builder import bess_scenario_to_params
from app.schemas.bess import BessMode, BessScenario, BessUnit


def test_maps_units_to_pyomo_param_dicts():
    scenario = BessScenario(
        mode=BessMode.arbitrage,
        penetration_level="10pct",
        units=[
            BessUnit(
                name="B1",
                mwh_nom=100.0,
                hours_to_deplete=4.0,
                initial_soc=0.5,
                min_soc=0.1,
                max_soc=0.9,
                efficiency=0.92,
                charge_bid=20.0,
                discharge_bid=60.0,
            ),
            BessUnit(
                name="B2",
                mwh_nom=50.0,
                hours_to_deplete=2.0,
                initial_soc=1.0,
                min_soc=0.0,
                max_soc=1.0,
                efficiency=0.85,
                charge_bid=15.0,
                discharge_bid=55.0,
            ),
        ],
    )
    names, params = bess_scenario_to_params(scenario)

    assert names == ["B1", "B2"]
    assert params["bess_soc_0"] == {"B1": 50.0, "B2": 50.0}
    assert params["bess_min_soc"] == {"B1": 10.0, "B2": 0.0}
    assert params["bess_max_soc"] == {"B1": 90.0, "B2": 50.0}
    assert params["bess_max_charge"] == {"B1": 25.0, "B2": 25.0}
    assert params["bess_max_discharge"] == {"B1": 25.0, "B2": 25.0}
    assert params["efficiency"] == {"B1": 0.92, "B2": 0.85}
    assert params["bess_charge_bid"] == {"B1": 20.0, "B2": 15.0}
    assert params["bess_discharge_bid"] == {"B1": 60.0, "B2": 55.0}


def test_grid_asset_scenario_omits_absent_bids():
    scenario = BessScenario(
        mode=BessMode.grid_asset,
        penetration_level="10pct",
        units=[
            BessUnit(
                name="B1",
                mwh_nom=100.0,
                hours_to_deplete=4.0,
                initial_soc=0.5,
                min_soc=0.1,
                max_soc=0.9,
                efficiency=0.9,
            )
        ],
    )
    _, params = bess_scenario_to_params(scenario)
    assert params["bess_charge_bid"] == {}
    assert params["bess_discharge_bid"] == {}
