import pytest

from app.pipeline.scenarios import load_bess_scenario
from app.schemas.bess import BessMode


def test_loads_named_scenario_from_library():
    scenario = load_bess_scenario("20pct_arbitrage")
    assert scenario.mode == BessMode.arbitrage
    assert scenario.penetration_level == "20pct"
    assert scenario.units[0].name == "BESS1"
    assert scenario.units[0].charge_bid == 20.0


def test_loads_grid_asset_named_scenario():
    scenario = load_bess_scenario("10pct_grid_asset")
    assert scenario.mode == BessMode.grid_asset
    assert scenario.units[0].charge_bid is None


def test_loads_literal_path(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "mode: grid_asset\n"
        "penetration_level: custom\n"
        "units:\n"
        "  - name: B1\n"
        "    mwh_nom: 50.0\n"
        "    hours_to_deplete: 2.0\n"
        "    initial_soc: 0.5\n"
        "    min_soc: 0.0\n"
        "    max_soc: 1.0\n"
        "    efficiency: 0.9\n"
    )
    scenario = load_bess_scenario(str(path))
    assert scenario.penetration_level == "custom"


def test_unknown_name_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_bess_scenario("does-not-exist-anywhere")
