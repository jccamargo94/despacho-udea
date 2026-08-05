"""Layer-2 check: build_case's own data-assembly logic (name mapping, unit
scaling, CC-empty path, initial-condition parsing) against the fixture,
independent of whether the model actually solves."""

from datetime import date
from pathlib import Path

import pytest

from app.pipeline.case_builder import build_case
from app.schemas.case import DispatchCase, DispatchLevel
from app.schemas.input_pack import InputPack, InputSource

DD = str(Path(__file__).parent / "fixtures" / "xm_smoke")
FECHA = date(2024, 4, 18)


@pytest.fixture
def built():
    case = DispatchCase(dispatch_date=FECHA, level=DispatchLevel.preideal, solver="cbc")
    inputs = InputPack(dispatch_date=FECHA, source=InputSource.historical, data_dir=DD)
    return build_case(case, inputs, ders=None)


def test_sets(built):
    set_data, _, _ = built
    assert sorted(set_data["G"]) == ["TERMO1", "TERMO2"]
    assert sorted(set_data["I"]) == ["TERMO1", "TERMO2"]
    assert list(set_data["gen_on"]) == ["TERMO1"]
    assert list(set_data["gen_off"]) == ["TERMO2"]
    assert set_data["combined_cycle"] == []
    assert len(list(set_data["T"])) == 24


def test_params_scaled_correctly(built):
    _, param_data, _ = built
    ts = date(2024, 4, 18)
    # dispo x1e-3 (kW -> MW): 300_000 kW -> 300 MW, 200_000 kW -> 200 MW
    pmax = param_data["Pmax"]
    for hour in range(24):
        import pandas as pd

        t = pd.Timestamp(ts) + pd.Timedelta(hours=hour)
        assert pmax[("TERMO1", t)] == 300.0
        assert pmax[("TERMO2", t)] == 200.0
    # PrId used raw (MW, no scaling): 350 every hour
    assert set(param_data["demand"].values()) == {350}
    # ofertas x1e3 (COP/kWh -> COP/MWh)
    beta = dict(param_data["beta"])
    assert beta == {"TERMO1": 150000.0, "TERMO2": 180000.0}


def test_cold_start_and_commitment_state(built):
    _, param_data, _ = built
    assert param_data["cold_start"] == {"TERMO1": 1500000.0, "TERMO2": 1500000.0}
    assert dict(param_data["TMG"]) == {"TERMO1": 1, "TERMO2": 1}
    assert dict(param_data["Ton"]) == {"TERMO1": 5}
    assert param_data["z_on_t0_minus_1"] == {"TERMO1": 1}
    assert param_data["ramp_up"] == {}  # ramps.json={}, RU/RD fall to model default=10000
