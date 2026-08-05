from enum import Enum

from pydantic import BaseModel, ValidationInfo, field_validator


class BessMode(str, Enum):
    arbitrage = "arbitrage"
    grid_asset = "grid_asset"
    generator = "generator"


class BessUnit(BaseModel):
    name: str
    mwh_nom: float
    hours_to_deplete: float
    initial_soc: float
    min_soc: float
    max_soc: float
    efficiency: float
    charge_bid: float | None = None
    discharge_bid: float | None = None


class BessScenario(BaseModel):
    mode: BessMode
    penetration_level: str
    units: list[BessUnit]

    @field_validator("units")
    @classmethod
    def _check_bids(cls, units: list[BessUnit], info: ValidationInfo) -> list[BessUnit]:
        mode = info.data.get("mode")
        for u in units:
            if mode == BessMode.arbitrage and u.charge_bid is None:
                raise ValueError(f"{u.name}: charge_bid required in mode arbitrage")
            if mode in (BessMode.arbitrage, BessMode.generator) and u.discharge_bid is None:
                raise ValueError(f"{u.name}: discharge_bid required in mode {mode.value}")
        return units
