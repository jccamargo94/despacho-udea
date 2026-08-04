from datetime import date
from enum import Enum

from pydantic import BaseModel

from app.schemas.bess import BessScenario


class DispatchLevel(str, Enum):
    preideal = "preideal"
    ideal = "ideal"


class DispatchCase(BaseModel):
    dispatch_date: date
    level: DispatchLevel
    bess_scenario: BessScenario | None = None
    solver: str = "cbc"
    compute_prices: bool = True
