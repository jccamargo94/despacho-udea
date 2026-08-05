from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel


class InputSource(str, Enum):
    historical = "historical"
    live = "live"
    forecast = "forecast"


class InputPack(BaseModel):
    dispatch_date: date
    source: InputSource
    data_dir: str
    checksum: str | None = None
    downloaded_at: datetime | None = None
