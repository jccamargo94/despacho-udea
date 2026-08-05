from pydantic import BaseModel

from app.schemas.case import DispatchCase


class RunResult(BaseModel):
    case: DispatchCase
    ok: bool
    dispatch_path: str | None = None
    price_path: str | None = None
    bess_path: str | None = None
    bess_summary: dict[str, float] | None = None
    metrics_path: str | None = None
    metrics: dict[str, float] | None = None
    error: str | None = None
