from fastapi import Depends, FastAPI

from app.db import queries
from app.db.session import get_engine, get_sessionmaker
from app.schemas import BessScenario
from services.api.auth import get_current_user_id

app = FastAPI(title="despacho-udea API")

_engine = None
_session_local = None


def get_session():
    global _engine, _session_local
    if _session_local is None:
        _engine = get_engine()
        _session_local = get_sessionmaker(_engine)
    session = _session_local()
    try:
        yield session
    finally:
        session.close()


@app.post("/scenarios")
def create_scenario(
    scenario: BessScenario,
    user_id: str = Depends(get_current_user_id),
    session=Depends(get_session),
):
    row = queries.create_scenario(session, scenario, created_by=user_id)
    return {"id": row.id}
