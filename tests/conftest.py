import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient

    from app.db.models import Base
    from services.api.auth import get_current_user_id
    from services.api.main import app, get_session

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    test_session_factory = sessionmaker(bind=engine)

    def _override_session():
        session = test_session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    client = TestClient(app)
    client.SessionLocal = test_session_factory
    yield client
    app.dependency_overrides.clear()
