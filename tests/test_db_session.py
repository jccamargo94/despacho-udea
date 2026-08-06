from sqlalchemy.engine import Engine

from app.db.session import get_engine, get_sessionmaker


def test_get_engine_from_arg():
    engine = get_engine("sqlite:///:memory:")
    assert isinstance(engine, Engine)


def test_get_engine_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    engine = get_engine()
    assert isinstance(engine, Engine)


def test_get_sessionmaker_binds_engine():
    engine = get_engine("sqlite:///:memory:")
    session_factory = get_sessionmaker(engine)
    session = session_factory()
    assert session.bind is engine
    session.close()
