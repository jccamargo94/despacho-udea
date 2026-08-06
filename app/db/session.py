import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


def get_engine(database_url: str | None = None):
    url = database_url or os.environ["DATABASE_URL"]
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


def get_sessionmaker(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine)
