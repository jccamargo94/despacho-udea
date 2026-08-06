from sqlalchemy import create_engine, inspect

from alembic import command
from alembic.config import Config


def test_alembic_upgrade_head_creates_all_tables(tmp_path):
    db_path = tmp_path / "migration_smoke.db"
    database_url = f"sqlite:///{db_path}"

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")

    engine = create_engine(database_url)
    tables = set(inspect(engine).get_table_names())
    assert {"scenarios", "cases", "runs", "metric_sets"}.issubset(tables)


def test_alembic_upgrade_head_adds_runs_log_path_column(tmp_path):
    db_path = tmp_path / "migration_smoke_log.db"
    database_url = f"sqlite:///{db_path}"

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")

    engine = create_engine(database_url)
    columns = {c["name"] for c in inspect(engine).get_columns("runs")}
    assert "log_path" in columns


def test_alembic_upgrade_head_creates_input_datasets_table(tmp_path):
    db_path = tmp_path / "migration_smoke_input_datasets.db"
    database_url = f"sqlite:///{db_path}"

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")

    engine = create_engine(database_url)
    tables = set(inspect(engine).get_table_names())
    assert "input_datasets" in tables

    columns = {c["name"] for c in inspect(engine).get_columns("input_datasets")}
    assert {
        "id",
        "dataset",
        "partition_key",
        "source",
        "checksum",
        "row_count",
        "fetched_at",
    }.issubset(columns)
