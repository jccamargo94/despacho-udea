import uuid
from datetime import date as date_
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _new_id() -> str:
    return uuid.uuid4().hex


class Scenario(Base):
    __tablename__ = "scenarios"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    penetration_level: Mapped[str] = mapped_column(String, nullable=False)
    units: Mapped[list] = mapped_column(JSON, nullable=False)
    created_by: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    dispatch_date: Mapped[date_] = mapped_column(Date, nullable=False)
    level: Mapped[str] = mapped_column(String, nullable=False)
    solver: Mapped[str] = mapped_column(String, default="cbc")
    compute_prices: Mapped[bool] = mapped_column(Boolean, default=True)
    scenario_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("scenarios.id"), nullable=True
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    case_id: Mapped[str] = mapped_column(String, ForeignKey("cases.id"), nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    out_dir: Mapped[str | None] = mapped_column(String, nullable=True)
    dispatch_path: Mapped[str | None] = mapped_column(String, nullable=True)
    price_path: Mapped[str | None] = mapped_column(String, nullable=True)
    bess_path: Mapped[str | None] = mapped_column(String, nullable=True)
    log_path: Mapped[str | None] = mapped_column(String, nullable=True)


class MetricSet(Base):
    __tablename__ = "metric_sets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.id"), unique=True, nullable=False)
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    bias: Mapped[float | None] = mapped_column(Float, nullable=True)
    wape: Mapped[float | None] = mapped_column(Float, nullable=True)
    smape: Mapped[float | None] = mapped_column(Float, nullable=True)
    r2: Mapped[float | None] = mapped_column(Float, nullable=True)
    bess_charge_mwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    bess_discharge_mwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    bess_avg_soc_mwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    bess_net_revenue: Mapped[float | None] = mapped_column(Float, nullable=True)


class InputDataset(Base):
    __tablename__ = "input_datasets"
    __table_args__ = (
        UniqueConstraint(
            "dataset", "partition_key", name="uq_input_datasets_dataset_partition_key"
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    dataset: Mapped[str] = mapped_column(String, nullable=False)
    partition_key: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    checksum: Mapped[str | None] = mapped_column(String, nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
