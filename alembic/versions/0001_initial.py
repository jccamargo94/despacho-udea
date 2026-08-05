"""initial schema: scenarios, cases, runs, metric_sets

Revision ID: 0001
Revises:
Create Date: 2026-08-05
"""

import sqlalchemy as sa

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scenarios",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("penetration_level", sa.String(), nullable=False),
        sa.Column("units", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "cases",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("dispatch_date", sa.Date(), nullable=False),
        sa.Column("level", sa.String(), nullable=False),
        sa.Column("solver", sa.String(), nullable=False),
        sa.Column("compute_prices", sa.Boolean(), nullable=False),
        sa.Column("scenario_id", sa.String(), sa.ForeignKey("scenarios.id"), nullable=True),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("case_id", sa.String(), sa.ForeignKey("cases.id"), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("out_dir", sa.String(), nullable=True),
        sa.Column("dispatch_path", sa.String(), nullable=True),
        sa.Column("price_path", sa.String(), nullable=True),
        sa.Column("bess_path", sa.String(), nullable=True),
    )
    op.create_table(
        "metric_sets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("runs.id"), nullable=False, unique=True),
        sa.Column("rmse", sa.Float(), nullable=True),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("bias", sa.Float(), nullable=True),
        sa.Column("wape", sa.Float(), nullable=True),
        sa.Column("smape", sa.Float(), nullable=True),
        sa.Column("r2", sa.Float(), nullable=True),
        sa.Column("bess_charge_mwh", sa.Float(), nullable=True),
        sa.Column("bess_discharge_mwh", sa.Float(), nullable=True),
        sa.Column("bess_avg_soc_mwh", sa.Float(), nullable=True),
        sa.Column("bess_net_revenue", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("metric_sets")
    op.drop_table("runs")
    op.drop_table("cases")
    op.drop_table("scenarios")
