"""add input_datasets manifest table

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-06
"""

import sqlalchemy as sa

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "input_datasets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("partition_key", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("checksum", sa.String(), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "dataset", "partition_key", name="uq_input_datasets_dataset_partition_key"
        ),
    )


def downgrade() -> None:
    op.drop_table("input_datasets")
