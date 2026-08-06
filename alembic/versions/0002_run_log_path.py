"""add runs.log_path

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-05
"""

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("log_path", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "log_path")
