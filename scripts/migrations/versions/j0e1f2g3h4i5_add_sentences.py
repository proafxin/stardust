"""add sentences table

Revision ID: j0e1f2g3h4i5
Revises: i9d0e1f2g3h4
Create Date: 2026-05-20 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "j0e1f2g3h4i5"
down_revision: Union[str, Sequence[str], None] = "i9d0e1f2g3h4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("sentences", sa.Column("token_count", sa.Integer(), nullable=True))
    op.execute("UPDATE sentences SET token_count = 0 WHERE token_count IS NULL")
    op.alter_column("sentences", "token_count", nullable=False)


def downgrade() -> None:
    op.drop_column("sentences", "token_count")
