"""drop metadata column

Revision ID: b2c3d4e5f6a7
Revises: 46d501f44259
Create Date: 2026-05-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "46d501f44259"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("tree_nodes", "metadata")
    op.drop_column("atoms", "metadata")


def downgrade() -> None:
    from sqlalchemy.dialects import postgresql
    import sqlalchemy as sa
    op.add_column("tree_nodes", sa.Column("metadata", postgresql.JSONB(), nullable=True))
    op.add_column("atoms", sa.Column("metadata", postgresql.JSONB(), nullable=True))
