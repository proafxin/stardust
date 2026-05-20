"""drop offsets, fix resolved_text not null

Revision ID: k1f2g3h4i5j6
Revises: j0e1f2g3h4i5
Create Date: 2026-05-20 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "k1f2g3h4i5j6"
down_revision: Union[str, Sequence[str], None] = "j0e1f2g3h4i5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("tree_nodes", "raw_offset")
    op.drop_column("tree_nodes", "clean_offset")
    op.drop_column("atoms", "raw_offset")
    op.drop_column("atoms", "clean_offset")
    op.alter_column("sentences", "resolved_text", existing_type=sa.Text(), nullable=False)


def downgrade() -> None:
    op.alter_column("sentences", "resolved_text", existing_type=sa.Text(), nullable=True)
    op.add_column("atoms", sa.Column("clean_offset", sa.dialects.postgresql.JSONB(), nullable=True))
    op.add_column("atoms", sa.Column("raw_offset", sa.dialects.postgresql.JSONB(), nullable=True))
    op.add_column("tree_nodes", sa.Column("clean_offset", sa.dialects.postgresql.JSONB(), nullable=True))
    op.add_column("tree_nodes", sa.Column("raw_offset", sa.dialects.postgresql.JSONB(), nullable=True))
