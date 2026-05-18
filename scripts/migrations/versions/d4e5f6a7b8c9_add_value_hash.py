"""add value_hash to atoms

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-05-18 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("atoms", sa.Column("value_hash", sa.String(64), nullable=True))
    op.create_index("ix_atoms_value_hash", "atoms", ["value_hash"])


def downgrade() -> None:
    op.drop_index("ix_atoms_value_hash", table_name="atoms")
    op.drop_column("atoms", "value_hash")
