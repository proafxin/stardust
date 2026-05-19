"""disambiguation default empty dict

Revision ID: g7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-05-19 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "g7b8c9d0e1f2"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE atoms SET disambiguation = '{\"pronoun_map\": []}' WHERE disambiguation IS NULL OR disambiguation::text = 'null'")
    op.alter_column("atoms", "disambiguation", nullable=False, server_default=sa.text("'{\"pronoun_map\": []}'::jsonb"))


def downgrade() -> None:
    op.alter_column("atoms", "disambiguation", nullable=True, server_default=None)
