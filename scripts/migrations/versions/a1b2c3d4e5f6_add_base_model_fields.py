"""add base model fields

Revision ID: a1b2c3d4e5f6
Revises: f6a7b8c9d0e1
Create Date: 2026-05-19 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for table in ("tree_nodes", "atoms", "canonical_entities", "entity_mentions"):
        op.execute(f"CREATE SEQUENCE IF NOT EXISTS {table}_id_seq OWNED BY {table}.id")
        op.execute(f"ALTER TABLE {table} ALTER COLUMN id SET DEFAULT nextval('{table}_id_seq')")
        op.execute(f"SELECT setval('{table}_id_seq', COALESCE((SELECT MAX(id) FROM {table}), 0) + 1, false)")
        op.add_column(table, sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
        op.add_column(table, sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))

    op.execute("ALTER TABLE batch_prompts DROP CONSTRAINT batch_prompts_pkey")
    op.add_column("batch_prompts", sa.Column("id", sa.Integer(), autoincrement=True, nullable=False))
    op.execute("CREATE SEQUENCE IF NOT EXISTS batch_prompts_id_seq OWNED BY batch_prompts.id")
    op.execute("ALTER TABLE batch_prompts ALTER COLUMN id SET DEFAULT nextval('batch_prompts_id_seq')")
    op.execute("UPDATE batch_prompts SET id = batch_no")
    op.execute("SELECT setval('batch_prompts_id_seq', COALESCE((SELECT MAX(id) FROM batch_prompts), 0) + 1, false)")
    op.create_primary_key("batch_prompts_pkey", "batch_prompts", ["id"])
    op.add_column("batch_prompts", sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.add_column("batch_prompts", sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))


def downgrade() -> None:
    for table in ("tree_nodes", "atoms", "canonical_entities", "entity_mentions"):
        op.drop_column(table, "created_at")
        op.drop_column(table, "updated_at")
        op.execute(f"ALTER TABLE {table} ALTER COLUMN id DROP DEFAULT")

    op.drop_column("batch_prompts", "created_at")
    op.drop_column("batch_prompts", "updated_at")
    op.execute("ALTER TABLE batch_prompts DROP CONSTRAINT batch_prompts_pkey")
    op.drop_column("batch_prompts", "id")
    op.create_primary_key("batch_prompts_pkey", "batch_prompts", ["batch_no"])
