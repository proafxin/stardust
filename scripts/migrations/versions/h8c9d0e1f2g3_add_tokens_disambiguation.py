"""add tokens and disambiguation tables

Revision ID: h8c9d0e1f2g3
Revises: g7b8c9d0e1f2
Create Date: 2026-05-19 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from alembic import op

revision: str = "h8c9d0e1f2g3"
down_revision: Union[str, Sequence[str], None] = "g7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tokens",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("atom_id", sa.Integer(), sa.ForeignKey("atoms.id"), nullable=False),
        sa.Column("token_index", sa.Integer(), nullable=False),
        sa.Column("start", sa.Integer(), nullable=False),
        sa.Column("end", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("pos", sa.String(16), nullable=False),
        sa.Column("dep", sa.String(32), nullable=False),
        sa.Column("morph", JSONB(), nullable=False),
        sa.Column("ent_type", sa.String(64), nullable=True),
        sa.Column("ent_iob", sa.String(4), nullable=True),
        sa.Column("context", sa.Text(), nullable=False),
    )
    op.create_index("ix_tokens_atom_id", "tokens", ["atom_id"])

    op.create_table(
        "disambiguation",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("token_id", sa.Integer(), sa.ForeignKey("tokens.id"), nullable=False),
        sa.Column("referent_id", sa.Integer(), sa.ForeignKey("tokens.id"), nullable=False),
        sa.Column("canonical_token_id", sa.Integer(), sa.ForeignKey("tokens.id"), nullable=False),
        sa.Column("atom_id", sa.Integer(), sa.ForeignKey("atoms.id"), nullable=False),
        sa.Column("canonical_entity_id", sa.Integer(), sa.ForeignKey("canonical_entities.id"), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
    )
    op.create_index("ix_disambiguation_token_id", "disambiguation", ["token_id"])
    op.create_index("ix_disambiguation_atom_id", "disambiguation", ["atom_id"])

    op.drop_column("atoms", "nlp_attributes")
    op.drop_column("atoms", "disambiguation")
    op.drop_column("tree_nodes", "nlp_attributes")
    op.drop_column("tree_nodes", "disambiguation")


def downgrade() -> None:
    op.drop_table("disambiguation")
    op.drop_table("tokens")
    op.add_column("atoms", sa.Column("nlp_attributes", JSONB(), nullable=False, server_default="[]"))
    op.add_column("atoms", sa.Column("disambiguation", JSONB(), nullable=True))
    op.add_column("tree_nodes", sa.Column("nlp_attributes", JSONB(), nullable=False, server_default="[]"))
    op.add_column("tree_nodes", sa.Column("disambiguation", JSONB(), nullable=True))
