"""initial

Revision ID: 46d501f44259
Revises: 
Create Date: 2026-05-16 19:29:00.558351

"""
from typing import Sequence, Union

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "46d501f44259"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    op.create_table(
        "tree_nodes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("record_id", sa.String(length=256), nullable=False),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("node_type", sa.String(length=32), nullable=False),
        sa.Column("modality", sa.String(length=16), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("raw_offset", postgresql.JSONB(), nullable=False),
        sa.Column("clean_offset", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(["parent_id"], ["tree_nodes.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_tree_nodes_record_id", "tree_nodes", ["record_id"])
    op.create_table(
        "atoms",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("record_id", sa.String(length=256), nullable=False),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("raw_offset", postgresql.JSONB(), nullable=False),
        sa.Column("clean_offset", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(["parent_id"], ["tree_nodes.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_atoms_record_id", "atoms", ["record_id"])
    op.create_table(
        "table_signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("record_id", sa.String(length=256), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("col_names", postgresql.JSONB(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_table_signals_record_id", "table_signals", ["record_id"])
    op.create_table(
        "table_rows",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("signal_id", sa.Integer(), sa.ForeignKey("table_signals.id"), nullable=False),
        sa.Column("row_idx", sa.Integer(), nullable=False),
        sa.Column("col_idx", sa.Integer(), nullable=False),
        sa.Column("cell_value", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_table_rows_signal_id", "table_rows", ["signal_id"])
    op.create_table(
        "sentences",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("atom_id", sa.Integer(), sa.ForeignKey("atoms.id"), nullable=False),
        sa.Column("sentence_idx", sa.Integer(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("resolved_text", sa.Text(), nullable=True),
        sa.Column("value_hash", sa.String(64), nullable=True),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(1024), nullable=True),
    )
    op.create_index("ix_sentences_atom_id", "sentences", ["atom_id"])
    op.create_index(
        "ix_sentences_embedding", "sentences", ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_sentences_embedding", table_name="sentences", postgresql_using="hnsw")
    op.drop_index("ix_sentences_atom_id", table_name="sentences")
    op.drop_table("sentences")
    op.drop_table("table_rows")
    op.drop_table("table_signals")
    op.drop_index("ix_atoms_record_id", table_name="atoms")
    op.drop_table("atoms")
    op.drop_index("ix_tree_nodes_record_id", table_name="tree_nodes")
    op.drop_table("tree_nodes")
