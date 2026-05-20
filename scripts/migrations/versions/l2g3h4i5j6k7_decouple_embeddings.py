"""decouple embeddings into sentence_embeddings table

Revision ID: l2g3h4i5j6k7
Revises: k1f2g3h4i5j6
Create Date: 2026-05-20 00:00:00.000000

"""
from typing import Sequence, Union

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision: str = "l2g3h4i5j6k7"
down_revision: Union[str, Sequence[str], None] = "k1f2g3h4i5j6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sentence_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("sentence_id", sa.Integer(), sa.ForeignKey("sentences.id"), nullable=False, unique=True),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(1024), nullable=False),
    )
    op.create_index("ix_sentence_embeddings_sentence_id", "sentence_embeddings", ["sentence_id"])
    op.create_index(
        "ix_sentence_embeddings_embedding",
        "sentence_embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 32, "ef_construction": 128},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.drop_index("ix_sentences_embedding", table_name="sentences", postgresql_using="hnsw")
    op.drop_column("sentences", "embedding")


def downgrade() -> None:
    op.add_column("sentences", sa.Column("embedding", pgvector.sqlalchemy.Vector(1024), nullable=True))
    op.create_index(
        "ix_sentences_embedding",
        "sentences",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 32, "ef_construction": 128},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.drop_index("ix_sentence_embeddings_embedding", table_name="sentence_embeddings", postgresql_using="hnsw")
    op.drop_index("ix_sentence_embeddings_sentence_id", table_name="sentence_embeddings")
    op.drop_table("sentence_embeddings")
