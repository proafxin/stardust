from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from stardust.config import EMBEDDING_DIM


class Base(DeclarativeBase):
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class TreeNode(Base):
    __tablename__ = "tree_nodes"

    record_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    parent_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("tree_nodes.id"), nullable=True)
    node_type: Mapped[str] = mapped_column(String(32))
    modality: Mapped[str] = mapped_column(String(16))
    value: Mapped[str] = mapped_column(Text)
    raw_offset: Mapped[dict] = mapped_column(JSONB)
    clean_offset: Mapped[dict] = mapped_column(JSONB)

    children: Mapped[list["TreeNode"]] = relationship("TreeNode", back_populates="parent")
    parent: Mapped["TreeNode | None"] = relationship("TreeNode", back_populates="children", remote_side="TreeNode.id")


class Atom(Base):
    __tablename__ = "atoms"

    record_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    parent_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("tree_nodes.id"), nullable=True)
    value: Mapped[str] = mapped_column(Text)
    value_enriched: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_offset: Mapped[dict] = mapped_column(JSONB)
    clean_offset: Mapped[dict] = mapped_column(JSONB)
    value_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)

    __table_args__ = (
        Index("ix_atoms_record_id", "record_id"),
        Index(
            "ix_atoms_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class TableSignal(Base):
    __tablename__ = "table_signals"

    record_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    col_names: Mapped[list] = mapped_column(JSONB, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)


class TableRow(Base):
    __tablename__ = "table_rows"

    signal_id: Mapped[int] = mapped_column(Integer, ForeignKey("table_signals.id"), nullable=False, index=True)
    row_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    col_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    cell_value: Mapped[str] = mapped_column(Text, nullable=False)
