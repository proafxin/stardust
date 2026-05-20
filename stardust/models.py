from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
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

    children: Mapped[list["TreeNode"]] = relationship("TreeNode", back_populates="parent")
    parent: Mapped["TreeNode | None"] = relationship("TreeNode", back_populates="children", remote_side="TreeNode.id")


class Atom(Base):
    __tablename__ = "atoms"

    record_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    parent_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("tree_nodes.id"), nullable=True)
    value: Mapped[str] = mapped_column(Text)

    __table_args__ = (Index("ix_atoms_record_id", "record_id"),)


class Sentence(Base):
    __tablename__ = "sentences"

    atom_id: Mapped[int] = mapped_column(Integer, ForeignKey("atoms.id"), nullable=False, index=True)
    sentence_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    resolved_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    value_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)

    __table_args__ = (
        Index("ix_sentences_atom_id", "atom_id"),
        Index(
            "ix_sentences_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 32, "ef_construction": 128},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
