from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
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
    nlp_attributes: Mapped[list] = mapped_column(JSONB, default=[])
    disambiguation: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    children: Mapped[list["TreeNode"]] = relationship("TreeNode", back_populates="parent")
    parent: Mapped["TreeNode | None"] = relationship("TreeNode", back_populates="children", remote_side="TreeNode.id")


class Atom(Base):
    __tablename__ = "atoms"

    record_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    parent_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("tree_nodes.id"), nullable=True)
    value: Mapped[str] = mapped_column(Text)
    raw_offset: Mapped[dict] = mapped_column(JSONB)
    clean_offset: Mapped[dict] = mapped_column(JSONB)
    nlp_attributes: Mapped[list] = mapped_column(JSONB, default=[])
    disambiguation: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
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


class BatchPrompt(Base):
    __tablename__ = "batch_prompts"

    batch_no: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    prompt: Mapped[str] = mapped_column(Text)


class CanonicalEntity(Base):
    __tablename__ = "canonical_entities"

    canonical_name: Mapped[str] = mapped_column(Text)
    entity_type: Mapped[str] = mapped_column(String(64))
    aliases: Mapped[list] = mapped_column(JSONB, default=[])

    mentions: Mapped[list["EntityMention"]] = relationship("EntityMention", back_populates="canonical_entity")


class EntityMention(Base):
    __tablename__ = "entity_mentions"

    atom_id: Mapped[int] = mapped_column(Integer, ForeignKey("atoms.id"))
    record_id: Mapped[str] = mapped_column(String(256), nullable=False)
    canonical_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("canonical_entities.id"), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    entity_type: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_offset: Mapped[dict] = mapped_column(JSONB)
    clean_offset: Mapped[dict] = mapped_column(JSONB)

    canonical_entity: Mapped["CanonicalEntity | None"] = relationship("CanonicalEntity", back_populates="mentions")
