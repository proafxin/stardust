from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    pass


class TreeNodeModel(Base):
    __tablename__ = "tree_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    parent_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("tree_nodes.id"), nullable=True)
    node_type: Mapped[str] = mapped_column(String(32))
    modality: Mapped[str] = mapped_column(String(16))
    value: Mapped[str] = mapped_column(Text)
    raw_offset: Mapped[dict] = mapped_column(JSONB)
    clean_offset: Mapped[dict] = mapped_column(JSONB)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    nlp_attributes: Mapped[list] = mapped_column(JSONB, default=[])
    disambiguation: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    children: Mapped[list[TreeNodeModel]] = relationship("TreeNodeModel", back_populates="parent")
    parent: Mapped[TreeNodeModel | None] = relationship(
        "TreeNodeModel", back_populates="children", remote_side=[id]
    )


class AtomModel(Base):
    __tablename__ = "atoms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    parent_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("tree_nodes.id"), nullable=True)
    value: Mapped[str] = mapped_column(Text)
    raw_offset: Mapped[dict] = mapped_column(JSONB)
    clean_offset: Mapped[dict] = mapped_column(JSONB)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    nlp_attributes: Mapped[list] = mapped_column(JSONB, default=[])
    disambiguation: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)

    __table_args__ = (
        Index("ix_atoms_doc_id", "doc_id"),
        Index("ix_atoms_embedding", "embedding", postgresql_using="hnsw", postgresql_with={"m": 16, "ef_construction": 64}),
    )


class CanonicalEntityModel(Base):
    __tablename__ = "canonical_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(Text)
    entity_type: Mapped[str] = mapped_column(String(64))
    aliases: Mapped[list] = mapped_column(JSONB, default=[])

    mentions: Mapped[list[EntityMentionModel]] = relationship(
        "EntityMentionModel", back_populates="canonical_entity"
    )


class EntityMentionModel(Base):
    __tablename__ = "entity_mentions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    atom_id: Mapped[int] = mapped_column(Integer, ForeignKey("atoms.id"))
    doc_id: Mapped[str] = mapped_column(String(256), nullable=False)
    canonical_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("canonical_entities.id"), nullable=True
    )
    text: Mapped[str] = mapped_column(Text)
    entity_type: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_offset: Mapped[dict] = mapped_column(JSONB)
    clean_offset: Mapped[dict] = mapped_column(JSONB)

    canonical_entity: Mapped[CanonicalEntityModel | None] = relationship(
        "CanonicalEntityModel", back_populates="mentions"
    )
