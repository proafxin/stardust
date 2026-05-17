import operator
from dataclasses import dataclass

from pgvector.sqlalchemy import Vector
from sqlalchemy import cast, insert, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from stardust.config import RRF_K
from stardust.models import AtomModel, CanonicalEntityModel, EntityMentionModel, TreeNodeModel
from stardust.registry import embedder as load_embedder
from stardust.registry import reranker as load_reranker
from stardust.resolution.global_resolution import CanonicalEntity
from stardust.tree.atom import Node


@dataclass
class RankedAtom:
    id: int
    doc_id: str
    value: str
    score: float


_INSERT_CHUNK = 3_000  # 9 cols in tree_nodes, 8 in atoms → safe under 32767 param limit


async def insert_index(docs: list[tuple[str, list[Node], list[int]]], session: AsyncSession) -> None:
    atom_set_per_doc = {doc_id: set(atoms) for doc_id, _, atoms in docs}
    tree_rows = [
        {
            "id": node.id, "doc_id": doc_id, "parent_id": node.parent_id,
            "node_type": node.node_type, "modality": node.modality.value,
            "value": node.value, "raw_offset": node.raw_offset.model_dump(),
            "clean_offset": node.clean_offset.model_dump(), "nlp_attributes": [],
            "disambiguation": None,
        }
        for doc_id, nodes, _ in docs
        for node in nodes
    ]
    atom_rows = [
        {
            "id": node.id, "doc_id": doc_id, "parent_id": node.parent_id,
            "value": node.value, "raw_offset": node.raw_offset.model_dump(),
            "clean_offset": node.clean_offset.model_dump(), "nlp_attributes": [],
            "disambiguation": None, "embedding": None,
        }
        for doc_id, nodes, _ in docs
        for node in nodes
        if node.id in atom_set_per_doc[doc_id]
    ]
    for i in range(0, len(tree_rows), _INSERT_CHUNK):
        await session.execute(insert(TreeNodeModel), tree_rows[i : i + _INSERT_CHUNK])
    for i in range(0, len(atom_rows), _INSERT_CHUNK):
        await session.execute(insert(AtomModel), atom_rows[i : i + _INSERT_CHUNK])
    await session.commit()


async def insert_canonical_entities(entities: list[CanonicalEntity], doc_id: str, session: AsyncSession) -> None:
    for entity in entities:
        ce = CanonicalEntityModel(
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            aliases=entity.aliases,
        )
        session.add(ce)
        await session.flush()
        for text_val, offset, atom_id, _ in entity.mentions:
            session.add(
                EntityMentionModel(
                    atom_id=atom_id,
                    doc_id=doc_id,
                    canonical_id=ce.id,
                    text=text_val,
                    entity_type=entity.entity_type,
                    raw_offset=offset.model_dump(),
                    clean_offset=offset.model_dump(),
                )
            )
    await session.commit()


async def dense_search(query: str, session: AsyncSession, doc_id: str | None = None) -> list[RankedAtom]:
    vec = cast(load_embedder().encode(query, normalize_embeddings=True).tolist(), Vector)
    distance = AtomModel.embedding.cosine_distance(vec).label("distance")
    stmt = (
        select(AtomModel.id, AtomModel.doc_id, AtomModel.value, (1 - distance).label("score"))
        .where(AtomModel.embedding.is_not(None))
        .order_by(distance)
    )
    if doc_id:
        stmt = stmt.where(AtomModel.doc_id == doc_id)
    rows = (await session.execute(stmt)).fetchall()
    return [RankedAtom(id=r.id, doc_id=r.doc_id, value=r.value, score=r.score) for r in rows]


async def sparse_search(query: str, session: AsyncSession, doc_id: str | None = None) -> list[RankedAtom]:
    rows = (
        await session.execute(
            text("""
            SELECT id, doc_id, value,
                   paradedb.score(id) AS score
            FROM atoms
            WHERE value @@@ :query
            AND (:doc_id IS NULL OR doc_id = :doc_id)
            ORDER BY score DESC
        """),
            {"query": query, "doc_id": doc_id},
        )
    ).fetchall()
    return [RankedAtom(id=r.id, doc_id=r.doc_id, value=r.value, score=r.score) for r in rows]


def _rrf(dense: list[RankedAtom], sparse: list[RankedAtom], k: int = RRF_K) -> list[RankedAtom]:
    scores: dict[int, float] = {}
    all_atoms: dict[int, RankedAtom] = {}
    for rank, atom in enumerate(dense):
        scores[atom.id] = scores.get(atom.id, 0.0) + 1.0 / (k + rank + 1)
        all_atoms[atom.id] = atom
    for rank, atom in enumerate(sparse):
        scores[atom.id] = scores.get(atom.id, 0.0) + 1.0 / (k + rank + 1)
        all_atoms[atom.id] = atom
    ranked = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    return [
        RankedAtom(id=all_atoms[aid].id, doc_id=all_atoms[aid].doc_id, value=all_atoms[aid].value, score=s)
        for aid, s in ranked
    ]


def _rerank(query: str, results: list[RankedAtom], top_k: int) -> list[RankedAtom]:
    pairs = [(query, r.value) for r in results]
    scores = load_reranker().predict(pairs)
    reranked = sorted(zip(results, scores, strict=False), key=operator.itemgetter(1), reverse=True)
    return [RankedAtom(id=r.id, doc_id=r.doc_id, value=r.value, score=float(s)) for r, s in reranked[:top_k]]


async def retrieve(
    query: str,
    session: AsyncSession,
    top_k: int = 10,
    rerank_top_k: int = 5,
    doc_id: str | None = None,
    use_reranker: bool = False,
) -> list[RankedAtom]:
    dense, sparse = (
        await dense_search(query, session, doc_id),
        await sparse_search(query, session, doc_id),
    )
    fused = _rrf(dense, sparse)
    candidates = fused[:top_k]
    return _rerank(query, candidates, rerank_top_k) if use_reranker else candidates
