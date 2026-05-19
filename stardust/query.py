import operator
from dataclasses import dataclass

from pgvector.sqlalchemy import Vector
from sqlalchemy import cast, insert, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from stardust.config import RRF_K
from stardust.models import Atom, CanonicalEntity, Disambiguation, EntityMention, Token, TreeNode
from stardust.registry import embedder as load_embedder
from stardust.registry import reranker as load_reranker
from stardust.tree.atom import Node


@dataclass
class RankedAtom:
    id: int
    record_id: str
    value: str
    score: float


async def insert_index(docs: list[tuple[str, list[Node], list[int]]], session: AsyncSession) -> None:
    for record_id, nodes, atom_indices in docs:
        atom_set = set(atom_indices)
        db_ids: list[int] = []
        for i, node in enumerate(nodes):
            db_parent_id = db_ids[node.parent_index] if node.parent_index is not None else None
            result = await session.execute(
                insert(TreeNode)
                .values(
                    record_id=record_id,
                    parent_id=db_parent_id,
                    node_type=node.node_type,
                    modality=node.modality.value,
                    value=node.value,
                    raw_offset=node.raw_offset.model_dump(),
                    clean_offset=node.clean_offset.model_dump(),
                )
                .returning(TreeNode.id)
            )
            db_id = result.scalar_one()
            db_ids.append(db_id)
            if i in atom_set:
                await session.execute(
                    insert(Atom).values(
                        id=db_id,
                        record_id=record_id,
                        parent_id=db_parent_id,
                        value=node.value,
                        raw_offset=node.raw_offset.model_dump(),
                        clean_offset=node.clean_offset.model_dump(),
                        embedding=None,
                    )
                )
    await session.commit()


async def insert_tokens(atom_id: int, tokens: list[dict], session: AsyncSession) -> None:
    if not tokens:
        return
    await session.execute(insert(Token), [{"atom_id": atom_id, **t} for t in tokens])


async def insert_canonical_entities(
    entities: list[tuple[str, str, list[str], list[tuple[int, str, int]]]],
    session: AsyncSession,
) -> None:
    for canonical_name, entity_type, aliases, mentions in entities:
        result = await session.execute(
            insert(CanonicalEntity)
            .values(canonical_name=canonical_name, entity_type=entity_type, aliases=aliases)
            .returning(CanonicalEntity.id)
        )
        ce_id = result.scalar_one()
        member_token_ids = [token_id for token_id, _, _ in mentions]
        for token_id, record_id, atom_id in mentions:
            session.add(
                EntityMention(
                    atom_id=atom_id,
                    record_id=record_id,
                    canonical_id=ce_id,
                    text=canonical_name,
                    entity_type=entity_type,
                    raw_offset={"start": 0, "end": 0},
                    clean_offset={"start": 0, "end": 0},
                )
            )
        await session.execute(
            Disambiguation.__table__.update()
            .where(Disambiguation.canonical_token_id.in_(member_token_ids))
            .values(canonical_entity_id=ce_id)
        )
    await session.commit()


async def dense_search(query: str, session: AsyncSession, record_id: str | None = None) -> list[RankedAtom]:
    vec = cast(load_embedder().encode(query, normalize_embeddings=True).tolist(), Vector)
    distance = Atom.embedding.cosine_distance(vec).label("distance")
    stmt = (
        select(Atom.id, Atom.record_id, Atom.value, (1 - distance).label("score"))
        .where(Atom.embedding.is_not(None))
        .order_by(distance)
    )
    if record_id:
        stmt = stmt.where(Atom.record_id == record_id)
    rows = (await session.execute(stmt)).fetchall()
    return [RankedAtom(id=r.id, record_id=r.record_id, value=r.value, score=r.score) for r in rows]


async def sparse_search(query: str, session: AsyncSession, record_id: str | None = None) -> list[RankedAtom]:
    rows = (
        await session.execute(
            text("""
            SELECT id, record_id, value,
                   paradedb.score(id) AS score
            FROM atoms
            WHERE value @@@ :query
            AND (:record_id IS NULL OR record_id = :record_id)
            ORDER BY score DESC
        """),
            {"query": query, "record_id": record_id},
        )
    ).fetchall()
    return [RankedAtom(id=r.id, record_id=r.record_id, value=r.value, score=r.score) for r in rows]


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
        RankedAtom(id=all_atoms[aid].id, record_id=all_atoms[aid].record_id, value=all_atoms[aid].value, score=s)
        for aid, s in ranked
    ]


def _rerank(query: str, results: list[RankedAtom], top_k: int) -> list[RankedAtom]:
    pairs = [(query, r.value) for r in results]
    scores = load_reranker().predict(pairs)
    reranked = sorted(zip(results, scores, strict=False), key=operator.itemgetter(1), reverse=True)
    return [RankedAtom(id=r.id, record_id=r.record_id, value=r.value, score=float(s)) for r, s in reranked[:top_k]]


async def retrieve(
    query: str,
    session: AsyncSession,
    top_k: int = 10,
    rerank_top_k: int = 5,
    record_id: str | None = None,
    use_reranker: bool = False,
) -> list[RankedAtom]:
    dense, sparse = (
        await dense_search(query, session, record_id),
        await sparse_search(query, session, record_id),
    )
    fused = _rrf(dense, sparse)
    candidates = fused[:top_k]
    return _rerank(query, candidates, rerank_top_k) if use_reranker else candidates
