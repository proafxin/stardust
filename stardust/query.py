import operator
from dataclasses import dataclass

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import cast, insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from stardust.config import RRF_K
from stardust.models import Atom, Sentence, SentenceEmbedding, TreeNode
from stardust.registry import embedder as load_embedder
from stardust.registry import reranker as load_reranker
from stardust.tree.atom import Node


@dataclass
class RankedSentence:
    id: int
    atom_id: int
    sentence_idx: int
    raw_text: str
    score: float


async def insert_index(docs: list[tuple[str, list[Node], list[int]]], session: AsyncSession) -> list[int]:
    atom_ids: list[int] = []
    for record_id, nodes, atom_indices in docs:
        atom_set = set(atom_indices)
        n = len(nodes)
        ids_result = await session.execute(text(f"SELECT nextval('tree_nodes_id_seq') FROM generate_series(1, {n})"))
        pre_ids = [row[0] for row in ids_result.fetchall()]
        await session.execute(
            insert(TreeNode),
            [
                {
                    "id": pre_ids[i],
                    "record_id": record_id,
                    "parent_id": pre_ids[node.parent_index] if node.parent_index is not None else None,
                    "node_type": node.node_type,
                    "modality": node.modality.value,
                    "value": node.value,
                }
                for i, node in enumerate(nodes)
            ],
        )
        atom_rows = [
            {
                "id": pre_ids[i],
                "record_id": record_id,
                "parent_id": pre_ids[nodes[i].parent_index] if nodes[i].parent_index is not None else None,
                "value": nodes[i].value,
            }
            for i in sorted(atom_set)
        ]
        if atom_rows:
            await session.execute(insert(Atom), atom_rows)
            atom_ids.extend(pre_ids[i] for i in sorted(atom_set))
    return atom_ids


async def insert_sentences(
    atom_id: int, sentences: list[tuple[str, str, int, str]], session: AsyncSession
) -> list[int]:
    if not sentences:
        return []
    result = await session.execute(
        insert(Sentence).returning(Sentence.id),
        [
            {
                "atom_id": atom_id,
                "sentence_idx": i,
                "raw_text": raw,
                "resolved_text": resolved,
                "token_count": tc,
                "value_hash": value_hash,
            }
            for i, (raw, resolved, tc, value_hash) in enumerate(sentences)
        ],
    )
    return [row[0] for row in result.fetchall()]


async def update_resolved_texts(updates: list[tuple[int, str, int, str]], session: AsyncSession) -> None:
    if not updates:
        return
    await session.execute(
        update(Sentence),
        [
            {"id": sid, "resolved_text": resolved, "token_count": tc, "value_hash": vh}
            for sid, resolved, tc, vh in updates
        ],
    )


async def insert_embeddings(rows: list[tuple[int, np.ndarray]], session: AsyncSession) -> None:
    if not rows:
        return
    await session.execute(
        insert(SentenceEmbedding),
        [{"sentence_id": sid, "embedding": vec.tolist()} for sid, vec in rows],
    )


async def dense_search(query: str, session: AsyncSession, top_k: int = 500, record_id: str | None = None) -> list[RankedSentence]:
    vec = cast(load_embedder().encode(query, normalize_embeddings=True).tolist(), Vector)
    distance = SentenceEmbedding.embedding.cosine_distance(vec).label("distance")
    stmt = (
        select(Sentence.id, Sentence.atom_id, Sentence.sentence_idx, Sentence.raw_text, (1 - distance).label("score"))
        .join(SentenceEmbedding, SentenceEmbedding.sentence_id == Sentence.id)
        .order_by(distance)
        .limit(top_k)
    )
    if record_id:
        stmt = stmt.join(Atom, Atom.id == Sentence.atom_id).where(Atom.record_id == record_id)
    rows = (await session.execute(stmt)).fetchall()
    return [
        RankedSentence(id=r.id, atom_id=r.atom_id, sentence_idx=r.sentence_idx, raw_text=r.raw_text, score=r.score)
        for r in rows
    ]


async def sparse_search(query: str, session: AsyncSession, top_k: int = 500, record_id: str | None = None) -> list[RankedSentence]:
    if record_id:
        sql = text("""
            SELECT s.id, s.atom_id, s.sentence_idx, s.raw_text,
                   paradedb.score(s.id) AS score
            FROM sentences s
            JOIN atoms a ON a.id = s.atom_id
            WHERE s.resolved_text @@@ :query
            AND a.record_id = :record_id
            ORDER BY score DESC
            LIMIT :top_k
        """)
    else:
        sql = text("""
            SELECT s.id, s.atom_id, s.sentence_idx, s.raw_text,
                   paradedb.score(s.id) AS score
            FROM sentences s
            WHERE s.resolved_text @@@ :query
            ORDER BY score DESC
            LIMIT :top_k
        """)
    rows = (await session.execute(sql, {"query": query, "record_id": record_id, "top_k": top_k})).fetchall()
    return [
        RankedSentence(id=r.id, atom_id=r.atom_id, sentence_idx=r.sentence_idx, raw_text=r.raw_text, score=r.score)
        for r in rows
    ]


def _rrf(dense: list[RankedSentence], sparse: list[RankedSentence], k: int = RRF_K) -> list[RankedSentence]:
    scores: dict[int, float] = {}
    all_sents: dict[int, RankedSentence] = {}
    for rank, s in enumerate(dense):
        scores[s.id] = scores.get(s.id, 0.0) + 1.0 / (k + rank + 1)
        all_sents[s.id] = s
    for rank, s in enumerate(sparse):
        scores[s.id] = scores.get(s.id, 0.0) + 1.0 / (k + rank + 1)
        all_sents[s.id] = s
    ranked = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    return [
        RankedSentence(
            id=all_sents[sid].id,
            atom_id=all_sents[sid].atom_id,
            sentence_idx=all_sents[sid].sentence_idx,
            raw_text=all_sents[sid].raw_text,
            score=sc,
        )
        for sid, sc in ranked
    ]


def _rerank(query: str, results: list[RankedSentence], top_k: int) -> list[RankedSentence]:
    pairs = [(query, r.raw_text) for r in results]
    scores = load_reranker().predict(pairs)
    reranked = sorted(zip(results, scores, strict=False), key=operator.itemgetter(1), reverse=True)
    return [
        RankedSentence(id=r.id, atom_id=r.atom_id, sentence_idx=r.sentence_idx, raw_text=r.raw_text, score=float(s))
        for r, s in reranked[:top_k]
    ]


async def retrieve(
    query: str,
    session: AsyncSession,
    top_k: int = 10,
    rerank_top_k: int = 5,
    record_id: str | None = None,
    use_reranker: bool = False,
) -> list[RankedSentence]:
    dense, sparse = (
        await dense_search(query, session, top_k, record_id),
        await sparse_search(query, session, top_k, record_id),
    )
    fused = _rrf(dense, sparse)
    candidates = fused[:top_k]
    return _rerank(query, candidates, rerank_top_k) if use_reranker else candidates
