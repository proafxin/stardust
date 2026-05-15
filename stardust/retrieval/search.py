import operator
from dataclasses import dataclass

import numpy as np
from config import RRF_K

from stardust.registry import get_embedder, get_reranker
from stardust.tree.atom import AtomIndex, Node


@dataclass
class RankedAtom:
    node: Node
    score: float


def embed_atoms(index: AtomIndex) -> None:
    embedder = get_embedder()
    atom_ids = index.atoms
    texts = [index.nodes[a].value for a in atom_ids]
    vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    for atom_id, vec in zip(atom_ids, vecs, strict=False):
        index.embeddings[atom_id] = vec.tolist()


def dense_search(query: str, index: AtomIndex, top_k: int = 10) -> list[RankedAtom]:
    embedder = get_embedder()
    query_vec = embedder.encode(query, normalize_embeddings=True)
    scores: list[tuple[int, float]] = []
    for atom_id in index.atoms:
        vec = index.embeddings.get(atom_id)
        if vec is None:
            continue
        score = float(np.dot(query_vec, np.array(vec)))
        scores.append((atom_id, score))
    scores.sort(key=operator.itemgetter(1), reverse=True)
    return [RankedAtom(node=index.nodes[aid], score=s) for aid, s in scores[:top_k]]


def sparse_search(query: str, index: AtomIndex, top_k: int = 10) -> list[RankedAtom]:
    query_terms = set(query.lower().split())
    scores: list[tuple[int, float]] = []
    for atom_id in index.atoms:
        text = index.nodes[atom_id].value.lower()
        score = sum(text.count(term) for term in query_terms)
        if score > 0:
            scores.append((atom_id, float(score)))
    scores.sort(key=operator.itemgetter(1), reverse=True)
    return [RankedAtom(node=index.nodes[aid], score=s) for aid, s in scores[:top_k]]


def rrf(
    dense_results: list[RankedAtom],
    sparse_results: list[RankedAtom],
    k: int = RRF_K,
) -> list[RankedAtom]:
    scores: dict[int, float] = {}
    for rank, result in enumerate(dense_results):
        nid = result.node.id
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
    for rank, result in enumerate(sparse_results):
        nid = result.node.id
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)

    all_nodes: dict[int, Node] = {r.node.id: r.node for r in dense_results + sparse_results}
    ranked = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    return [RankedAtom(node=all_nodes[nid], score=s) for nid, s in ranked]


def rerank(query: str, results: list[RankedAtom], top_k: int = 5) -> list[RankedAtom]:
    reranker = get_reranker()
    pairs = [(query, r.node.value) for r in results]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(results, scores, strict=False), key=operator.itemgetter(1), reverse=True)
    return [RankedAtom(node=r.node, score=float(s)) for r, s in reranked[:top_k]]


def retrieve(
    query: str,
    index: AtomIndex,
    top_k: int = 10,
    rerank_top_k: int = 5,
    use_reranker: bool = False,
) -> list[RankedAtom]:
    dense = dense_search(query, index, top_k)
    sparse = sparse_search(query, index, top_k)
    fused = rrf(dense, sparse)
    if use_reranker:
        return rerank(query, fused, rerank_top_k)
    return fused[:rerank_top_k]
