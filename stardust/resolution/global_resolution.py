import logging
import re
import unicodedata
from collections import defaultdict

import faiss
import numpy as np
import torch

from stardust.config import EMBEDDING_INTERNAL_BATCH_SIZE, ENTITY_MERGE_THRESHOLD, GLOBAL_MERGE_THRESHOLD
from stardust.registry import embedder as load_embedder
from stardust.tree.atom import SpanOffset

log = logging.getLogger(__name__)


class CanonicalEntity:
    def __init__(self, entity_type: str, mentions: list[tuple[str, SpanOffset, int, str]]) -> None:
        self.entity_type = entity_type
        self.mentions = mentions
        self.canonical_name: str = mentions[0][0] if mentions else ""
        self.aliases: list[str] = list({m[0] for m in mentions})


def _normalize(surface: str) -> str:
    s = unicodedata.normalize("NFKC", surface)
    s = re.sub(r"'s\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


class _UF:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x: int, y: int) -> None:
        self.p[self.find(x)] = self.find(y)


def _encode(embedder, texts: list[str]) -> np.ndarray:
    log.info("encode: %d texts, VRAM free %.2fGB", len(texts), torch.cuda.mem_get_info()[0] / 1024**3)
    if not texts:
        return np.array([])
    result = embedder.encode(
        texts,
        batch_size=EMBEDDING_INTERNAL_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    log.info("encode done, VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)
    return np.array(result)


def _greedy_cluster(vecs: np.ndarray, threshold: float) -> list[list[int]]:
    if len(vecs) == 0:
        return []
    vecs = vecs.astype(np.float32)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    assigned = [False] * len(vecs)
    clusters: list[list[int]] = []
    k = min(64, len(vecs))
    distances, indices = index.search(vecs, k)
    for i in range(len(vecs)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j, dist in zip(indices[i], distances[i]):
            if j == i or assigned[j]:
                continue
            if dist >= threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)
    return clusters


def _disambiguate(
    per_record: list[tuple[str, list[tuple[str, str, SpanOffset, int, str]]]],
    atom_texts: dict[int, str],
) -> list[CanonicalEntity]:
    embedder = load_embedder()

    all_mentions: list[tuple[str, str, SpanOffset, int, str, str]] = []
    for record_id, mentions in per_record:
        for surface, ent_type, offset, atom_id, context in mentions:
            all_mentions.append((surface, ent_type, offset, atom_id, record_id, context))

    if not all_mentions:
        return []

    # step 1: canonicalize entity types
    distinct_types = list({m[1] for m in all_mentions})
    type_vecs = _encode(embedder, distinct_types)
    type_clusters = _greedy_cluster(type_vecs, ENTITY_MERGE_THRESHOLD)
    type_to_canonical: dict[str, str] = {}
    for cluster in type_clusters:
        canonical_type = distinct_types[cluster[0]]
        for idx in cluster:
            type_to_canonical[distinct_types[idx]] = canonical_type

    # step 2: group by canonical type
    by_type: dict[str, list[tuple[str, SpanOffset, int, str, str]]] = defaultdict(list)
    for surface, ent_type, offset, atom_id, record_id, context in all_mentions:
        by_type[type_to_canonical[ent_type]].append((surface, offset, atom_id, record_id, context))

    # step 3: encode unique atom texts once
    unique_atom_ids = list({atom_id for mentions in by_type.values() for _, _, atom_id, _, _ in mentions})
    unique_texts = [atom_texts.get(atom_id, "") for atom_id in unique_atom_ids]
    log.info("before unique_vecs encode: VRAM free %.2fGB, unique_texts: %d", torch.cuda.mem_get_info()[0] / 1024**3, len(unique_texts))
    unique_vecs = _encode(embedder, unique_texts)
    atom_vec_map = {atom_id: unique_vecs[i].copy() for i, atom_id in enumerate(unique_atom_ids)}

    global_entities: list[CanonicalEntity] = []
    for canonical_type, mentions in by_type.items():
        n = len(mentions)
        uf = _UF(n)
        norms = [_normalize(m[0]) for m in mentions]

        # step 4: union exact normalized surface matches
        norm_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, norm in enumerate(norms):
            norm_to_indices[norm].append(i)
        for indices in norm_to_indices.values():
            for idx in indices[1:]:
                uf.union(indices[0], idx)

        # step 5: subsumption — if norm(A) is a non-empty substring of norm(B), union them
        for i in range(n):
            if not norms[i]:
                continue
            for j in range(n):
                if i == j or not norms[j] or norms[i] == norms[j]:
                    continue
                if norms[i] in norms[j]:
                    uf.union(i, j)

        # step 6: one representative per union-find group for embedding clustering
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[uf.find(i)].append(i)

        rep_indices = [indices[0] for indices in groups.values()]
        rep_contexts = [mentions[i][4] for i in rep_indices]
        rep_atom_ids = [mentions[i][2] for i in rep_indices]

        unique_rep_contexts = list({c for c in rep_contexts})
        context_vec_map = dict(zip(unique_rep_contexts, _encode(embedder, unique_rep_contexts)))
        context_vecs = np.array([context_vec_map[rep_contexts[k]] for k in range(len(rep_indices))])
        atom_vecs = np.array([atom_vec_map[rep_atom_ids[k]] for k in range(len(rep_indices))])
        vecs = (atom_vecs + context_vecs) / 2
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        local_clusters = _greedy_cluster(vecs, ENTITY_MERGE_THRESHOLD)
        cluster_vecs = np.array([np.mean(vecs[cluster], axis=0) for cluster in local_clusters])
        cluster_vecs = cluster_vecs / np.linalg.norm(cluster_vecs, axis=1, keepdims=True)
        global_clusters = _greedy_cluster(cluster_vecs, GLOBAL_MERGE_THRESHOLD)

        for global_cluster in global_clusters:
            merged: list[tuple[str, SpanOffset, int, str]] = []
            for local_idx in global_cluster:
                rep_i = rep_indices[local_idx]
                for mention_idx in groups[uf.find(rep_i)]:
                    surface, offset, atom_id, record_id, _ = mentions[mention_idx]
                    merged.append((surface, offset, atom_id, record_id))
            global_entities.append(CanonicalEntity(canonical_type, merged))

    return global_entities


async def merge_across_records(
    per_record: list[tuple[str, list[tuple[str, str, SpanOffset, int, str]]]],
    atom_texts: dict[int, str],
) -> list[CanonicalEntity]:
    return _disambiguate(per_record, atom_texts)
