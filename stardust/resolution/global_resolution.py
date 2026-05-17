import asyncio
from collections import defaultdict

import numpy as np

from stardust.config import EMBEDDING_BATCH_SIZE, ENTITY_MERGE_THRESHOLD, GLOBAL_MERGE_THRESHOLD
from stardust.registry import embedder as load_embedder
from stardust.tree.atom import SpanOffset


class CanonicalEntity:
    def __init__(self, entity_type: str, mentions: list[tuple[str, SpanOffset, int, str]]) -> None:
        self.entity_type = entity_type
        self.mentions = mentions
        self.canonical_name: str = mentions[0][0] if mentions else ""
        self.aliases: list[str] = list({m[0] for m in mentions})


def _batched_encode(embedder, texts: list[str]) -> np.ndarray:
    vecs = [embedder.encode(texts[i : i + EMBEDDING_BATCH_SIZE], normalize_embeddings=True, batch_size=EMBEDDING_BATCH_SIZE) for i in range(0, len(texts), EMBEDDING_BATCH_SIZE)]
    return np.vstack(vecs) if len(vecs) > 1 else vecs[0]


def _greedy_cluster(vecs: np.ndarray, threshold: float) -> list[list[int]]:
    assigned = [False] * len(vecs)
    clusters: list[list[int]] = []
    for i in range(len(vecs)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(vecs)):
            if not assigned[j] and float(np.dot(vecs[i], vecs[j])) >= threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)
    return clusters


def _disambiguate(
    per_record: list[tuple[str, list[tuple[str, str, SpanOffset, int, str]]]],
) -> list[CanonicalEntity]:
    embedder = load_embedder()

    # collect all mentions: (surface, ent_type, offset, atom_id, record_id, atom_value)
    all_mentions: list[tuple[str, str, SpanOffset, int, str, str]] = []
    for record_id, mentions in per_record:
        for surface, ent_type, offset, atom_id, atom_value in mentions:
            all_mentions.append((surface, ent_type, offset, atom_id, record_id, atom_value))

    if not all_mentions:
        return []

    # step 1: canonicalize entity types via embedding at ENTITY_MERGE_THRESHOLD
    distinct_types = list({m[1] for m in all_mentions})
    type_vecs = _batched_encode(embedder, distinct_types)
    type_clusters = _greedy_cluster(type_vecs, ENTITY_MERGE_THRESHOLD)
    # map each original type to its canonical type (first member of cluster)
    type_to_canonical: dict[str, str] = {}
    for cluster in type_clusters:
        canonical_type = distinct_types[cluster[0]]
        for idx in cluster:
            type_to_canonical[distinct_types[idx]] = canonical_type

    # step 2: group mentions by canonical type
    by_type: dict[str, list[tuple[str, SpanOffset, int, str, str]]] = defaultdict(list)
    for surface, ent_type, offset, atom_id, record_id, atom_value in all_mentions:
        canonical_type = type_to_canonical[ent_type]
        by_type[canonical_type].append((surface, offset, atom_id, record_id, atom_value))

    # step 3: within each canonical type, embed full atom context and cluster at ENTITY_MERGE_THRESHOLD
    # then merge across records at GLOBAL_MERGE_THRESHOLD
    global_entities: list[CanonicalEntity] = []
    for canonical_type, mentions in by_type.items():
        # embed atom value (full context) for each mention
        atom_texts = [m[4] for m in mentions]
        vecs = _batched_encode(embedder, atom_texts)

        # local clusters at ENTITY_MERGE_THRESHOLD
        local_clusters = _greedy_cluster(vecs, ENTITY_MERGE_THRESHOLD)

        # represent each local cluster by centroid for global merge
        cluster_vecs = np.array([
            np.mean(vecs[[idx for idx in cluster]], axis=0)
            for cluster in local_clusters
        ])
        cluster_vecs = cluster_vecs / np.linalg.norm(cluster_vecs, axis=1, keepdims=True)

        # global merge at GLOBAL_MERGE_THRESHOLD
        global_clusters = _greedy_cluster(cluster_vecs, GLOBAL_MERGE_THRESHOLD)

        for global_cluster in global_clusters:
            merged_mentions: list[tuple[str, SpanOffset, int, str]] = []
            for local_idx in global_cluster:
                for mention_idx in local_clusters[local_idx]:
                    surface, offset, atom_id, record_id, _ = mentions[mention_idx]
                    merged_mentions.append((surface, offset, atom_id, record_id))
            global_entities.append(CanonicalEntity(canonical_type, merged_mentions))

    return global_entities


async def merge_across_records(
    per_record: list[tuple[str, list[tuple[str, str, SpanOffset, int, str]]]],
) -> list[CanonicalEntity]:
    return await asyncio.to_thread(_disambiguate, per_record)
