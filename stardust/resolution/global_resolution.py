import asyncio
import logging
from collections import defaultdict

import numpy as np
import torch

from stardust.config import ENTITY_MERGE_THRESHOLD, GLOBAL_MERGE_THRESHOLD
from stardust.registry import embedder as load_embedder
from stardust.tree.atom import SpanOffset

log = logging.getLogger(__name__)


class CanonicalEntity:
    def __init__(self, entity_type: str, mentions: list[tuple[str, SpanOffset, int, str]]) -> None:
        self.entity_type = entity_type
        self.mentions = mentions
        self.canonical_name: str = mentions[0][0] if mentions else ""
        self.aliases: list[str] = list({m[0] for m in mentions})


def _dynamic_batches(texts: list[str]) -> list[list[str]]:
    free_bytes = torch.cuda.mem_get_info()[0]
    bytes_per_token = 24 * (1024 * 2 + 512 * 16 * 2) // 10
    batches: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    budget = int(free_bytes * 0.5)
    for text in texts:
        tokens = min(len(text.split()), 512)
        cost = tokens * bytes_per_token
        if current and current_bytes + cost > budget:
            batches.append(current)
            current, current_bytes = [], 0
        current.append(text)
        current_bytes += cost
    if current:
        batches.append(current)
    log.info("dynamic batches: %d texts → %d batches, free VRAM: %.2fGB, budget: %.2fGB, sizes: %s",
             len(texts), len(batches), free_bytes / 1024**3, budget / 1024**3, [len(b) for b in batches])
    return batches


def _batched_encode(embedder, texts: list[str]) -> np.ndarray:
    vecs = []
    for batch in _dynamic_batches(texts):
        with torch.no_grad():
            vecs.append(np.array(embedder.encode(batch, normalize_embeddings=True)))
        torch.cuda.empty_cache()
        log.info("after batch encode: VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)
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
    atom_texts: dict[int, str],
) -> list[CanonicalEntity]:
    embedder = load_embedder()

    # collect all mentions: (surface, ent_type, offset, atom_id, record_id, context)
    all_mentions: list[tuple[str, str, SpanOffset, int, str, str]] = []
    for record_id, mentions in per_record:
        for surface, ent_type, offset, atom_id, context in mentions:
            all_mentions.append((surface, ent_type, offset, atom_id, record_id, context))

    if not all_mentions:
        return []

    # step 1: canonicalize entity types via embedding at ENTITY_MERGE_THRESHOLD
    distinct_types = list({m[1] for m in all_mentions})
    type_vecs = _batched_encode(embedder, distinct_types)
    type_clusters = _greedy_cluster(type_vecs, ENTITY_MERGE_THRESHOLD)
    type_to_canonical: dict[str, str] = {}
    for cluster in type_clusters:
        canonical_type = distinct_types[cluster[0]]
        for idx in cluster:
            type_to_canonical[distinct_types[idx]] = canonical_type
    del type_vecs
    torch.cuda.empty_cache()

    # step 2: group mentions by canonical type
    by_type: dict[str, list[tuple[str, SpanOffset, int, str, str]]] = defaultdict(list)
    for surface, ent_type, offset, atom_id, record_id, context in all_mentions:
        canonical_type = type_to_canonical[ent_type]
        by_type[canonical_type].append((surface, offset, atom_id, record_id, context))

    # step 3: embed atom_value + context per mention, cluster locally then globally
    global_entities: list[CanonicalEntity] = []
    for canonical_type, mentions in by_type.items():
        embed_texts = [
            atom_texts.get(atom_id, "") + " " + context
            for _, _, atom_id, _, context in mentions
        ]
        vecs = _batched_encode(embedder, embed_texts)

        local_clusters = _greedy_cluster(vecs, ENTITY_MERGE_THRESHOLD)

        cluster_vecs = np.array([
            np.mean(vecs[[idx for idx in cluster]], axis=0)
            for cluster in local_clusters
        ])
        cluster_vecs = cluster_vecs / np.linalg.norm(cluster_vecs, axis=1, keepdims=True)

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
    atom_texts: dict[int, str],
) -> list[CanonicalEntity]:
    return await asyncio.to_thread(_disambiguate, per_record, atom_texts)
