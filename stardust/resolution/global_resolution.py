import asyncio
import logging
from collections import defaultdict

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


def _encode(embedder, texts: list[str]) -> np.ndarray:
    log.info("encode: %d texts, VRAM free %.2fGB", len(texts), torch.cuda.mem_get_info()[0] / 1024**3)
    
    if not texts:
        return np.array([])
    
    # Let SentenceTransformer handle batching internally with batch_size parameter
    result = embedder.encode(
        texts,
        batch_size=EMBEDDING_INTERNAL_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    
    log.info("encode done, VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)
    return np.array(result)


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
        unique_contexts = list({context for _, _, _, _, context in mentions})
        context_vec_map = dict(zip(unique_contexts, _encode(embedder, unique_contexts)))
        context_vecs = np.array([context_vec_map[context] for _, _, _, _, context in mentions])
        atom_vecs = np.array([atom_vec_map[atom_id] for _, _, atom_id, _, _ in mentions])
        vecs = (atom_vecs + context_vecs) / 2
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        local_clusters = _greedy_cluster(vecs, ENTITY_MERGE_THRESHOLD)
        cluster_vecs = np.array([np.mean(vecs[cluster], axis=0) for cluster in local_clusters])
        cluster_vecs = cluster_vecs / np.linalg.norm(cluster_vecs, axis=1, keepdims=True)
        global_clusters = _greedy_cluster(cluster_vecs, GLOBAL_MERGE_THRESHOLD)

        for global_cluster in global_clusters:
            merged: list[tuple[str, SpanOffset, int, str]] = []
            for local_idx in global_cluster:
                for mention_idx in local_clusters[local_idx]:
                    surface, offset, atom_id, record_id, _ = mentions[mention_idx]
                    merged.append((surface, offset, atom_id, record_id))
            global_entities.append(CanonicalEntity(canonical_type, merged))

    return global_entities


async def merge_across_records(
    per_record: list[tuple[str, list[tuple[str, str, SpanOffset, int, str]]]],
    atom_texts: dict[int, str],
) -> list[CanonicalEntity]:
    return _disambiguate(per_record, atom_texts)
