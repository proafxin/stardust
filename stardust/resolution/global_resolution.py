from collections import defaultdict

import numpy as np
from config import GLOBAL_MERGE_THRESHOLD

from stardust.registry import get_embedder
from stardust.tree.atom import SpanOffset


class CanonicalEntity:
    def __init__(self, entity_type: str, mentions: list[tuple[str, SpanOffset, int, str]]) -> None:
        self.entity_type = entity_type
        self.mentions = mentions
        self.canonical_name: str = mentions[0][0] if mentions else ""
        self.aliases: list[str] = list({m[0] for m in mentions})

    @property
    def representative_text(self) -> str:
        return f"{self.entity_type}: {' | '.join(self.aliases[:5])}"


def resolve_global(
    local_clusters: dict[str, list[list[tuple[str, SpanOffset, int]]]],
    doc_id: str,
) -> list[CanonicalEntity]:
    embedder = get_embedder()
    canonical_entities: list[CanonicalEntity] = []

    for ent_type, clusters in local_clusters.items():
        if not clusters:
            continue

        representatives = [cluster[0][0] for cluster in clusters]
        vecs = embedder.encode(representatives, normalize_embeddings=True)

        merged = [False] * len(clusters)
        for i in range(len(clusters)):
            if merged[i]:
                continue
            merged_mentions: list[tuple[str, SpanOffset, int, str]] = [
                (text, offset, atom_id, doc_id) for text, offset, atom_id in clusters[i]
            ]
            for j in range(i + 1, len(clusters)):
                if not merged[j] and float(np.dot(vecs[i], vecs[j])) >= GLOBAL_MERGE_THRESHOLD:
                    merged_mentions.extend((text, offset, atom_id, doc_id) for text, offset, atom_id in clusters[j])
                    merged[j] = True
            canonical_entities.append(CanonicalEntity(ent_type, merged_mentions))

    return canonical_entities


def merge_across_documents(
    per_doc: list[tuple[str, dict[str, list[list[tuple[str, SpanOffset, int]]]]]],
) -> list[CanonicalEntity]:
    embedder = get_embedder()
    by_type: dict[str, list[tuple[str, list[tuple[str, SpanOffset, int, str]]]]] = defaultdict(list)

    for doc_id, clusters in per_doc:
        for ent_type, type_clusters in clusters.items():
            for cluster in type_clusters:
                mentions = [(text, offset, atom_id, doc_id) for text, offset, atom_id in cluster]
                representative = cluster[0][0]
                by_type[ent_type].append((representative, mentions))

    global_entities: list[CanonicalEntity] = []

    for ent_type, items in by_type.items():
        if not items:
            continue
        texts = [item[0] for item in items]
        vecs = embedder.encode(texts, normalize_embeddings=True)
        merged = [False] * len(items)

        for i in range(len(items)):
            if merged[i]:
                continue
            all_mentions = list(items[i][1])
            for j in range(i + 1, len(items)):
                if not merged[j] and float(np.dot(vecs[i], vecs[j])) >= GLOBAL_MERGE_THRESHOLD:
                    all_mentions.extend(items[j][1])
                    merged[j] = True
            global_entities.append(CanonicalEntity(ent_type, all_mentions))

    return global_entities
