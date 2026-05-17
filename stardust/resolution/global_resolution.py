import asyncio
from collections import defaultdict

import numpy as np

from stardust.config import GLOBAL_MERGE_THRESHOLD
from stardust.registry import embedder as load_embedder
from stardust.tree.atom import SpanOffset


class CanonicalEntity:
    def __init__(self, entity_type: str, mentions: list[tuple[str, SpanOffset, int, str]]) -> None:
        self.entity_type = entity_type
        self.mentions = mentions
        self.canonical_name: str = mentions[0][0] if mentions else ""
        self.aliases: list[str] = list({m[0] for m in mentions})


def _merge_sync(
    per_record: list[tuple[str, dict[str, list[list[tuple[str, SpanOffset, int]]]]]],
) -> list[CanonicalEntity]:
    embedder = load_embedder()
    by_type: dict[str, list[tuple[str, list[tuple[str, SpanOffset, int, str]]]]] = defaultdict(list)

    for record_id, clusters in per_record:
        for ent_type, type_clusters in clusters.items():
            for cluster in type_clusters:
                mentions = [(text, offset, atom_id, record_id) for text, offset, atom_id in cluster]
                by_type[ent_type].append((cluster[0][0], mentions))

    global_entities: list[CanonicalEntity] = []
    for ent_type, items in by_type.items():
        if not items:
            continue
        vecs = embedder.encode([item[0] for item in items], normalize_embeddings=True)
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


async def merge_across_records(
    per_record: list[tuple[str, dict[str, list[list[tuple[str, SpanOffset, int]]]]]],
) -> list[CanonicalEntity]:
    return await asyncio.to_thread(_merge_sync, per_record)
