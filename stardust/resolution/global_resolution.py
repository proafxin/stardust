import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from stardust.config import CANONICALIZATION_THRESHOLD, EMBEDDING_INTERNAL_BATCH_SIZE

log = logging.getLogger(__name__)


def canonicalize(tokens: list[dict], embedder: SentenceTransformer) -> dict[int, list[int]]:
    if not tokens:
        return {}
    texts = [f"{t['text']} {t['context']}" for t in tokens]
    vecs = embedder.encode(texts, batch_size=EMBEDDING_INTERNAL_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)
    vecs = vecs.astype(np.float32)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    k = min(64, len(tokens))
    distances, indices = index.search(vecs, k)

    assigned = [-1] * len(tokens)
    canonical_id: int = 0
    clusters: dict[int, list[int]] = {}

    for i in range(len(tokens)):
        if assigned[i] != -1:
            continue
        cluster_idx = canonical_id
        canonical_id += 1
        clusters[cluster_idx] = [i]
        assigned[i] = cluster_idx
        for j, dist in zip(indices[i], distances[i], strict=False):
            if j == i or assigned[j] != -1:
                continue
            if dist >= CANONICALIZATION_THRESHOLD:
                clusters[cluster_idx].append(j)
                assigned[j] = cluster_idx

    # pick canonical token as the one with longest text (most complete surface form)
    result: dict[int, list[int]] = {}
    for members in clusters.values():
        canonical_i = max(members, key=lambda i: len(tokens[i]["text"]))
        canonical_token_id = tokens[canonical_i]["id"]
        result[canonical_token_id] = [tokens[i]["id"] for i in members]
    return result
