import logging

import faiss
import numpy as np

from stardust.config import CANONICALIZATION_THRESHOLD

log = logging.getLogger(__name__)


def canonicalize(tokens: list[dict], vecs: np.ndarray) -> dict[int, list[int]]:
    if not tokens:
        return {}
    vecs = vecs.astype(np.float32)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    k = min(64, len(tokens))
    distances, indices = index.search(vecs, k)

    assigned = [-1] * len(tokens)
    canonical_id = 0
    clusters: dict[int, list[int]] = {}

    for i in range(len(tokens)):
        if assigned[i] != -1:
            continue
        clusters[canonical_id] = [i]
        assigned[i] = canonical_id
        for j, dist in zip(indices[i], distances[i], strict=False):
            if j == i or assigned[j] != -1:
                continue
            if dist >= CANONICALIZATION_THRESHOLD:
                clusters[canonical_id].append(j)
                assigned[j] = canonical_id
        canonical_id += 1

    result: dict[int, list[int]] = {}
    for members in clusters.values():
        canonical_i = max(members, key=lambda i: len(tokens[i]["text"]))
        canonical_token_id = tokens[canonical_i]["id"]
        result[canonical_token_id] = [tokens[i]["id"] for i in members]
    return result
