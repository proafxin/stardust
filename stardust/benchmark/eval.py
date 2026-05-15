
import math


def recall_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def mrr(relevant: set[int], retrieved: list[int]) -> float:
    for rank, atom_id in enumerate(retrieved, start=1):
        if atom_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, atom_id in enumerate(retrieved[:k], start=1)
        if atom_id in relevant
    )
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": sum(scores) / len(scores), "min": min(scores), "max": max(scores)}
