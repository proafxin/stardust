import math


def recall_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def precision_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    if not retrieved[:k]:
        return 0.0
    return len(relevant & set(retrieved[:k])) / k


def f1_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    p = precision_at_k(relevant, retrieved, k)
    r = recall_at_k(relevant, retrieved, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mrr(relevant: set[int], retrieved: list[int]) -> float:
    for rank, atom_id in enumerate(retrieved, start=1):
        if atom_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, atom_id in enumerate(retrieved[:k], start=1) if atom_id in relevant)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": sum(scores) / len(scores), "min": min(scores), "max": max(scores)}


def compute_metrics(relevant: set[int], retrieved: list[int]) -> dict[str, float]:
    return {
        "recall@5": recall_at_k(relevant, retrieved, 5),
        "recall@10": recall_at_k(relevant, retrieved, 10),
        "precision@5": precision_at_k(relevant, retrieved, 5),
        "precision@10": precision_at_k(relevant, retrieved, 10),
        "f1@5": f1_at_k(relevant, retrieved, 5),
        "f1@10": f1_at_k(relevant, retrieved, 10),
        "mrr": mrr(relevant, retrieved),
        "ndcg@5": ndcg_at_k(relevant, retrieved, 5),
        "ndcg@10": ndcg_at_k(relevant, retrieved, 10),
    }


def aggregate_metrics(all_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    return {k: aggregate([m[k] for m in all_metrics]) for k in keys}
