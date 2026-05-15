
import logging

from datasets import load_dataset

from stardust.benchmark.eval import aggregate, mrr, ndcg_at_k, recall_at_k
from stardust.orchestrator import index_documents
from stardust.parse import normalize_hotpotqa
from stardust.retrieval.search import retrieve

log = logging.getLogger(__name__)

K = 10


def _relevant_atoms(index, supporting_facts: dict) -> set[int]:
    relevant: set[int] = set()
    pairs = set(zip(supporting_facts["title"], supporting_facts["sent_id"], strict=False))
    for atom_id in index.atoms:
        node = index.nodes[atom_id]
        doc_title = node.metadata.get("title", "")
        for title, _ in pairs:
            if title == doc_title and title in node.value:
                relevant.add(atom_id)
    return relevant


def run(split: str = "train", n: int | None = None) -> dict:
    ds = load_dataset("hotpot_qa", "fullwiki", split=split)  # nosec B615
    if n:
        ds = ds.select(range(n))

    recall_scores, mrr_scores, ndcg_scores = [], [], []
    all_clusters: list[tuple[str, dict]] = []

    for i, record in enumerate(ds):
        doc_id = f"hotpotqa_{i}"
        docs = normalize_hotpotqa(record)
        index, _ = index_documents(docs, doc_id, all_clusters)

        relevant = _relevant_atoms(index, record["supporting_facts"])
        if not relevant:
            log.warning("no relevant atoms found for record %d, skipping", i)
            continue

        results = retrieve(record["question"], index, top_k=K, rerank_top_k=K)
        retrieved = [r.node.id for r in results]

        recall_scores.append(recall_at_k(relevant, retrieved, K))
        mrr_scores.append(mrr(relevant, retrieved))
        ndcg_scores.append(ndcg_at_k(relevant, retrieved, K))

        log.info(
            "[%d/%d] recall@%d=%.3f mrr=%.3f ndcg@%d=%.3f | %s",
            i + 1, len(ds), K, recall_scores[-1], mrr_scores[-1], K, ndcg_scores[-1],
            record["question"][:80],
        )

    metrics = {
        f"recall@{K}": aggregate(recall_scores),
        "mrr": aggregate(mrr_scores),
        f"ndcg@{K}": aggregate(ndcg_scores),
        "n": len(recall_scores),
    }
    log.info("HotpotQA results: %s", metrics)
    return metrics
