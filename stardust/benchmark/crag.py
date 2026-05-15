
import logging

from datasets import load_dataset

from stardust.benchmark.eval import aggregate, mrr, ndcg_at_k, recall_at_k
from stardust.orchestrator import index_documents
from stardust.parse import normalize_crag
from stardust.retrieval.search import retrieve

log = logging.getLogger(__name__)

K = 10

DOMAINS = ["finance", "movie", "music", "open", "sports"]


def _relevant_atoms(index, answer: str) -> set[int]:
    relevant: set[int] = set()
    answer_lower = answer.lower()
    for atom_id in index.atoms:
        if answer_lower in index.nodes[atom_id].value.lower():
            relevant.add(atom_id)
    return relevant


def _run_domain(domain: str, split: str, n: int | None) -> tuple[list, list, list]:
    ds = load_dataset("DataRobot-Research/crag", f"qapairs_{domain}", split=split)  # nosec B615
    if n:
        ds = ds.select(range(n))

    recall_scores, mrr_scores, ndcg_scores = [], [], []
    all_clusters: list[tuple[str, dict]] = []

    for i, record in enumerate(ds):
        doc_id = f"crag_{domain}_{i}"
        docs = normalize_crag(record)
        index, _ = index_documents(docs, doc_id, all_clusters)

        answer: str = record.get("answer", "")
        if not answer:
            continue

        relevant = _relevant_atoms(index, answer)
        if not relevant:
            log.warning("no relevant atoms for query: %s", record["query"][:60])
            continue

        results = retrieve(record["query"], index, top_k=K, rerank_top_k=K)
        retrieved = [r.node.id for r in results]

        recall_scores.append(recall_at_k(relevant, retrieved, K))
        mrr_scores.append(mrr(relevant, retrieved))
        ndcg_scores.append(ndcg_at_k(relevant, retrieved, K))

        log.info(
            "[%s %d] recall@%d=%.3f mrr=%.3f ndcg@%d=%.3f | %s",
            domain, i, K, recall_scores[-1], mrr_scores[-1], K, ndcg_scores[-1],
            record["query"][:80],
        )

    return recall_scores, mrr_scores, ndcg_scores


def run(split: str = "train", n: int | None = None, domains: list[str] | None = None) -> dict:
    domains = domains or DOMAINS
    all_recall, all_mrr, all_ndcg = [], [], []

    for domain in domains:
        log.info("running domain: %s", domain)
        r, m, nd = _run_domain(domain, split, n)
        all_recall.extend(r)
        all_mrr.extend(m)
        all_ndcg.extend(nd)

    metrics = {
        f"recall@{K}": aggregate(all_recall),
        "mrr": aggregate(all_mrr),
        f"ndcg@{K}": aggregate(all_ndcg),
        "n": len(all_recall),
    }
    log.info("CRAG results: %s", metrics)
    return metrics
