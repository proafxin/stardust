
import logging

from datasets import load_dataset

from stardust.benchmark.eval import aggregate, mrr, ndcg_at_k, recall_at_k
from stardust.orchestrator import index_documents
from stardust.parse import normalize_qasper
from stardust.retrieval.search import retrieve

log = logging.getLogger(__name__)

K = 10


def _relevant_atoms(index, answer_texts: list[str]) -> set[int]:
    relevant: set[int] = set()
    for atom_id in index.atoms:
        value = index.nodes[atom_id].value.lower()
        for ans in answer_texts:
            if ans and ans.lower() in value:
                relevant.add(atom_id)
                break
    return relevant


def run(split: str = "train", n: int | None = None) -> dict:
    ds = load_dataset("hulki/allenai_qasper", split=split)  # nosec B615
    if n:
        ds = ds.select(range(n))

    recall_scores, mrr_scores, ndcg_scores = [], [], []
    all_clusters: list[tuple[str, dict]] = []

    for i, record in enumerate(ds):
        doc_id = f"qasper_{i}"
        docs = normalize_qasper(record)
        index, _ = index_documents(docs, doc_id, all_clusters)

        questions: list[str] = record["questions"]
        answers: list[list[str]] = record["answers"]

        for q, ans_list in zip(questions, answers, strict=False):
            ans_texts = [a for a in ans_list if a]
            if not ans_texts:
                continue

            relevant = _relevant_atoms(index, ans_texts)
            if not relevant:
                log.warning("no relevant atoms for question: %s", q[:60])
                continue

            results = retrieve(q, index, top_k=K, rerank_top_k=K)
            retrieved = [r.node.id for r in results]

            recall_scores.append(recall_at_k(relevant, retrieved, K))
            mrr_scores.append(mrr(relevant, retrieved))
            ndcg_scores.append(ndcg_at_k(relevant, retrieved, K))

            log.info(
                "[%d] recall@%d=%.3f mrr=%.3f ndcg@%d=%.3f | %s",
                i, K, recall_scores[-1], mrr_scores[-1], K, ndcg_scores[-1], q[:80],
            )

    metrics = {
        f"recall@{K}": aggregate(recall_scores),
        "mrr": aggregate(mrr_scores),
        f"ndcg@{K}": aggregate(ndcg_scores),
        "n": len(recall_scores),
    }
    log.info("QASPER results: %s", metrics)
    return metrics
