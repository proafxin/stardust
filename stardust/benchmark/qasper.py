import asyncio
import logging

from datasets import load_dataset

from stardust.benchmark.eval import aggregate_metrics, compute_metrics
from stardust.db import get_session
from stardust.query import retrieve

log = logging.getLogger(__name__)

SPLIT = "train"
N = 100


async def _process(record: dict, i: int) -> list[dict[str, float]]:
    per_question: list[dict[str, float]] = []
    async for session in get_session():
        for q, ans_list in zip(record["questions"], record["answers"], strict=False):
            ans_texts = [a for a in ans_list if a]
            if not ans_texts:
                continue
            results = await retrieve(q, session, top_k=10, rerank_top_k=10)
            relevant = {r.id for r in results if any(ans.lower() in r.value.lower() for ans in ans_texts)}
            if not relevant:
                log.warning("no relevant atoms for question: %s", q[:60])
                continue
            retrieved = [r.id for r in results]
            m = compute_metrics(relevant, retrieved)
            log.info("[%d] %s | %s", i, q[:80], {k: f"{v:.3f}" for k, v in m.items()})
            per_question.append(m)
    return per_question


async def run_async() -> dict:
    ds = load_dataset("hulki/allenai_qasper", split=SPLIT)  # nosec B615
    if N:
        ds = ds.select(range(N))
    results = await asyncio.gather(*[_process(record, i) for i, record in enumerate(ds)])
    all_metrics = [m for batch in results for m in batch]
    metrics = {**aggregate_metrics(all_metrics), "n": len(all_metrics)}
    log.info("QASPER results: %s", metrics)
    return metrics


def run() -> dict:
    return asyncio.run(run_async())
