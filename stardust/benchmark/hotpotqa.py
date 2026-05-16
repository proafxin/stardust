import asyncio
import logging

from datasets import load_dataset

from stardust.benchmark.eval import aggregate_metrics, compute_metrics
from stardust.db import get_session
from stardust.query import retrieve

log = logging.getLogger(__name__)

SPLIT = "train"
N = 100


def _relevant_doc_ids(record: dict) -> set[str]:
    return {f"hotpotqa_{title}" for title in record["supporting_facts"]["title"]}


async def _process(record: dict, i: int) -> dict[str, float] | None:
    async for session in get_session():
        results = await retrieve(record["question"], session, top_k=10, rerank_top_k=10)
    if not results:
        return None
    relevant = {r.id for r in results if any(title in r.value for title in record["supporting_facts"]["title"])}
    if not relevant:
        log.warning("no relevant atoms for record %d, skipping", i)
        return None
    retrieved = [r.id for r in results]
    m = compute_metrics(relevant, retrieved)
    log.info("[%d] %s | %s", i, record["question"][:80], {k: f"{v:.3f}" for k, v in m.items()})
    return m


async def run_async() -> dict:
    ds = load_dataset("hotpot_qa", "fullwiki", split=SPLIT)  # nosec B615
    if N:
        ds = ds.select(range(N))
    results = await asyncio.gather(*[_process(record, i) for i, record in enumerate(ds)])
    all_metrics = [m for m in results if m is not None]
    metrics = {**aggregate_metrics(all_metrics), "n": len(all_metrics)}
    log.info("HotpotQA results: %s", metrics)
    return metrics


def run() -> dict:
    return asyncio.run(run_async())
