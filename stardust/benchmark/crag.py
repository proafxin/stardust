import asyncio
import logging

from datasets import load_dataset

from stardust.benchmark.eval import aggregate_metrics, compute_metrics
from stardust.db import get_session
from stardust.query import retrieve

log = logging.getLogger(__name__)

DOMAINS = ["finance", "movie", "music", "open", "sports"]
SPLIT = "train"
N = 100


async def _process(record: dict, domain: str, i: int) -> dict[str, float] | None:
    answer: str = record.get("answer", "")
    if not answer:
        return None
    async for session in get_session():
        results = await retrieve(record["query"], session, top_k=10, rerank_top_k=10)
    relevant = {r.id for r in results if answer.lower() in r.raw_text.lower()}
    if not relevant:
        log.warning("no relevant atoms for query: %s", record["query"][:60])
        return None
    retrieved = [r.id for r in results]
    m = compute_metrics(relevant, retrieved)
    log.info("[%s %d] %s | %s", domain, i, record["query"][:80], {k: f"{v:.3f}" for k, v in m.items()})
    return m


async def _run_domain(domain: str) -> list[dict[str, float]]:
    ds = load_dataset("DataRobot-Research/crag", f"qapairs_{domain}", split=SPLIT)  # nosec B615
    if N:
        ds = ds.select(range(N))
    results = await asyncio.gather(*[_process(record, domain, i) for i, record in enumerate(ds)])
    return [m for m in results if m is not None]


async def run_async(domains: list[str] | None = None) -> dict:
    domains = domains or DOMAINS
    domain_results = await asyncio.gather(*[_run_domain(d) for d in domains])
    all_metrics = [m for batch in domain_results for m in batch]
    metrics = {**aggregate_metrics(all_metrics), "n": len(all_metrics)}
    log.info("CRAG results: %s", metrics)
    return metrics


def run(domains: list[str] | None = None) -> dict:
    return asyncio.run(run_async(domains))
