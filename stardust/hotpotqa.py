import asyncio
import json
import logging
from pathlib import Path

from stardust.eval import aggregate_metrics, compute_metrics
from stardust.db import SessionLocal
from stardust.query import retrieve

log = logging.getLogger(__name__)

DEV_PATH = Path("data/hotpotqa/hotpot_dev_distractor_v1.json")
PRED_PATH = Path("data/hotpotqa/predictions.json")
N = 0
TOP_K = 10


async def _process(record: dict) -> tuple[str, str, list, dict[str, float] | None]:
    qid = record["_id"]
    question = record["question"]
    supporting = {(title, sent_idx) for title, sent_idx in record["supporting_facts"]}
    context_map = dict(record["context"])

    async with SessionLocal() as session:
        results = await retrieve(question, session, top_k=TOP_K, rerank_top_k=TOP_K)

    sp_pred: list[list] = []
    relevant: set[int] = set()

    for r in results:
        for title, sent_idx in supporting:
            if title not in r.raw_text:
                continue
            sents = context_map.get(title, [])
            if sent_idx < len(sents) and sents[sent_idx].strip() in r.raw_text:
                if [title, sent_idx] not in sp_pred:
                    sp_pred.append([title, sent_idx])
                relevant.add(r.id)

    metrics = compute_metrics(relevant, [r.id for r in results]) if relevant else None
    return qid, "n/a", sp_pred, metrics


async def run_async() -> dict:
    with Path(DEV_PATH).open(encoding="utf-8") as f:
        dev = json.load(f)
    if N:
        dev = dev[:N]

    results = await asyncio.gather(*[_process(record) for record in dev])

    answer_pred: dict[str, str] = {}
    sp_pred: dict[str, list] = {}
    all_metrics: list[dict[str, float]] = []

    for qid, answer, sp, metrics in results:
        answer_pred[qid] = answer
        sp_pred[qid] = sp
        if metrics:
            all_metrics.append(metrics)

    PRED_PATH.write_text(json.dumps({"answer": answer_pred, "sp": sp_pred}), encoding="utf-8")
    log.info("predictions written to %s", PRED_PATH)

    agg = {**aggregate_metrics(all_metrics), "n": len(all_metrics)}
    log.info("HotpotQA SP retrieval: %s", agg)
    return agg


def run() -> dict:
    return asyncio.run(run_async())
