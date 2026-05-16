import asyncio
import logging

from datasets import load_dataset

from stardust.db import get_session
from stardust.orchestrator import index_documents
from stardust.parse import normalize_crag, normalize_hotpotqa, normalize_qasper
from stardust.query import insert_canonical_entities, insert_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET = "hotpotqa"
SPLIT = "train"
N = 100


async def index_hotpotqa() -> None:
    ds = load_dataset("hotpot_qa", "fullwiki", split=SPLIT)  # nosec B615
    if N:
        ds = ds.select(range(N))
    all_clusters: list[tuple[str, dict]] = []
    async for session in get_session():
        for i, record in enumerate(ds):
            doc_id = f"hotpotqa_{i}"
            log.info("indexing %s", doc_id)
            index, entities = await index_documents(normalize_hotpotqa(record), doc_id, all_clusters)
            await insert_index(index, doc_id, session)
            await insert_canonical_entities(entities, doc_id, session)


async def index_qasper() -> None:
    ds = load_dataset("hulki/allenai_qasper", split=SPLIT)  # nosec B615
    if N:
        ds = ds.select(range(N))
    all_clusters: list[tuple[str, dict]] = []
    async for session in get_session():
        for i, record in enumerate(ds):
            doc_id = f"qasper_{i}"
            log.info("indexing %s", doc_id)
            index, entities = await index_documents(normalize_qasper(record), doc_id, all_clusters)
            await insert_index(index, doc_id, session)
            await insert_canonical_entities(entities, doc_id, session)


async def index_crag() -> None:
    domains = ["finance", "movie", "music", "open", "sports"]
    all_clusters: list[tuple[str, dict]] = []
    async for session in get_session():
        for domain in domains:
            ds = load_dataset("DataRobot-Research/crag", f"qapairs_{domain}", split=SPLIT)  # nosec B615
            if N:
                ds = ds.select(range(N))
            for i, record in enumerate(ds):
                doc_id = f"crag_{domain}_{i}"
                log.info("indexing %s", doc_id)
                index, entities = await index_documents(normalize_crag(record), doc_id, all_clusters)
                await insert_index(index, doc_id, session)
                await insert_canonical_entities(entities, doc_id, session)


async def main() -> None:
    if DATASET == "hotpotqa":
        await index_hotpotqa()
    elif DATASET == "qasper":
        await index_qasper()
    elif DATASET == "crag":
        await index_crag()
    else:
        log.error("unknown dataset: %s", DATASET)


if __name__ == "__main__":
    asyncio.run(main())
