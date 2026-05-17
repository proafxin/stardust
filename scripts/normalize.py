import asyncio
import logging
from pathlib import Path

import pyarrow.parquet as pq
from sqlalchemy import text

from stardust.db import SessionLocal
from stardust.parse import normalize_crag, normalize_hotpotqa, normalize_qasper
from stardust.query import insert_index
from stardust.tree.atom import Node

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
N = 100

DATASETS: list[tuple[str, Path, str]] = [
    ("hotpotqa", DATA_DIR / "hotpotqa" / "corpus.parquet", "hotpotqa"),
    # ("qasper", DATA_DIR / "qasper" / "train.parquet", "qasper"),  # TODO: fix chunking
    ("crag_open", DATA_DIR / "crag" / "open" / "train.parquet", "crag_open"),
]

_NORMALIZERS = {"hotpotqa": normalize_hotpotqa, "qasper": normalize_qasper, "crag_open": normalize_crag}


async def main() -> None:
    async with SessionLocal() as session:
        await session.execute(
            text("TRUNCATE tree_nodes, atoms, entity_mentions, canonical_entities RESTART IDENTITY CASCADE")
        )
        await session.commit()
    log.info("tables truncated")

    global_counter = 0
    for dataset_name, parquet_path, id_prefix in DATASETS:
        log.info("normalizing %s", dataset_name)
        records = pq.read_table(parquet_path).to_pylist()
        if N:
            records = records[:N]
        normalizer = _NORMALIZERS[dataset_name]
        docs: list[tuple[str, list[Node], list[int]]] = []
        for i, record in enumerate(records):
            doc_id = f"{id_prefix}_{i}"
            nodes: list[Node] = []
            atoms: list[int] = []
            async for parsed in normalizer(record, global_counter):
                nodes.append(parsed.node)
                if parsed.is_atom:
                    atoms.append(parsed.node.id)
            if nodes:
                global_counter = max(n.id for n in nodes) + 1
            docs.append((doc_id, nodes, atoms))
            if i % 10 == 0:
                log.info("[%s] %d/%d", dataset_name, i + 1, len(records))
        async with SessionLocal() as session:
            await insert_index(docs, session)
        log.info("%s done", dataset_name)


if __name__ == "__main__":
    asyncio.run(main())
