import asyncio
import gc
import hashlib
import logging
from pathlib import Path
from typing import Any

import cupy
import pyarrow.parquet as pq
import torch
from sqlalchemy import select, text, update

from stardust.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_INTERNAL_BATCH_SIZE,
    NLP_COMMIT_BATCH_SIZE,
)
from stardust.db import SessionLocal
from stardust.extract import extract_batch
from stardust.models import Atom
from stardust.parse import (
    _parse_md_tables,
    clean_value,
    normalize_hotpotqa,
)
from stardust.query import (
    insert_index,
    insert_table_rows,
    insert_table_signal,
)
from stardust.registry import embedder as load_embedder
from stardust.registry import nlp as load_nlp
from stardust.registry import unload_embedder, unload_nlp, unload_reranker
from stardust.tree.atom import Node

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
N = 200  # 0 = no limit

DATASETS: list[tuple[str, Path, str]] = [
    ("hotpotqa", DATA_DIR / "hotpotqa" / "corpus.parquet", "hotpotqa"),
    # ("qasper", DATA_DIR / "qasper" / "train.parquet", "qasper"),
    # ("crag_open", DATA_DIR / "crag" / "open" / "train.parquet", "crag_open"),
]

PARQUET_BATCH_SIZE = 10_000

_NORMALIZER_MAP = {
    "hotpotqa": normalize_hotpotqa,
}


async def phase_normalize() -> None:
    log.info("phase 1: normalize + persist")
    done = 0

    for dataset, parquet_path, id_prefix in DATASETS:
        i = 0
        for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=PARQUET_BATCH_SIZE):
            for record in batch.to_pylist():
                if N and i >= N:
                    break
                record_id = f"{id_prefix}_{i}"
                parsed_nodes: list[Any] = [p async for p in _NORMALIZER_MAP[id_prefix](record, record_id)]
                if not parsed_nodes:
                    i += 1
                    continue
                non_atoms = [p.node for p in parsed_nodes if not p.is_atom]
                atoms = [p for p in parsed_nodes if p.is_atom]
                all_nodes = non_atoms + [p.node for p in atoms]
                atom_indices = [len(non_atoms) + j for j in range(len(atoms))]
                if all_nodes:
                    async with SessionLocal() as session:
                        await insert_index([(record_id, all_nodes, atom_indices)], session)

                if id_prefix == "crag_open":
                    markdown = record.get("markdown", "") or ""
                    tables = _parse_md_tables(markdown)
                    if tables:
                        async with SessionLocal() as session:
                            for table in tables:
                                signal_id = await insert_table_signal(
                                    record_id, table.title, table.col_names, table.row_count, session
                                )
                                rows = [
                                    (signal_id, row_idx, col_idx, cell)
                                    for row_idx, row_str in enumerate(table.rows)
                                    for col_idx, cell in enumerate(row_str.split(" | "))
                                    if cell
                                ]
                                await insert_table_rows(rows, session)
                            await session.commit()

                i += 1
                done += 1
            if N and i >= N:
                break
        log.info("phase 1: %s done (%d records)", dataset, i)

    log.info("phase 1: done (%d total records)", done)


# ── Phase 2: NLP + disambiguation + embedding ────────────────────────────────


async def phase_nlp_embed() -> None:
    log.info("phase 2: nlp + embed")
    log.info("phase 2: VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    embedder = load_embedder()
    done = stale = 0

    async with SessionLocal() as read_session, read_session.begin():
        stream = await read_session.stream(
            select(Atom.id, Atom.value, Atom.clean_offset, Atom.value_hash)
            .execution_options(yield_per=NLP_COMMIT_BATCH_SIZE)
        )
        async for partition in stream.partitions(NLP_COMMIT_BATCH_SIZE):
            atom_ids = [r.id for r in partition]
            texts = [r.value for r in partition]
            clean_starts = [r.clean_offset["start"] for r in partition]
            old_hashes = [r.value_hash for r in partition]

            enriched_map: dict[int, str] = {}
            async for atom_id, enriched in extract_batch(atom_ids, texts, clean_starts, embedder):
                enriched_map[atom_id] = enriched

            id_to_text = dict(zip(atom_ids, texts, strict=False))
            ids, embed_texts, hashes = [], [], []
            for atom_id, old_hash in zip(atom_ids, old_hashes, strict=False):
                enriched = enriched_map.get(atom_id, clean_value(id_to_text[atom_id]))
                new_hash = hashlib.sha256(enriched.encode()).hexdigest()
                if old_hash != new_hash:
                    ids.append(atom_id)
                    embed_texts.append(enriched)
                    hashes.append(new_hash)

            if ids:
                vecs = embedder.encode(
                    embed_texts,
                    batch_size=EMBEDDING_INTERNAL_BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                async with SessionLocal() as write_session:
                    for atom_id, enriched, new_hash, vec in zip(ids, embed_texts, hashes, vecs, strict=False):
                        await write_session.execute(
                            update(Atom)
                            .where(Atom.id == atom_id)
                            .values(value_enriched=enriched, embedding=vec.tolist(), value_hash=new_hash)
                        )
                    await write_session.commit()
                stale += len(ids)

            done += len(partition)
            log.info("phase 2: %d atoms done, %d stale, VRAM free %.2fGB", done, stale, torch.cuda.mem_get_info()[0] / 1024**3)

    async with SessionLocal() as session:
        await session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_atoms_embedding ON atoms "
                "USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )
        await session.commit()
    log.info("phase 2: hnsw index created")
    log.info("phase 2: done")


# ── Main ─────────────────────────────────────────────────────────────────────


async def main(skip_normalize: bool = False) -> None:
    if not skip_normalize:
        unload_embedder()
        unload_reranker()
        unload_nlp()
        gc.collect()
        torch.cuda.empty_cache()
        await phase_normalize()
    unload_embedder()
    unload_reranker()
    unload_nlp()
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    load_nlp()
    load_embedder()
    await phase_nlp_embed()
    unload_embedder()
    unload_reranker()
    unload_nlp()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-normalize", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(skip_normalize=args.skip_normalize))
