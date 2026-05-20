import asyncio
import gc
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import cupy
import numpy as np
import torch
from sqlalchemy import text

from stardust.db import SessionLocal
from stardust.extract import embed_with_token_budget, resolve_atom_coref
from stardust.parse import normalize_hotpotqa
from stardust.query import insert_embeddings, insert_index, insert_sentences, update_resolved_texts
from stardust.registry import embedder as load_embedder
from stardust.registry import nlp as load_nlp
from stardust.registry import unload_embedder, unload_nlp

TUNING_PATH = Path("tuning.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
N = 1000  # 0 = no limit

DATASETS: list[tuple[str, Path, str]] = [
    ("hotpotqa", DATA_DIR / "hotpotqa" / "hotpot_train_v1.1.json", "hotpotqa"),
]

_NORMALIZER_MAP = {
    "hotpotqa": normalize_hotpotqa,
}

# per atom: list of (raw, resolved, sentence_db_id)
_AtomSentences = list[tuple[str, str, int]]
_Record = tuple[str, list[Any], list[int], list[_AtomSentences]]


def _vram_free_gb() -> float:
    free_torch, _ = torch.cuda.mem_get_info()
    cupy_used = cupy.get_default_memory_pool().used_bytes()
    return (free_torch - cupy_used) / 1024**3


def _load_token_budget() -> int:
    if not TUNING_PATH.exists():
        raise RuntimeError("tuning.json not found")
    data = json.loads(TUNING_PATH.read_text(encoding="utf-8"))
    budget = data.get("embedding_token_budget")
    if budget is None:
        raise RuntimeError("embedding_token_budget required in tuning.json")
    return budget


async def _insert_record_batch(
    batch: list[tuple[str, list[Any], list[int], list[list[tuple[str, str, int, str]]]]],
) -> list[_Record]:
    records: list[_Record] = []
    async with SessionLocal() as session:
        for record_id, all_nodes, atom_indices, atom_sentences_list in batch:
            atom_db_ids = await insert_index([(record_id, all_nodes, atom_indices)], session)
            atom_sentences: list[_AtomSentences] = []
            for atom_db_id, sentences in zip(atom_db_ids, atom_sentences_list, strict=False):
                sent_ids = await insert_sentences(atom_db_id, sentences, session)
                atom_sentences.append([
                    (raw, resolved, sid)
                    for (raw, resolved, _, _), sid in zip(sentences, sent_ids, strict=False)
                ])
            records.append((record_id, all_nodes, atom_indices, atom_sentences))
        await session.commit()
    return records


PERSIST_BATCH_SIZE = 10000


async def normalize_and_persist() -> list[_Record]:
    nlp = load_nlp()
    log.info("spaCy+coref: loaded, VRAM free %.2fGB", _vram_free_gb())
    all_records: list[_Record] = []

    for dataset, data_path, id_prefix in DATASETS:
        with Path(data_path).open(encoding="utf-8") as f:
            raw_records = json.load(f)
        count = 0
        batch: list[tuple[str, list[Any], list[int], list[list[tuple[str, str, int, str]]]]] = []
        batch_sentences = 0

        for i, record in enumerate(raw_records):
            if N and i >= N:
                break
            record_id = record.get("_id", f"{id_prefix}_{i}")
            parsed_nodes: list[Any] = [p async for p in _NORMALIZER_MAP[id_prefix](record, record_id)]
            if not parsed_nodes:
                continue
            non_atoms = [p.node for p in parsed_nodes if not p.is_atom]
            atoms = [p for p in parsed_nodes if p.is_atom]
            all_nodes = non_atoms + [p.node for p in atoms]
            atom_indices = [len(non_atoms) + j for j in range(len(atoms))]

            atom_sentences_list: list[list[tuple[str, str, int, str]]] = []
            for p in atoms:
                resolved_sentences = resolve_atom_coref(p.sentences, nlp)
                sents = []
                for raw, resolved in resolved_sentences:
                    tc = 0  # computed after resolution during embed phase
                    vh = hashlib.sha256(resolved.encode()).hexdigest()
                    sents.append((raw, resolved, tc, vh))
                atom_sentences_list.append(sents)

            record_sentences = sum(len(s) for s in atom_sentences_list)
            if batch_sentences + record_sentences > PERSIST_BATCH_SIZE and batch:
                all_records.extend(await _insert_record_batch(batch))
                batch, batch_sentences = [], 0
            batch.append((record_id, all_nodes, atom_indices, atom_sentences_list))
            batch_sentences += record_sentences
            count += 1

        if batch:
            all_records.extend(await _insert_record_batch(batch))
        log.info("normalize+persist: %s %d records", dataset, count)

    unload_nlp()
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    log.info("spaCy+coref: unloaded, VRAM free %.2fGB", _vram_free_gb())
    return all_records


async def embed_and_persist(all_records: list[_Record], embedder: Any, token_budget: int) -> None:
    flat: list[tuple[int, str]] = []
    for _, _, _, atom_sentences_list in all_records:
        for sentences in atom_sentences_list:
            for _, resolved, sid in sentences:
                flat.append((sid, resolved))

    log.info("embed: %d resolved texts, VRAM free %.2fGB", len(flat), torch.cuda.mem_get_info()[0] / 1024**3)
    all_resolved = [resolved for _, resolved in flat]
    all_token_counts = [len(ids) for ids in embedder.tokenizer(all_resolved, add_special_tokens=True)["input_ids"]]
    all_vecs = embed_with_token_budget(all_resolved, embedder, token_budget)
    log.info("embed: done, VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    updates = [
        (sid, resolved, all_token_counts[i], hashlib.sha256(resolved.encode()).hexdigest())
        for i, (sid, resolved) in enumerate(flat)
    ]
    async with SessionLocal() as session:
        for start in range(0, len(updates), 1000):
            await update_resolved_texts(updates[start: start + 1000], session)
        await session.commit()

    rows = [(sid, all_vecs[i]) for i, (sid, _) in enumerate(flat)]
    async with SessionLocal() as session:
        for start in range(0, len(rows), 5000):
            await insert_embeddings(rows[start: start + 5000], session)
            await session.commit()
            log.info("embed: inserted %d/%d embeddings", min(start + 5000, len(rows)), len(rows))


async def drop_hnsw_index() -> None:
    async with SessionLocal() as session:
        await session.execute(text("DROP INDEX IF EXISTS ix_sentence_embeddings_embedding"))
        await session.commit()
    log.info("HNSW index dropped")


async def build_hnsw_index() -> None:
    async with SessionLocal() as session:
        await session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_sentence_embeddings_embedding ON sentence_embeddings "
                "USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 32, ef_construction = 128)"
            )
        )
        await session.commit()
    log.info("HNSW index created")


async def main() -> None:
    log.info("stardust index: start")
    unload_embedder()
    unload_nlp()
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    log.info("VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    all_records = await normalize_and_persist()

    token_budget = _load_token_budget()
    embedder = load_embedder()
    log.info("embedder loaded, VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    await drop_hnsw_index()
    await embed_and_persist(all_records, embedder, token_budget)
    await build_hnsw_index()

    unload_embedder()
    gc.collect()
    torch.cuda.empty_cache()
    log.info("embedder unloaded, VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)
    log.info("stardust index: done")


if __name__ == "__main__":
    asyncio.run(main())
