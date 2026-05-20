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
from stardust.extract import embed_with_token_budget, resolve_atoms, run_nlp
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
# per record: (record_id, all_nodes, atom_indices, atom_sentences)
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


async def _insert_record_batch(batch: list[tuple[str, list[Any], list[int], list[list[tuple[str, str, int, str]]]]]) -> list[_Record]:
    records: list[_Record] = []
    async with SessionLocal() as session:
        for record_id, all_nodes, atom_indices, atom_sentences_list in batch:
            atom_db_ids = await insert_index([(record_id, all_nodes, atom_indices)], session)
            atom_sentences: list[_AtomSentences] = []
            for atom_db_id, sentences in zip(atom_db_ids, atom_sentences_list, strict=False):
                sent_ids = await insert_sentences(atom_db_id, sentences, session)
                atom_sentences.append([(raw, resolved, sid) for (raw, resolved, _, _), sid in zip(sentences, sent_ids, strict=False)])
            records.append((record_id, all_nodes, atom_indices, atom_sentences))
        await session.commit()
    return records


PERSIST_BATCH_SIZE = 10000


async def normalize_and_persist() -> list[_Record]:
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
                sents = []
                for raw, resolved in p.sentences:
                    tc = 0  # placeholder, computed after resolution
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
    return all_records


def run_spacy(all_records: list[_Record]) -> dict[tuple[int, int], list[Any]]:
    flat_raw: list[str] = []
    flat_index: list[tuple[int, int]] = []

    for rec_i, (_, _, _, atom_sentences_list) in enumerate(all_records):
        for atom_i, sentences in enumerate(atom_sentences_list):
            for raw, _, _ in sentences:
                flat_raw.append(raw)
                flat_index.append((rec_i, atom_i))

    log.info("spaCy: processing %d sentences", len(flat_raw))
    nlp_results = run_nlp(flat_raw)
    log.info("spaCy: done")

    atom_nlp: dict[tuple[int, int], list[Any]] = {}
    for flat_i, (rec_i, atom_i) in enumerate(flat_index):
        key = (rec_i, atom_i)
        if key not in atom_nlp:
            atom_nlp[key] = []
        atom_nlp[key].append(nlp_results[flat_i])
    return atom_nlp


def resolve_pronouns(
    all_records: list[_Record],
    atom_nlp: dict[tuple[int, int], list[Any]],
    embedder: Any,
    token_budget: int,
) -> list[_Record]:
    needs_resolution = [(k, v) for k, v in atom_nlp.items() if any(morphs for morphs, _ in v if morphs)]
    total_sents = sum(len(v) for v in atom_nlp.values())
    total_unresolved_sents = sum(sum(1 for morphs, _ in v if morphs) for _, v in needs_resolution)
    log.info(
        "resolve: %d/%d sentences need resolution across %d atoms",
        total_unresolved_sents,
        total_sents,
        len(needs_resolution),
    )

    atoms_data = []
    atom_keys: list[tuple[int, int]] = []
    all_raw_texts: list[str] = []
    for (rec_i, atom_i), sent_nlp in needs_resolution:
        sentences = all_records[rec_i][3][atom_i]
        raw_texts = [raw for raw, _, _ in sentences]
        resolved_texts = [resolved for _, resolved, _ in sentences]
        embed_indices = [i for i, (unresolved_morphs, propns) in enumerate(sent_nlp) if unresolved_morphs or propns]
        atoms_data.append((raw_texts, resolved_texts, sent_nlp, embed_indices))
        atom_keys.append((rec_i, atom_i))
        all_raw_texts.extend(raw_texts[i] for i in embed_indices)

    resolvable = 0
    if all_raw_texts:
        unique_texts = list(dict.fromkeys(all_raw_texts))
        text_to_idx = {t: i for i, t in enumerate(unique_texts)}
        unique_vecs = embed_with_token_budget(unique_texts, embedder, token_budget)
        all_vecs = unique_vecs[np.array([text_to_idx[t] for t in all_raw_texts])]
        updated_list, resolvable = resolve_atoms(atoms_data, all_vecs)
        for (rec_i, atom_i), updated, (raw_texts, _, _, _) in zip(atom_keys, updated_list, atoms_data, strict=False):
            sentences = all_records[rec_i][3][atom_i]
            all_records[rec_i][3][atom_i] = [
                (raw, new_resolved, sid)
                for (raw, _, sid), new_resolved in zip(sentences, updated, strict=False)
            ]

    log.info(
        "resolve: %d/%d sentences resolved, VRAM free %.2fGB",
        resolvable,
        total_unresolved_sents,
        torch.cuda.mem_get_info()[0] / 1024**3,
    )
    return all_records


async def persist_resolutions(all_records: list[_Record], embedder: Any) -> None:
    updates: list[tuple[int, str, int, str]] = []
    for _, _, _, atom_sentences_list in all_records:
        for sentences in atom_sentences_list:
            for _, resolved, sid in sentences:
                tc = len(embedder.tokenizer([resolved], add_special_tokens=True)["input_ids"][0])
                vh = hashlib.sha256(resolved.encode()).hexdigest()
                updates.append((sid, resolved, tc, vh))

    batch_size = 1000
    async with SessionLocal() as session:
        for start in range(0, len(updates), batch_size):
            await update_resolved_texts(updates[start: start + batch_size], session)
        await session.commit()
    log.info("persist_resolutions: %d sentences updated", len(updates))


async def embed_and_persist(all_records: list[_Record], embedder: Any, token_budget: int) -> None:
    flat: list[tuple[int, str]] = []
    for _, _, _, atom_sentences_list in all_records:
        for sentences in atom_sentences_list:
            for _, resolved, sid in sentences:
                flat.append((sid, resolved))

    log.info("embed: %d resolved texts, VRAM free %.2fGB", len(flat), torch.cuda.mem_get_info()[0] / 1024**3)
    all_resolved = [resolved for _, resolved in flat]
    all_vecs = embed_with_token_budget(all_resolved, embedder, token_budget)
    log.info("embed: done, VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    rows = [(sid, all_vecs[i]) for i, (sid, _) in enumerate(flat)]
    batch_size = 5000
    async with SessionLocal() as session:
        for start in range(0, len(rows), batch_size):
            await insert_embeddings(rows[start: start + batch_size], session)
            await session.commit()
            log.info("embed: inserted %d/%d embeddings", min(start + batch_size, len(rows)), len(rows))


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

    load_nlp()
    log.info("spaCy: loaded, VRAM free %.2fGB", _vram_free_gb())
    atom_nlp = run_spacy(all_records)
    unload_nlp()
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    log.info("spaCy: unloaded, VRAM free %.2fGB", _vram_free_gb())

    token_budget = _load_token_budget()
    embedder = load_embedder()
    log.info("embedder loaded, VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    all_records = resolve_pronouns(all_records, atom_nlp, embedder, token_budget)
    await persist_resolutions(all_records, embedder)

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
