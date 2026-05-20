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
from stardust.extract import embed_sentences, resolve_atoms, run_nlp
from stardust.parse import normalize_hotpotqa
from stardust.query import insert_index, insert_sentences
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

# atom_sentences at normalize stage: per atom, list of (raw, resolved)
_RawRecord = tuple[str, list[Any], list[int], list[list[tuple[str, str]]]]
# atom_sentences at final stage: per atom, list of (raw, resolved, tc, hash, vec)
_FinalRecord = tuple[str, list[Any], list[int], list[list[tuple[str, str, int, str, np.ndarray]]]]


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


async def normalize_records() -> list[_RawRecord]:
    all_records: list[_RawRecord] = []
    for dataset, data_path, id_prefix in DATASETS:
        with Path(data_path).open(encoding="utf-8") as f:
            records = json.load(f)
        count = 0
        for i, record in enumerate(records):
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
            atom_sentences = [p.sentences for p in atoms]
            all_records.append((record_id, all_nodes, atom_indices, atom_sentences))
            count += 1
        log.info("normalize: %s %d records", dataset, count)
    return all_records


def run_spacy(all_records: list[_RawRecord]) -> dict[tuple[int, int], list[tuple[bool, list[str]]]]:
    flat_raw: list[str] = []
    flat_index: list[tuple[int, int]] = []  # (rec_i, atom_i)

    for rec_i, (_, _, _, atom_sentences_list) in enumerate(all_records):
        for atom_i, sentences in enumerate(atom_sentences_list):
            for raw, _ in sentences:
                flat_raw.append(raw)
                flat_index.append((rec_i, atom_i))

    total = len(flat_raw)
    log.info("spaCy: processing %d sentences", total)
    nlp_results: list[tuple[bool, list[str]]] = run_nlp(flat_raw)
    log.info("spaCy: done")

    atom_nlp: dict[tuple[int, int], list[tuple[bool, list[str]]]] = {}
    for flat_i, (rec_i, atom_i) in enumerate(flat_index):
        key = (rec_i, atom_i)
        if key not in atom_nlp:
            atom_nlp[key] = []
        atom_nlp[key].append(nlp_results[flat_i])
    return atom_nlp


def resolve_pronouns(
    all_records: list[_RawRecord],
    atom_nlp: dict[tuple[int, int], list[tuple[bool, list[str]]]],
) -> list[_RawRecord]:
    needs_resolution = [(k, v) for k, v in atom_nlp.items() if any(hu for hu, _ in v)]
    total_sents = sum(len(v) for v in atom_nlp.values())
    total_unresolved_sents = sum(sum(1 for hu, _ in v if hu) for _, v in needs_resolution)
    log.info(
        "resolve: %d/%d sentences need resolution across %d atoms",
        total_unresolved_sents,
        total_sents,
        len(needs_resolution),
    )
    embedder = load_embedder()
    log.info("resolve: embedder loaded, VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    atoms_data: list[tuple[list[str], list[str], list[tuple[bool, list[str]]]]] = []
    atom_keys: list[tuple[int, int]] = []
    all_raw_texts: list[str] = []
    for (rec_i, atom_i), sent_nlp in needs_resolution:
        sentences = all_records[rec_i][3][atom_i]
        raw_texts = [raw for raw, _ in sentences]
        resolved_texts = [resolved for _, resolved in sentences]
        atoms_data.append((raw_texts, resolved_texts, sent_nlp))
        atom_keys.append((rec_i, atom_i))
        all_raw_texts.extend(raw_texts)

    all_vecs = embed_sentences(all_raw_texts, embedder)
    updated_list, resolvable = resolve_atoms(atoms_data, all_vecs)

    for (rec_i, atom_i), updated, (raw_texts, _, _) in zip(atom_keys, updated_list, atoms_data, strict=False):
        all_records[rec_i][3][atom_i] = list(zip(raw_texts, updated, strict=False))

    unload_embedder()
    gc.collect()
    torch.cuda.empty_cache()
    log.info(
        "resolve: %d/%d sentences resolved, embedder unloaded, VRAM free %.2fGB",
        resolvable,
        total_unresolved_sents,
        torch.cuda.mem_get_info()[0] / 1024**3,
    )
    return all_records


def embed_and_finalize(all_records: list[_RawRecord]) -> list[_FinalRecord]:
    token_budget = _load_token_budget()

    flat: list[tuple[int, int, int, str]] = []
    for rec_i, (_, _, _, atom_sentences_list) in enumerate(all_records):
        for atom_i, sentences in enumerate(atom_sentences_list):
            for sent_i, (_, resolved) in enumerate(sentences):
                flat.append((rec_i, atom_i, sent_i, resolved))

    embedder = load_embedder()
    log.info("embed: %d resolved texts, VRAM free %.2fGB", len(flat), torch.cuda.mem_get_info()[0] / 1024**3)

    all_resolved = [resolved for _, _, _, resolved in flat]
    all_token_counts = [len(ids) for ids in embedder.tokenizer(all_resolved, add_special_tokens=True)["input_ids"]]

    all_vecs: list[np.ndarray] = [None] * len(flat)  # type: ignore[list-item]
    batch_indices: list[int] = []
    batch_tokens = 0
    batches_done = 0
    for i, tc in enumerate(all_token_counts):
        if batch_tokens + tc > token_budget and batch_indices:
            vecs = embed_sentences([all_resolved[j] for j in batch_indices], embedder)
            for j, vec in zip(batch_indices, vecs, strict=False):
                all_vecs[j] = vec
            batches_done += 1
            log.info(
                "embed: batch %d done (%d sentences, %d tokens), VRAM free %.2fGB",
                batches_done,
                len(batch_indices),
                batch_tokens,
                torch.cuda.mem_get_info()[0] / 1024**3,
            )
            batch_indices, batch_tokens = [], 0
        batch_indices.append(i)
        batch_tokens += tc
    if batch_indices:
        vecs = embed_sentences([all_resolved[j] for j in batch_indices], embedder)
        for j, vec in zip(batch_indices, vecs, strict=False):
            all_vecs[j] = vec
        batches_done += 1
        log.info(
            "embed: batch %d done (%d sentences, %d tokens), VRAM free %.2fGB",
            batches_done,
            len(batch_indices),
            batch_tokens,
            torch.cuda.mem_get_info()[0] / 1024**3,
        )

    unload_embedder()
    gc.collect()
    torch.cuda.empty_cache()
    log.info(
        "embed: done (%d batches), embedder unloaded, VRAM free %.2fGB",
        batches_done,
        torch.cuda.mem_get_info()[0] / 1024**3,
    )

    vec_map: dict[tuple[int, int, int], tuple[int, np.ndarray]] = {
        (rec_i, atom_i, sent_i): (all_token_counts[fi], all_vecs[fi])
        for fi, (rec_i, atom_i, sent_i, _) in enumerate(flat)
    }
    final_records: list[_FinalRecord] = []
    for rec_i, (record_id, all_nodes, atom_indices, atom_sentences_list) in enumerate(all_records):
        final_atom_sentences: list[list[tuple[str, str, int, str, np.ndarray]]] = []
        for atom_i, sentences in enumerate(atom_sentences_list):
            final_sents: list[tuple[str, str, int, str, np.ndarray]] = []
            for sent_i, (raw, resolved) in enumerate(sentences):
                tc, vec = vec_map[rec_i, atom_i, sent_i]
                value_hash = hashlib.sha256(resolved.encode()).hexdigest()
                final_sents.append((raw, resolved, tc, value_hash, vec))
            final_atom_sentences.append(final_sents)
        final_records.append((record_id, all_nodes, atom_indices, final_atom_sentences))
    return final_records


async def _write_batch(batch: list[_FinalRecord]) -> int:
    total = 0
    async with SessionLocal() as session:
        for record_id, all_nodes, atom_indices, atom_sentences_list in batch:
            atom_db_ids = await insert_index([(record_id, all_nodes, atom_indices)], session)
            for atom_db_id, sentences in zip(atom_db_ids, atom_sentences_list, strict=False):
                await insert_sentences(atom_db_id, sentences, session)
                total += len(sentences)
        await session.commit()
    return total


PERSIST_BATCH_SIZE = 10000  # max sentences per DB transaction


async def persist(final_records: list[_FinalRecord]) -> None:
    batch: list[_FinalRecord] = []
    batch_sentences = 0
    total_sentences = 0

    for record in final_records:
        record_sentences = sum(len(sents) for sents in record[3])
        if batch_sentences + record_sentences > PERSIST_BATCH_SIZE and batch:
            total_sentences += await _write_batch(batch)
            batch, batch_sentences = [], 0
        batch.append(record)
        batch_sentences += record_sentences

    total_sentences += await _write_batch(batch)
    log.info("%d sentences written to DB", total_sentences)


async def build_hnsw_index() -> None:
    async with SessionLocal() as session:
        await session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_sentences_embedding ON sentences "
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

    all_records = await normalize_records()

    load_nlp()
    log.info("spaCy: loaded, VRAM free %.2fGB", _vram_free_gb())
    atom_nlp = run_spacy(all_records)
    unload_nlp()
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    log.info("spaCy: unloaded, VRAM free %.2fGB", _vram_free_gb())

    all_records = resolve_pronouns(all_records, atom_nlp)
    final_records = embed_and_finalize(all_records)

    await persist(final_records)
    await build_hnsw_index()

    unload_embedder()
    unload_nlp()
    gc.collect()
    torch.cuda.empty_cache()
    log.info("stardust index: done")


if __name__ == "__main__":
    asyncio.run(main())
