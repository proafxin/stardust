import asyncio
import gc
import hashlib
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from sqlalchemy import select, text, update

from stardust.config import (
    ATOM_TOKEN_LIMIT,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_INTERNAL_BATCH_SIZE,
    NLP_COMMIT_BATCH_SIZE,
)
from stardust.db import SessionLocal
from stardust.extract import extract_batch
from stardust.llm import ollama_complete, ollama_unload
from stardust.models import Atom, BatchPrompt
from stardust.parse import _token_count, clean_value, normalize_crag, normalize_hotpotqa, normalize_qasper
from stardust.query import insert_canonical_entities, insert_index
from stardust.registry import embedder as load_embedder
from stardust.registry import nlp as load_nlp
from stardust.registry import unload_embedder, unload_llm_tokenizer, unload_nlp
from stardust.resolution.global_resolution import merge_across_records
from stardust.resolution.local import (
    _nominal_spans,
    build_batch_prompts,
    collect_entity_mentions,
)
from stardust.tree.atom import Node, TokenAttributes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
N = 200  # 0 = no limit

DATASETS: list[tuple[str, Path, str]] = [
    ("hotpotqa", DATA_DIR / "hotpotqa" / "corpus.parquet", "hotpotqa"),
    # ("qasper", DATA_DIR / "qasper" / "train.parquet", "qasper"),  # TODO: fix chunking
    ("crag_open", DATA_DIR / "crag" / "open" / "train.parquet", "crag_open"),
]

DISAMBIGUATION_CACHE = DATA_DIR / "disambiguation_cache.json"


PARQUET_BATCH_SIZE = 10_000

_NORMALIZER_MAP = {
    "hotpotqa": normalize_hotpotqa,
    "qasper": normalize_qasper,
    "crag_open": normalize_crag,
}


async def _flush_atom_buffer(buf_values: list[str], buf_node: Node, record_id: str) -> None:
    bundled = buf_node.model_copy(update={"value": " ".join(buf_values)})
    async with SessionLocal() as session:
        await insert_index([(record_id, [bundled], [0])], session)
    buf_values.clear()


async def phase_normalize() -> None:
    log.info("phase 1: normalize + persist")
    buf_values: list[str] = []
    buf_node: Node | None = None
    done = 0

    for dataset, parquet_path, id_prefix in DATASETS:
        i = 0
        for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=PARQUET_BATCH_SIZE):
            for record in batch.to_pylist():
                if N and i >= N:
                    break
                record_id = f"{id_prefix}_{i}"
                parsed_nodes: list[Any] = [p async for p in _NORMALIZER_MAP[id_prefix](record)]
                if not parsed_nodes:
                    i += 1
                    continue
                non_atoms = [p.node for p in parsed_nodes if not p.is_atom]
                atoms = [p.node for p in parsed_nodes if p.is_atom]
                all_nodes = non_atoms + atoms
                atom_indices = [len(non_atoms) + i for i in range(len(atoms))]
                if all_nodes:
                    async with SessionLocal() as session:
                        await insert_index([(record_id, all_nodes, atom_indices)], session)
                for node in atoms:
                    if buf_values and _token_count(" ".join([*buf_values, node.value])) > ATOM_TOKEN_LIMIT:
                        await _flush_atom_buffer(buf_values, buf_node, record_id)
                        buf_node = None
                    buf_values.append(node.value)
                    buf_node = node.model_copy(update={"parent_index": None})
                    if _token_count(" ".join(buf_values)) >= ATOM_TOKEN_LIMIT:
                        await _flush_atom_buffer(buf_values, buf_node, record_id)
                        buf_node = None
                i += 1
                done += 1
            if N and i >= N:
                break
        log.info("phase 1: %s done (%d records)", dataset, i)

    if buf_values and buf_node is not None:
        await _flush_atom_buffer(buf_values, buf_node, "bundled")

    log.info("phase 1: done (%d total records)", done)


# ── Phase 2: NLP extraction ──────────────────────────────────────────────────


async def phase_nlp() -> None:
    log.info("phase 2: nlp extraction")
    log.info("phase 2: VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    done = 0
    async with SessionLocal() as read_session, read_session.begin():
        stream = await read_session.stream(select(Atom.id, Atom.value, Atom.clean_offset))
        async for partition in stream.partitions(NLP_COMMIT_BATCH_SIZE):
            atom_ids = [r.id for r in partition]
            texts = [r.value for r in partition]
            clean_starts = [r.clean_offset["start"] for r in partition]
            async with SessionLocal() as write_session:
                async for atom_id, attrs in extract_batch(atom_ids, texts, clean_starts):
                    await write_session.execute(
                        update(Atom).where(Atom.id == atom_id).values(nlp_attributes=[a.model_dump() for a in attrs])
                    )
                await write_session.commit()
            done += len(partition)
            log.info("phase 2: %d atoms done, VRAM free %.2fGB", done, torch.cuda.mem_get_info()[0] / 1024**3)


# ── Phase 3: LLM pronoun resolution (Ollama / Qwen3 4B) ─────────────────────


async def _stream_pending_llm_rows() -> AsyncGenerator[tuple[int, str, list]]:
    async with SessionLocal() as session, session.begin():
        stream = await session.stream(
            select(Atom.id, Atom.value, Atom.nlp_attributes).where(Atom.disambiguation.is_(None))
        )
        async for row in stream:
            yield row.id, row.value, row.nlp_attributes


async def phase_llm() -> None:
    log.info("phase 3: llm pronoun resolution")
    log.info("phase 3: VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    if DISAMBIGUATION_CACHE.exists():
        cache = json.loads(DISAMBIGUATION_CACHE.read_text())
        async with SessionLocal() as session:
            for entry in cache:
                await session.execute(
                    update(Atom)
                    .where(Atom.id == entry["id"], Atom.disambiguation.is_(None))
                    .values(disambiguation=entry["disambiguation"])
                )
            await session.commit()
        log.info("phase 3: loaded %d disambiguation entries from cache", len(cache))

    async def _pending_rows():
        async with SessionLocal() as session, session.begin():
            stream = await session.stream(
                select(Atom.id, Atom.value, Atom.nlp_attributes).where(Atom.disambiguation.is_(None))
            )
            async for row in stream:
                yield row.id, row.value, row.nlp_attributes

    i = 0
    async for prompt, batch in build_batch_prompts(_stream_pending_llm_rows()):
        while True:
            try:
                async with SessionLocal() as session:
                    session.add(BatchPrompt(batch_no=i, prompt=prompt))
                    await session.commit()
                raw = await ollama_complete(prompt, max_tokens=max(500, len(batch) * 50))
                try:
                    start, end = raw.find("{"), raw.rfind("}") + 1
                    result = json.loads(raw[start:end]) if start != -1 and end > 0 else {}
                    pronoun_map = {
                        str(k): v for k, v in result.items() if str(k).lstrip("-").isdigit() and isinstance(v, int)
                    }
                except json.JSONDecodeError:
                    log.warning("phase 3: failed to parse LLM response: %s", raw[:200])
                    pronoun_map = {}
                global_tokens: dict[int, tuple[int, TokenAttributes]] = {}
                token_counter = 0
                for atom_id, _value, nlp_attrs in batch:
                    attrs = [TokenAttributes(**a) for a in (nlp_attrs or [])]
                    for nominal in _nominal_spans(attrs):
                        global_tokens[token_counter] = (atom_id, nominal)
                        token_counter += 1
                async with SessionLocal() as session:
                    atom_disambig: dict[int, list] = {atom_id: [] for atom_id, _, _ in batch}
                    for tid_str, rid in pronoun_map.items():
                        tid = int(tid_str)
                        if tid not in global_tokens or rid not in global_tokens:
                            continue
                        atom_id, matched = global_tokens[tid]
                        _, referent_token = global_tokens[rid]
                        atom_disambig[atom_id].append(
                            {
                                "offset": matched.offset.model_dump(),
                                "token": matched.text,
                                "referent": referent_token.text,
                                "confidence": 1.0,
                            }
                        )
                    for atom_id, _value, _ in batch:
                        await session.execute(
                            update(Atom)
                            .where(Atom.id == atom_id)
                            .values(disambiguation={"pronoun_map": atom_disambig[atom_id]})
                        )
                    await session.commit()
                log.info("phase 3: batch %d done, VRAM free %.2fGB", i + 1, torch.cuda.mem_get_info()[0] / 1024**3)
                break
            except Exception as e:
                log.warning("phase 3: batch %d failed (%s), retrying in 10s", i + 1, e)
                await asyncio.sleep(10)
        i += 1

    log.info("phase 3: done")


# ── Phase 4: Disambiguation + embedding ─────────────────────────────────────


async def _collect_entity_mentions() -> list[tuple[str, list]]:
    per_record: dict[str, list] = {}
    async with SessionLocal() as session, session.begin():
        stream = await session.stream(select(Atom.id, Atom.record_id, Atom.nlp_attributes, Atom.disambiguation))
        async for row in stream:
            if row.record_id not in per_record:
                per_record[row.record_id] = []
            per_record[row.record_id].append((row.id, row.nlp_attributes, row.disambiguation))
    return [(record_id, collect_entity_mentions(record_id, rows)) for record_id, rows in per_record.items()]


async def _stream_atom_texts(atom_ids: set[int]) -> dict[int, str]:
    atom_texts: dict[int, str] = {}
    async with SessionLocal() as session, session.begin():
        stream = await session.stream(select(Atom.id, Atom.value).where(Atom.id.in_(atom_ids)))
        async for row in stream:
            atom_texts[row.id] = clean_value(row.value)
    return atom_texts


async def phase_disambiguation() -> None:
    log.info("phase 4: disambiguation + embedding")

    per_record = await _collect_entity_mentions()
    needed_atom_ids = {atom_id for _, mentions in per_record for _, _, _, atom_id, _ in mentions}
    atom_texts = await _stream_atom_texts(needed_atom_ids)

    log.info("before merge_across_records: VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    global_entities = await merge_across_records(per_record, atom_texts)
    del atom_texts
    log.info("phase 4: %d canonical entities", len(global_entities))

    atom_aliases: dict[int, list[str]] = {}
    for entity in global_entities:
        for _, _, atom_id, _ in entity.mentions:
            atom_aliases.setdefault(atom_id, []).extend(entity.aliases)

    async with SessionLocal() as session, session.begin():
        coref_stream = await session.stream(
            select(Atom.id, Atom.disambiguation).where(Atom.disambiguation.is_not(None))
        )
        coref_map: dict[int, list[str]] = {}
        async for row in coref_stream:
            referents = [e["referent"] for e in row.disambiguation.get("pronoun_map", []) if e.get("referent")]
            if referents:
                coref_map[row.id] = referents

    embedder = load_embedder()
    done = stale = 0
    async with SessionLocal() as read_session, read_session.begin():
        stream = await read_session.stream(
            select(Atom.id, Atom.value, Atom.value_hash).execution_options(yield_per=EMBEDDING_BATCH_SIZE)
        )
        async for partition in stream.partitions(EMBEDDING_BATCH_SIZE):
            texts, ids, hashes = [], [], []
            for row in partition:
                parts = [clean_value(row.value)]
                if row.id in atom_aliases:
                    parts.append(" ".join(atom_aliases[row.id]))
                if row.id in coref_map:
                    parts.append(" ".join(coref_map[row.id]))
                enriched = " ".join(parts)
                new_hash = hashlib.sha256(enriched.encode()).hexdigest()
                texts.append(enriched)
                ids.append(row.id)
                hashes.append((row.value_hash, new_hash))
            stale_idx = [i for i, (old, new) in enumerate(hashes) if old != new]
            if stale_idx:
                stale_texts = [texts[i] for i in stale_idx]
                vecs = embedder.encode(
                    stale_texts,
                    batch_size=EMBEDDING_INTERNAL_BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                async with SessionLocal() as write_session:
                    for i, vec in zip(stale_idx, vecs, strict=False):
                        await write_session.execute(
                            update(Atom)
                            .where(Atom.id == ids[i])
                            .values(embedding=vec.tolist(), value=texts[i], value_hash=hashes[i][1])
                        )
                    await write_session.commit()
            stale += len(stale_idx)
            done += len(partition)
            log.info("phase 4: embedded %d/%d atoms", done, done)

    log.info("phase 4: embedded %d stale atoms", stale)

    async with SessionLocal() as session:
        await insert_canonical_entities(global_entities, "global", session)

    async with SessionLocal() as session:
        await session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_atoms_embedding ON atoms "
                "USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )
        await session.commit()
    log.info("phase 4: hnsw index created")

    log.info("phase 4: done")


# ── Main ─────────────────────────────────────────────────────────────────────


async def main(skip_normalize: bool = False, skip_nlp: bool = False, skip_llm: bool = False) -> None:
    if not skip_normalize:
        unload_embedder()
        unload_nlp()
        unload_llm_tokenizer()
        await ollama_unload()
        gc.collect()
        torch.cuda.empty_cache()
        load_embedder()
        await phase_normalize()
    if not skip_nlp:
        unload_embedder()
        unload_nlp()
        unload_llm_tokenizer()
        await ollama_unload()
        gc.collect()
        torch.cuda.empty_cache()
        load_nlp()
        await phase_nlp()
    if not skip_llm:
        unload_embedder()
        unload_nlp()
        unload_llm_tokenizer()
        await ollama_unload()
        gc.collect()
        torch.cuda.empty_cache()
        await phase_llm()
    unload_embedder()
    unload_nlp()
    unload_llm_tokenizer()
    await ollama_unload()
    gc.collect()
    torch.cuda.empty_cache()
    load_embedder()
    await phase_disambiguation()
    unload_embedder()
    unload_nlp()
    unload_llm_tokenizer()
    await ollama_unload()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--skip-nlp", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(skip_normalize=args.skip_normalize, skip_nlp=args.skip_nlp, skip_llm=args.skip_llm))
