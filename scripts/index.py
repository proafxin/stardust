import asyncio
import gc
import hashlib
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import cupy
import faiss
import numpy as np
import pyarrow.parquet as pq
import torch
from sqlalchemy import insert, select, text, update
from sqlalchemy.orm import aliased

from stardust.config import (
    ATOM_TOKEN_LIMIT,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_INTERNAL_BATCH_SIZE,
    NLP_COMMIT_BATCH_SIZE,
    NUMERIC_ENTITY_TYPES,
    UNRESOLVED_TOP_K,
)
from stardust.db import SessionLocal
from stardust.extract import extract_batch
from stardust.llm import ollama_complete, ollama_unload
from stardust.models import Atom, BatchPrompt, CanonicalEntity, Disambiguation, Token
from stardust.parse import _token_count, clean_value, normalize_crag, normalize_hotpotqa, normalize_qasper
from stardust.query import insert_canonical_entities, insert_index, insert_tokens
from stardust.registry import embedder as load_embedder
from stardust.registry import nlp as load_nlp
from stardust.registry import unload_embedder, unload_llm_tokenizer, unload_nlp, unload_reranker
from stardust.resolution.global_resolution import canonicalize_by_type
from stardust.resolution.local import build_batch_prompts
from stardust.tree.atom import Node

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_UNRESOLVED_PREAMBLE = (
    "Given an unresolved nominal token and a list of candidate named entities with context, "
    "output the token_id of the named entity this nominal refers to, or null if none apply.\n"
    "Output a single JSON object: {\"referent_id\": <token_id or null>}\n\n"
)

DATA_DIR = Path("data")
N = 200  # 0 = no limit

DATASETS: list[tuple[str, Path, str]] = [
    ("hotpotqa", DATA_DIR / "hotpotqa" / "corpus.parquet", "hotpotqa"),
    # ("qasper", DATA_DIR / "qasper" / "train.parquet", "qasper"),  # TODO: fix chunking
    ("crag_open", DATA_DIR / "crag" / "open" / "train.parquet", "crag_open"),
]

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
                async for atom_id, tokens in extract_batch(atom_ids, texts, clean_starts):
                    await insert_tokens(atom_id, tokens, write_session)
                await write_session.commit()
            done += len(partition)
            log.info("phase 2: %d atoms done, VRAM free %.2fGB", done, torch.cuda.mem_get_info()[0] / 1024**3)


# ── Phase 3: LLM local disambiguation ───────────────────────────────────────


async def _stream_atoms_with_tokens() -> AsyncGenerator[tuple[int, str, list[dict]]]:
    async with SessionLocal() as session, session.begin():
        atom_rows = (await session.execute(
            select(Atom.id, Atom.value)
            .join(Token, Token.atom_id == Atom.id)
            .distinct()
        )).fetchall()

    for row in atom_rows:
        async with SessionLocal() as session:
            token_rows = (await session.execute(
                select(Token.id, Token.token_index, Token.text)
                .where(Token.atom_id == row.id)
                .order_by(Token.token_index)
            )).fetchall()
        yield row.id, row.value, [{"id": t.id, "token_index": t.token_index, "text": t.text} for t in token_rows]


async def phase_llm() -> None:
    log.info("phase 3: llm local disambiguation")
    log.info("phase 3: VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    i = 0
    async for prompt, batch in build_batch_prompts(_stream_atoms_with_tokens()):
        while True:
            try:
                async with SessionLocal() as session:
                    session.add(BatchPrompt(batch_no=i, prompt=prompt))
                    await session.commit()
                raw = await ollama_complete(prompt, max_tokens=max(500, len(batch) * 50))
                try:
                    start, end = raw.find("{"), raw.rfind("}") + 1
                    result = json.loads(raw[start:end]) if start != -1 and end > 0 else {}
                    token_map = {
                        int(k): int(v) for k, v in result.items()
                        if str(k).lstrip("-").isdigit() and isinstance(v, int)
                    }
                except json.JSONDecodeError:
                    log.warning("phase 3: failed to parse LLM response: %s", raw[:200])
                    token_map = {}
                all_token_ids = {t["id"] for _, tokens in batch for t in tokens}
                async with SessionLocal() as session:
                    rows = []
                    for atom_id, tokens in batch:
                        for t in tokens:
                            referent_id = token_map.get(t["id"])
                            if referent_id is None or referent_id not in all_token_ids:
                                continue
                            rows.append({
                                "token_id": t["id"],
                                "referent_id": referent_id,
                                "canonical_token_id": referent_id,
                                "atom_id": atom_id,
                                "confidence": 1.0,
                            })
                    if rows:
                        await session.execute(insert(Disambiguation), rows)
                    await session.commit()
                log.info("phase 3: batch %d done, VRAM free %.2fGB", i + 1, torch.cuda.mem_get_info()[0] / 1024**3)
                break
            except Exception as e:
                log.warning("phase 3: batch %d failed (%s), retrying in 10s", i + 1, e)
                await asyncio.sleep(10)
        i += 1

    log.info("phase 3: done")


# ── Phase 3.5: Transitive closure ────────────────────────────────────────────


async def phase_transitive_closure() -> None:
    log.info("phase 3.5: transitive closure")
    rounds = 0
    while True:
        async with SessionLocal() as session:
            # find disambiguation rows whose canonical_token_id points to another token
            # that itself has a disambiguation row pointing to a named entity
            next_hop = aliased(Disambiguation)
            referent_token = aliased(Token)
            result = await session.execute(
                select(Disambiguation.id, next_hop.canonical_token_id.label("final_id"))
                .join(next_hop, next_hop.token_id == Disambiguation.canonical_token_id)
                .join(referent_token, referent_token.id == next_hop.canonical_token_id)
                .where(referent_token.ent_type.is_not(None))
            )
            rows = result.fetchall()
            if not rows:
                break
            for row in rows:
                await session.execute(
                    update(Disambiguation)
                    .where(Disambiguation.id == row.id)
                    .values(canonical_token_id=row.final_id)
                )
            await session.commit()
            rounds += 1
    log.info("phase 3.5: done (%d rounds)", rounds)


# ── Phase 3.6: Embedding-based resolution for unresolved nominals ────────────


async def phase_unresolved() -> None:
    log.info("phase 3.6: unresolved nominal resolution")
    log.info("phase 3.6: VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    async with SessionLocal() as session, session.begin():
        unresolved_rows = (await session.execute(
            select(Token.id, Token.atom_id, Token.text, Token.context)
            .outerjoin(Disambiguation, Disambiguation.token_id == Token.id)
            .where(Disambiguation.id.is_(None))
        )).fetchall()

        if not unresolved_rows:
            log.info("phase 3.6: no unresolved nominals")
            return

        named_entity_rows = (await session.execute(
            select(Token.id, Token.atom_id, Token.text, Token.context, Token.ent_type)
            .where(Token.ent_type.is_not(None))
            .where(Token.ent_type.not_in(NUMERIC_ENTITY_TYPES))
        )).fetchall()

    if not named_entity_rows:
        log.info("phase 3.6: no named entities to resolve against")
        return

    embedder = load_embedder()
    ne_contexts = [r.context for r in named_entity_rows]
    ne_vecs = embedder.encode(ne_contexts, batch_size=EMBEDDING_INTERNAL_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)

    index = faiss.IndexFlatIP(ne_vecs.shape[1])
    index.add(ne_vecs.astype(np.float32))

    unresolved_contexts = [r.context for r in unresolved_rows]
    unresolved_vecs = embedder.encode(unresolved_contexts, batch_size=EMBEDDING_INTERNAL_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)

    k = min(UNRESOLVED_TOP_K, len(named_entity_rows))
    distances, indices = index.search(unresolved_vecs.astype(np.float32), k)

    i = 0
    async with SessionLocal() as session:
        for unresolved, dists, idxs in zip(unresolved_rows, distances, indices, strict=False):
            candidates = [
                named_entity_rows[j] for j, d in zip(idxs, dists, strict=False) if d > 0.5
            ]
            if not candidates:
                continue
            candidates_str = ", ".join(f"{c.id}:{c.text} ({c.context})" for c in candidates)
            prompt = _UNRESOLVED_PREAMBLE + f"Token {unresolved.id}:{unresolved.text} ({unresolved.context})\nCandidates: {candidates_str}"
            raw = await ollama_complete(prompt, max_tokens=50)
            try:
                start, end = raw.find("{"), raw.rfind("}") + 1
                result = json.loads(raw[start:end]) if start != -1 and end > 0 else {}
                referent_id = result.get("referent_id")
                if referent_id is not None and isinstance(referent_id, int):
                    await session.execute(
                        insert(Disambiguation).values(
                            token_id=unresolved.id,
                            referent_id=referent_id,
                            canonical_token_id=referent_id,
                            atom_id=unresolved.atom_id,
                            confidence=float(dists[0]),
                        )
                    )
                    i += 1
            except (json.JSONDecodeError, ValueError):
                pass
        await session.commit()
    log.info("phase 3.6: resolved %d unresolved nominals", i)


# ── Phase 4: Global canonicalization ─────────────────────────────────────────


async def phase_canonicalization() -> None:
    log.info("phase 4: global canonicalization")

    async with SessionLocal() as session, session.begin():
        rows = (await session.execute(
            select(Token.id, Token.atom_id, Token.text, Token.context, Token.ent_type)
            .where(Token.ent_type.is_not(None))
            .where(Token.ent_type.not_in(NUMERIC_ENTITY_TYPES))
        )).fetchall()

    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r.ent_type, []).append({"id": r.id, "atom_id": r.atom_id, "text": r.text, "context": r.context})

    token_to_atom: dict[int, int] = {r.id: r.atom_id for r in rows}

    async with SessionLocal() as session:
        for ent_type, tokens in by_type.items():
            result = await canonicalize_by_type(ent_type, tokens)
            if not result:
                continue
            entities = []
            for canonical_token_id, member_token_ids in result.items():
                canonical_token = next((t for t in tokens if t["id"] == canonical_token_id), None)
                if not canonical_token:
                    continue
                aliases = list({t["text"] for t in tokens if t["id"] in member_token_ids})
                mentions = [(tid, "global", token_to_atom[tid]) for tid in member_token_ids if tid in token_to_atom]
                entities.append((canonical_token["text"], ent_type, aliases, mentions))
                await session.execute(
                    update(Disambiguation)
                    .where(Disambiguation.canonical_token_id.in_(member_token_ids))
                    .values(canonical_token_id=canonical_token_id)
                )
            await insert_canonical_entities(entities, session)
            await session.commit()
        log.info("phase 4: done")


# ── Phase 5: Embedding ────────────────────────────────────────────────────────


async def phase_embedding() -> None:
    log.info("phase 5: embedding")

    async with SessionLocal() as session, session.begin():
        alias_rows = (await session.execute(
            select(Disambiguation.atom_id, CanonicalEntity.aliases)
            .join(CanonicalEntity, CanonicalEntity.id == Disambiguation.canonical_entity_id)
            .where(Disambiguation.canonical_entity_id.is_not(None))
        )).fetchall()

    atom_aliases: dict[int, list[str]] = {}
    for row in alias_rows:
        atom_aliases.setdefault(row.atom_id, []).extend(row.aliases)

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
            log.info("phase 5: embedded %d atoms", done)

    log.info("phase 5: embedded %d stale atoms", stale)

    async with SessionLocal() as session:
        await session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_atoms_embedding ON atoms "
                "USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )
        await session.commit()
    log.info("phase 5: hnsw index created")
    log.info("phase 5: done")


# ── Main ─────────────────────────────────────────────────────────────────────


async def main(skip_normalize: bool = False, skip_nlp: bool = False, skip_llm: bool = False) -> None:
    if not skip_normalize:
        unload_embedder()
        unload_reranker()
        unload_nlp()
        unload_llm_tokenizer()
        await ollama_unload()
        gc.collect()
        torch.cuda.empty_cache()
        load_embedder()
        await phase_normalize()
    if not skip_nlp:
        unload_embedder()
        unload_reranker()
        unload_nlp()
        unload_llm_tokenizer()
        await ollama_unload()
        gc.collect()
        torch.cuda.empty_cache()
        load_nlp()
        await phase_nlp()
    if not skip_llm:
        unload_embedder()
        unload_reranker()
        unload_nlp()
        unload_llm_tokenizer()
        await ollama_unload()
        gc.collect()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        await phase_llm()
        await phase_transitive_closure()
        unload_embedder()
        unload_reranker()
        unload_nlp()
        unload_llm_tokenizer()
        await ollama_unload()
        gc.collect()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        load_embedder()
        await phase_unresolved()
        unload_embedder()
        unload_reranker()
        unload_nlp()
        unload_llm_tokenizer()
        await ollama_unload()
        gc.collect()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        await phase_canonicalization()
    unload_embedder()
    unload_reranker()
    unload_nlp()
    unload_llm_tokenizer()
    await ollama_unload()
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    load_embedder()
    await phase_embedding()
    unload_embedder()
    unload_reranker()
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
