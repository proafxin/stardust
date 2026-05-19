import asyncio
import contextlib
import gc
import hashlib
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from redis.asyncio import Redis
from sqlalchemy import select, text, update

from stardust.config import (
    ATOM_TOKEN_LIMIT,
    EMBEDDING_INTERNAL_BATCH_SIZE,
    NLP_COMMIT_BATCH_SIZE,
    settings,
)
from stardust.db import SessionLocal
from stardust.extract import extract_batch
from stardust.llm import ollama_complete, ollama_unload
from stardust.models import AtomModel, BatchPromptModel
from stardust.parse import _token_count, clean_value, normalize_crag, normalize_hotpotqa, normalize_qasper
from stardust.query import insert_canonical_entities, insert_index
from stardust.registry import embedder as load_embedder
from stardust.registry import nlp as load_nlp
from stardust.registry import unload_embedder, unload_nlp
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


NORMALIZE_WORKERS = 4
BATCH_WORKERS = 2
INSERT_WORKERS = 4
PARQUET_BATCH_SIZE = 10_000
INSERT_BATCH_SIZE = 50_000
READ_STREAM = "stardust:stream:read"
NORMALIZE_STREAM = "stardust:stream:normalize"
INSERT_STREAM = "stardust:stream:insert"
READ_GROUP = "stardust:read:group"
NORMALIZE_GROUP = "stardust:normalize:group"
INSERT_GROUP = "stardust:insert:group"
READ_DONE_KEY = "stardust:read_done"
NORMALIZE_DONE_KEY = "stardust:normalize_done"
BATCH_DONE_KEY = "stardust:batch_done"

_NORMALIZER_MAP = {
    "hotpotqa": normalize_hotpotqa,
    "qasper": normalize_qasper,
    "crag_open": normalize_crag,
}


async def _reader_worker(parquet_path: Path, id_prefix: str, redis: Redis, n_readers: int) -> None:
    i = 0
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=PARQUET_BATCH_SIZE):
        for record in batch.to_pylist():
            if N and i >= N:
                break
            await redis.xadd(READ_STREAM, {"record": json.dumps({"data": record, "id_prefix": id_prefix, "i": i})})
            i += 1
        if N and i >= N:
            break
    log.info("reader worker done: %s (%d records)", id_prefix, i)
    if int(await redis.incr(READ_DONE_KEY)) >= n_readers:
        await redis.set(READ_DONE_KEY + ":all", 1)


CROSS_RECORD_TOKEN_LIMIT = ATOM_TOKEN_LIMIT


async def _normalize_worker(redis: Redis, worker_id: int) -> None:
    consumer = f"normalize-{worker_id}"
    buf_values: list[str] = []
    buf_node: Node | None = None

    async def _flush(doc_id: str) -> None:
        nonlocal buf_node
        if not buf_values or buf_node is None:
            return
        bundled = buf_node.model_copy(update={"value": " ".join(buf_values)})
        await redis.xadd(
            NORMALIZE_STREAM,
            {
                "doc": json.dumps(
                    {
                        "record_id": doc_id,
                        "nodes": [bundled.model_dump()],
                        "atoms": [bundled.transient_id],
                    }
                )
            },
        )
        buf_values.clear()
        buf_node = None

    while True:
        entries = await redis.xreadgroup(READ_GROUP, consumer, {READ_STREAM: ">"}, count=10, block=500)
        if not entries:
            if await redis.exists(READ_DONE_KEY + ":all"):
                break
            continue
        for _, messages in entries:
            for msg_id, data in messages:
                payload = json.loads(data[b"record"])
                record, id_prefix, i = payload["data"], payload["id_prefix"], payload["i"]
                doc_id = f"{id_prefix}_{i}"
                raw_nodes: list[Any] = [parsed async for parsed in _NORMALIZER_MAP[id_prefix](record)]
                await redis.xack(READ_STREAM, READ_GROUP, msg_id)
                if not raw_nodes:
                    continue
                non_atom_nodes: list[Node] = []
                atom_nodes: list[Node] = []
                atom_transient_ids: list[int] = []
                for parsed in raw_nodes:
                    if parsed.is_atom:
                        atom_nodes.append(parsed.node)
                        atom_transient_ids.append(parsed.node.transient_id)
                    else:
                        non_atom_nodes.append(parsed.node)

                if non_atom_nodes or atom_nodes:
                    await redis.xadd(
                        NORMALIZE_STREAM,
                        {
                            "doc": json.dumps(
                                {
                                    "record_id": doc_id,
                                    "nodes": [n.model_dump() for n in non_atom_nodes + atom_nodes],
                                    "atoms": atom_transient_ids,
                                }
                            )
                        },
                    )

                for node in atom_nodes:
                    if buf_values and _token_count(" ".join([*buf_values, node.value])) > CROSS_RECORD_TOKEN_LIMIT:
                        await _flush(doc_id)
                    buf_values.append(node.value)
                    buf_node = node.model_copy(update={"transient_parent_id": None})
                    if _token_count(" ".join(buf_values)) >= CROSS_RECORD_TOKEN_LIMIT:
                        await _flush(doc_id)

    if buf_values:
        await _flush("bundled")

    log.info("normalize worker done")
    await redis.incr(NORMALIZE_DONE_KEY)


async def _batch_worker(redis: Redis, worker_id: int) -> None:
    consumer = f"batch-{worker_id}"
    batch: list[dict] = []
    while True:
        entries = await redis.xreadgroup(NORMALIZE_GROUP, consumer, {NORMALIZE_STREAM: ">"}, count=100, block=500)
        if not entries:
            if int(await redis.get(NORMALIZE_DONE_KEY) or 0) >= NORMALIZE_WORKERS:
                if batch:
                    await redis.xadd(INSERT_STREAM, {"batch": json.dumps(batch)})
                break
            continue
        for _, messages in entries:
            for msg_id, data in messages:
                await redis.xack(NORMALIZE_STREAM, NORMALIZE_GROUP, msg_id)
                batch.append(json.loads(data[b"doc"]))
                if len(batch) >= INSERT_BATCH_SIZE:
                    await redis.xadd(INSERT_STREAM, {"batch": json.dumps(batch)})
                    batch = []
    log.info("batch worker done")
    await redis.incr(BATCH_DONE_KEY)


async def _insert_worker(redis: Redis, worker_id: int) -> None:
    inserted = 0
    consumer = f"worker-{worker_id}"
    while True:
        entries = await redis.xreadgroup(INSERT_GROUP, consumer, {INSERT_STREAM: ">"}, count=1, block=500)
        if not entries:
            if int(await redis.get(BATCH_DONE_KEY) or 0) >= BATCH_WORKERS:
                pending = await redis.xpending(INSERT_STREAM, INSERT_GROUP)
                if pending["pending"] == 0:
                    break
            continue
        for _, messages in entries:
            for msg_id, data in messages:
                batch_data = json.loads(data[b"batch"])
                docs = [(d["record_id"], [Node(**n) for n in d["nodes"]], d["atoms"]) for d in batch_data]
                async with SessionLocal() as session:
                    await insert_index(docs, session)
                await redis.xack(INSERT_STREAM, INSERT_GROUP, msg_id)
                inserted += len(docs)
                log.info("insert worker wrote %d docs, total: %d", len(docs), inserted)
    log.info("insert worker done: %d total", inserted)


async def phase_normalize() -> None:
    log.info("phase 1: normalize + persist")
    redis = Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=False)
    await redis.delete(
        READ_STREAM,
        NORMALIZE_STREAM,
        INSERT_STREAM,
        READ_DONE_KEY,
        READ_DONE_KEY + ":all",
        NORMALIZE_DONE_KEY,
        BATCH_DONE_KEY,
    )
    for stream, group in [
        (READ_STREAM, READ_GROUP),
        (NORMALIZE_STREAM, NORMALIZE_GROUP),
        (INSERT_STREAM, INSERT_GROUP),
    ]:
        with contextlib.suppress(Exception):
            await redis.xgroup_create(stream, group, id="0", mkstream=True)

    n_readers = len(DATASETS)
    for _, parquet_path, id_prefix in DATASETS:
        asyncio.create_task(_reader_worker(parquet_path, id_prefix, redis, n_readers))

    normalize_tasks = [asyncio.create_task(_normalize_worker(redis, i)) for i in range(NORMALIZE_WORKERS)]
    batch_tasks = [asyncio.create_task(_batch_worker(redis, i)) for i in range(BATCH_WORKERS)]
    insert_tasks = [asyncio.create_task(_insert_worker(redis, i)) for i in range(INSERT_WORKERS)]

    await asyncio.gather(*normalize_tasks, *batch_tasks, *insert_tasks)
    await redis.aclose()
    log.info("phase 1: done")


# ── Phase 2: NLP extraction ──────────────────────────────────────────────────


async def phase_nlp() -> None:
    log.info("phase 2: nlp extraction")
    log.info("phase 2: VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    done = 0
    async with SessionLocal() as read_session, read_session.begin():
        stream = await read_session.stream(select(AtomModel.id, AtomModel.value, AtomModel.clean_offset))
        async for partition in stream.partitions(NLP_COMMIT_BATCH_SIZE):
            atom_ids = [r.id for r in partition]
            texts = [r.value for r in partition]
            clean_starts = [r.clean_offset["start"] for r in partition]
            async with SessionLocal() as write_session:
                async for atom_id, attrs in extract_batch(atom_ids, texts, clean_starts):
                    await write_session.execute(
                        update(AtomModel)
                        .where(AtomModel.id == atom_id)
                        .values(nlp_attributes=[a.model_dump() for a in attrs])
                    )
                await write_session.commit()
            done += len(partition)
            log.info("phase 2: %d atoms done", done)


# ── Phase 3: LLM pronoun resolution (Ollama / Qwen3 4B) ─────────────────────


async def _stream_pending_llm_rows() -> AsyncGenerator[tuple[int, str, list]]:
    async with SessionLocal() as session, session.begin():
        stream = await session.stream(
            select(AtomModel.id, AtomModel.value, AtomModel.nlp_attributes).where(AtomModel.disambiguation.is_(None))
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
                    update(AtomModel)
                    .where(AtomModel.id == entry["id"], AtomModel.disambiguation.is_(None))
                    .values(disambiguation=entry["disambiguation"])
                )
            await session.commit()
        log.info("phase 3: loaded %d disambiguation entries from cache", len(cache))

    async def _pending_rows():
        async with SessionLocal() as session, session.begin():
            stream = await session.stream(
                select(AtomModel.id, AtomModel.value, AtomModel.nlp_attributes).where(
                    AtomModel.disambiguation.is_(None)
                )
            )
            async for row in stream:
                yield row.id, row.value, row.nlp_attributes

    i = 0
    async for prompt, batch in build_batch_prompts(_stream_pending_llm_rows()):
        while True:
            try:
                async with SessionLocal() as session:
                    session.add(BatchPromptModel(batch_no=i, prompt=prompt))
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
                            update(AtomModel)
                            .where(AtomModel.id == atom_id)
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


async def _process_record_disambiguation(record_id: str) -> tuple[str, list]:
    async with SessionLocal() as session:
        result = await session.execute(
            select(AtomModel.id, AtomModel.nlp_attributes, AtomModel.disambiguation).where(
                AtomModel.record_id == record_id
            )
        )
        rows = [(r.id, r.nlp_attributes, r.disambiguation) for r in result.fetchall()]
    return record_id, collect_entity_mentions(record_id, rows)


async def phase_disambiguation() -> None:
    log.info("phase 4: disambiguation + embedding")

    async with SessionLocal() as session:
        result = await session.execute(select(AtomModel.record_id).distinct())
        record_ids = [r.record_id for r in result.fetchall()]

    per_record = list(await asyncio.gather(*[_process_record_disambiguation(record_id) for record_id in record_ids]))

    log.info("before merge_across_records: VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    async with SessionLocal() as session:
        result = await session.execute(
            select(AtomModel.id, AtomModel.value, AtomModel.value_hash, AtomModel.disambiguation)
        )
        rows = result.fetchall()

    atom_texts = {r.id: clean_value(r.value) for r in rows}

    global_entities = await merge_across_records(per_record, atom_texts)
    log.info("phase 4: %d canonical entities", len(global_entities))

    # build atom_id -> alias union map for enrichment
    atom_aliases: dict[int, list[str]] = {}
    for entity in global_entities:
        for _, _, atom_id, _ in entity.mentions:
            atom_aliases.setdefault(atom_id, []).extend(entity.aliases)

    def _content(value: str) -> str:
        return clean_value(value)

    coref_map: dict[int, list[str]] = {}
    for r in rows:
        if not r.disambiguation:
            continue
        referents = [e["referent"] for e in r.disambiguation.get("pronoun_map", []) if e.get("referent")]
        if referents:
            coref_map[r.id] = referents

    def _enrich(r) -> str:
        parts = [_content(r.value)]
        if r.id in atom_aliases:
            parts.append(" ".join(atom_aliases[r.id]))
        if r.id in coref_map:
            parts.append(" ".join(coref_map[r.id]))
        return " ".join(parts)

    texts = [_enrich(r) for r in rows]
    total = len(rows)

    existing_hashes = {r.id: r.value_hash for r in rows}
    new_hashes = {r.id: hashlib.sha256(texts[i].encode()).hexdigest() for i, r in enumerate(rows)}
    stale = [i for i, r in enumerate(rows) if new_hashes[r.id] != existing_hashes.get(r.id)]

    if stale:
        stale_texts = [texts[i] for i in stale]
        stale_vecs = load_embedder().encode(
            stale_texts,
            batch_size=EMBEDDING_INTERNAL_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        async with SessionLocal() as session:
            for idx, vec in zip(stale, stale_vecs, strict=False):
                r = rows[idx]
                await session.execute(
                    update(AtomModel)
                    .where(AtomModel.id == r.id)
                    .values(embedding=vec.tolist(), value=texts[idx], value_hash=new_hashes[r.id])
                )
            await session.commit()

    log.info("phase 4: embedded %d/%d atoms (skipped %d)", len(stale), total, total - len(stale))

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
        await ollama_unload()
        gc.collect()
        torch.cuda.empty_cache()
        load_embedder()
        await phase_normalize()
    if not skip_nlp:
        unload_embedder()
        unload_nlp()
        await ollama_unload()
        gc.collect()
        torch.cuda.empty_cache()
        load_nlp()
        await phase_nlp()
    if not skip_llm:
        unload_embedder()
        unload_nlp()
        await ollama_unload()
        gc.collect()
        torch.cuda.empty_cache()
        await phase_llm()
    unload_embedder()
    unload_nlp()
    await ollama_unload()
    gc.collect()
    torch.cuda.empty_cache()
    load_embedder()
    await phase_disambiguation()
    unload_embedder()
    unload_nlp()
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
