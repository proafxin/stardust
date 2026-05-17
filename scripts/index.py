import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from redis.asyncio import Redis
from sqlalchemy import select, text, update

from stardust.config import EMBEDDING_BATCH_SIZE, LLM_BATCH_TOKEN_LIMIT, NLP_BATCH_SIZE, NLP_COMMIT_BATCH_SIZE, settings
from stardust.db import SessionLocal
from stardust.extract import extract_batch
from stardust.llm import ollama_complete, ollama_unload
from stardust.models import AtomModel
from stardust.parse import normalize_crag, normalize_hotpotqa, normalize_qasper
from stardust.query import insert_canonical_entities, insert_index
from stardust.registry import embedder as load_embedder
from stardust.registry import nlp as load_nlp
from stardust.registry import unload_nlp
from stardust.resolution.global_resolution import merge_across_records
from stardust.resolution.local import (
    _pronoun_spans,
    build_batch_prompt,
    build_pronoun_prompt,
    collect_entity_mentions,
)
from stardust.tree.atom import AtomIndex, DisambiguationMetadata, Node, SpanOffset, TokenAttributes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
N = 100  # 0 = no limit

DATASETS: list[tuple[str, Path, str]] = [
    ("hotpotqa", DATA_DIR / "hotpotqa" / "corpus.parquet", "hotpotqa"),
    # ("qasper", DATA_DIR / "qasper" / "train.parquet", "qasper"),  # TODO: fix chunking
    ("crag_open", DATA_DIR / "crag" / "open" / "train.parquet", "crag_open"),
]


# ── Phase 1: Normalize and persist ──────────────────────────────────────────


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
NODE_ID_KEY = "stardust:node_id_counter"
READ_DONE_KEY = "stardust:read_done"
NORMALIZE_DONE_KEY = "stardust:normalize_done"
BATCH_DONE_KEY = "stardust:batch_done"

_NORMALIZER_MAP = {
    "hotpotqa": normalize_hotpotqa,
    "qasper": normalize_qasper,
    "crag_open": normalize_crag,
}


async def _next_ids(redis: Redis, count: int) -> int:
    end = await redis.incrby(NODE_ID_KEY, count)
    return int(end) - count


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


async def _normalize_worker(redis: Redis, worker_id: int) -> None:
    consumer = f"normalize-{worker_id}"
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
                raw_nodes: list[Any] = [parsed async for parsed in _NORMALIZER_MAP[id_prefix](record, 0)]
                await redis.xack(READ_STREAM, READ_GROUP, msg_id)
                if not raw_nodes:
                    continue
                start_id = await _next_ids(redis, len(raw_nodes))
                id_map = {j: start_id + j for j in range(len(raw_nodes))}
                nodes: list[Node] = []
                atoms: list[int] = []
                for j, parsed in enumerate(raw_nodes):
                    node = parsed.node.model_copy(update={
                        "id": id_map[j],
                        "parent_id": id_map[parsed.node.parent_id] if parsed.node.parent_id is not None else None,
                    })
                    nodes.append(node)
                    if parsed.is_atom:
                        atoms.append(node.id)
                await redis.xadd(NORMALIZE_STREAM, {"doc": json.dumps({
                    "record_id": doc_id, "nodes": [n.model_dump() for n in nodes], "atoms": atoms,
                })})
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
    await redis.delete(NODE_ID_KEY, READ_STREAM, NORMALIZE_STREAM, INSERT_STREAM, READ_DONE_KEY, READ_DONE_KEY + ":all", NORMALIZE_DONE_KEY, BATCH_DONE_KEY)
    for stream, group in [(READ_STREAM, READ_GROUP), (NORMALIZE_STREAM, NORMALIZE_GROUP), (INSERT_STREAM, INSERT_GROUP)]:
        try:
            await redis.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception:
            pass

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

    done = 0
    async with SessionLocal() as read_session:
        async with read_session.begin():
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


async def _process_llm_batch(batch: list[tuple[int, str, list]], keep_alive: str = "5m") -> list[dict]:
    atom_data = []
    for atom_id, value, nlp_attrs in batch:
        attrs = [TokenAttributes(**a) for a in (nlp_attrs or [])]
        entry = build_pronoun_prompt(atom_id, value, attrs)
        if entry:
            atom_data.append(entry)
    if not atom_data:
        return []
    prompt = build_batch_prompt(atom_data)
    raw = await ollama_complete(prompt, max_tokens=max(500, len(atom_data) * 50), keep_alive=keep_alive)
    try:
        start, end = raw.find("["), raw.rfind("]") + 1
        return json.loads(raw[start:end]) if start != -1 and end > 0 else []
    except json.JSONDecodeError:
        return []


async def phase_llm() -> None:
    log.info("phase 3: llm pronoun resolution")

    async with SessionLocal() as session:
        result = await session.execute(
            select(AtomModel.id, AtomModel.value, AtomModel.nlp_attributes, AtomModel.disambiguation)
        )
        rows = result.fetchall()

    # build token-budget batches, skip already-disambiguated atoms
    batches: list[list[tuple[int, str, list]]] = []
    current: list[tuple[int, str, list]] = []
    current_tokens = 0
    for row in rows:
        if row.disambiguation is not None:
            continue
        entry = build_pronoun_prompt(row.id, row.value, [TokenAttributes(**a) for a in (row.nlp_attributes or [])])
        tokens = len(json.dumps(entry).split()) if entry else len(row.value.split())
        if current_tokens + tokens > LLM_BATCH_TOKEN_LIMIT and current:
            batches.append(current)
            current, current_tokens = [], 0
        current.append((row.id, row.value, row.nlp_attributes))
        current_tokens += tokens
    if current:
        batches.append(current)

    log.info("phase 3: %d batches", len(batches))

    for i, batch in enumerate(batches):
        keep_alive = "0" if i == len(batches) - 1 else "5m"
        while True:
            try:
                pronoun_map = await _process_llm_batch(batch, keep_alive)
                async with SessionLocal() as session:
                    for atom_id, _value, nlp_attrs in batch:
                        disambiguation = {"pronoun_map": []}
                        attrs = [TokenAttributes(**a) for a in (nlp_attrs or [])]
                        for entry in pronoun_map:
                            if entry.get("atom_id") != atom_id:
                                continue
                            offset_val = entry.get("offset", [])
                            matched = next(
                                (
                                    a
                                    for a in _pronoun_spans(attrs)
                                    if len(offset_val) == 2 and a.offset.start == offset_val[0]
                                ),
                                None,
                            )
                            if matched is None:
                                continue
                            disambiguation["pronoun_map"].append(
                                {
                                    "offset": matched.offset.model_dump(),
                                    "pronoun": entry.get("pronoun", matched.text),
                                    "local_entity": entry.get("local_entity", ""),
                                    "confidence": float(entry.get("confidence", 1.0)),
                                }
                            )
                        await session.execute(
                            update(AtomModel).where(AtomModel.id == atom_id).values(disambiguation=disambiguation)
                        )
                    await session.commit()
                log.info("phase 3: batch %d/%d done", i + 1, len(batches))
                break
            except Exception as e:
                log.warning("phase 3: batch %d failed (%s), retrying in 10s", i + 1, e)
                await asyncio.sleep(10)

    log.info("phase 3: done, ollama unloaded")
    await ollama_unload()


# ── Phase 4: Disambiguation + embedding ─────────────────────────────────────


async def _process_record_disambiguation(record_id: str) -> tuple[str, list]:
    async with SessionLocal() as session:
        result = await session.execute(select(AtomModel).where(AtomModel.record_id == record_id))
        atom_rows = result.scalars().all()

    nodes = {}
    atoms = []
    for row in atom_rows:
        attrs = [TokenAttributes(**a) for a in (row.nlp_attributes or [])]
        disambig = DisambiguationMetadata(**row.disambiguation) if row.disambiguation else None
        nodes[row.id] = Node(
            id=row.id,
            node_type="paragraph",
            modality="text",
            value=row.value,
            raw_offset=SpanOffset(**row.raw_offset),
            clean_offset=SpanOffset(**row.clean_offset),
            parent_id=row.parent_id,
            terminal=True,
            nlp_attributes=attrs,
            disambiguation=disambig,
        )
        atoms.append(row.id)

    index = AtomIndex(nodes=nodes, children={}, atoms=atoms, embeddings={})
    mentions = collect_entity_mentions(record_id, index)
    return record_id, mentions


async def phase_disambiguation() -> None:
    log.info("phase 4: disambiguation + embedding")

    async with SessionLocal() as session:
        result = await session.execute(select(AtomModel.record_id).distinct())
        record_ids = [r.record_id for r in result.fetchall()]

    per_record = list(await asyncio.gather(*[_process_record_disambiguation(record_id) for record_id in record_ids]))

    async with SessionLocal() as session:
        result = await session.execute(select(AtomModel.id, AtomModel.value))
        rows = result.fetchall()
    atom_texts = {r.id: r.value for r in rows}

    global_entities = await merge_across_records(per_record, atom_texts)
    log.info("phase 4: %d canonical entities", len(global_entities))

    # build atom_id -> alias union map for enrichment
    atom_aliases: dict[int, list[str]] = {}
    for entity in global_entities:
        for _, _, atom_id, _ in entity.mentions:
            atom_aliases.setdefault(atom_id, []).extend(entity.aliases)

    atom_ids = [r.id for r in rows]
    texts = [r.value + (" " + " ".join(atom_aliases[r.id]) if r.id in atom_aliases else "") for r in rows]
    total = len(atom_ids)

    for batch_start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch_ids = atom_ids[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        batch_texts = texts[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        vecs = await asyncio.to_thread(
            load_embedder().encode,
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=EMBEDDING_BATCH_SIZE,
        )
        async with SessionLocal() as session:
            for atom_id, vec, text_val in zip(batch_ids, vecs, batch_texts, strict=False):
                await session.execute(
                    update(AtomModel)
                    .where(AtomModel.id == atom_id)
                    .values(embedding=vec.tolist(), value=text_val)
                )
            await session.commit()
        log.info("phase 4: embedded %d/%d", min(batch_start + EMBEDDING_BATCH_SIZE, total), total)

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
        await phase_normalize()
    if not skip_nlp:
        log.info("loading nlp model")
        load_nlp()
        await phase_nlp()
        log.info("unloading nlp model")
        unload_nlp()
    if not skip_llm:
        await phase_llm()
    log.info("loading embedding model")
    load_embedder()
    await phase_disambiguation()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--skip-nlp", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(skip_normalize=args.skip_normalize, skip_nlp=args.skip_nlp, skip_llm=args.skip_llm))
