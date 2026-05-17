import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import tiktoken
from redis.asyncio import Redis
from sqlalchemy import select, text, update

from stardust.config import EMBEDDING_BATCH_SIZE, LLM_BATCH_TOKEN_LIMIT, NLP_BATCH_SIZE, settings
from stardust.db import SessionLocal
from stardust.extract import extract_batch
from stardust.llm import async_groq_complete
from stardust.models import AtomModel
from stardust.parse import normalize_crag, normalize_hotpotqa, normalize_qasper
from stardust.query import insert_canonical_entities, insert_index
from stardust.registry import embedder as load_embedder
from stardust.registry import nlp as load_nlp
from stardust.registry import unload_nlp
from stardust.resolution.global_resolution import merge_across_documents
from stardust.resolution.local import (
    _pronoun_spans,
    build_batch_prompt,
    build_pronoun_prompt,
    resolve_local,
)
from stardust.tree.atom import AtomIndex, DisambiguationMetadata, Node, SpanOffset, TokenAttributes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
N = 0  # 0 = no limit

DATASETS: list[tuple[str, Path, str]] = [
    ("hotpotqa", DATA_DIR / "hotpotqa" / "corpus.parquet", "hotpotqa"),
    # ("qasper", DATA_DIR / "qasper" / "train.parquet", "qasper"),  # TODO: fix chunking
    ("crag_open", DATA_DIR / "crag" / "open" / "train.parquet", "crag_open"),
]


_enc = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


# ── Phase 1: Normalize and persist ──────────────────────────────────────────


NORMALIZE_BATCH_SIZE = 50_000


NORMALIZE_WORKERS = 4
INSERT_WORKERS = 2


NODE_ID_KEY = "stardust:node_id_counter"


async def _next_ids(redis: Redis, count: int) -> int:
    """Atomically reserve `count` IDs, returns the first ID in the range."""
    end = await redis.incrby(NODE_ID_KEY, count)
    return int(end) - count


async def _normalize_worker(
    records: list,
    id_prefix: str,
    batch_offset: int,
    normalizer: Any,
    queue: asyncio.Queue,
    redis: Redis,
) -> None:
    docs: list[tuple[str, list[Node], list[int]]] = []
    for i, record in enumerate(records):
        doc_id = f"{id_prefix}_{batch_offset + i}"
        # count nodes first to reserve a contiguous ID block atomically
        raw_nodes: list[Any] = []
        async for parsed in normalizer(record, 0):
            raw_nodes.append(parsed)
        if not raw_nodes:
            continue
        start_id = await _next_ids(redis, len(raw_nodes))
        # remap local 0-based IDs to globally unique IDs
        id_map: dict[int, int] = {j: start_id + j for j in range(len(raw_nodes))}
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
        docs.append((doc_id, nodes, atoms))
        if len(docs) >= NORMALIZE_BATCH_SIZE:
            await queue.put(docs)
            docs = []
    if docs:
        await queue.put(docs)


async def _insert_worker(queue: asyncio.Queue, sentinel: object) -> None:
    while True:
        docs = await queue.get()
        if docs is sentinel:
            queue.task_done()
            break
        async with SessionLocal() as session:
            await insert_index(docs, session)
        queue.task_done()


async def phase_normalize() -> None:
    log.info("phase 1: normalize + persist")
    async with SessionLocal() as session:
        await session.execute(text("DROP INDEX IF EXISTS ix_atoms_embedding"))
        await session.commit()
    log.info("phase 1: dropped hnsw index")
    normalizer_map = {
        "hotpotqa": normalize_hotpotqa,
        "qasper": normalize_qasper,
        "crag_open": normalize_crag,
    }
    sentinel = object()
    queue: asyncio.Queue = asyncio.Queue(maxsize=INSERT_WORKERS * 2)
    inserters = [
        asyncio.create_task(_insert_worker(queue, sentinel))
        for _ in range(INSERT_WORKERS)
    ]
    redis = Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=False)
    await redis.delete(NODE_ID_KEY)
    for dataset_name, parquet_path, id_prefix in DATASETS:
        records = pq.read_table(parquet_path).to_pylist()
        if N:
            records = records[:N]
        normalizer = normalizer_map[dataset_name]
        total = len(records)
        chunk = max(1, total // NORMALIZE_WORKERS)
        normalizers = [
            asyncio.create_task(_normalize_worker(
                records[i : i + chunk], id_prefix, i, normalizer, queue, redis
            ))
            for i in range(0, total, chunk)
        ]
        await asyncio.gather(*normalizers)
        log.info("phase 1 [%s]: %d records normalized", dataset_name, total)
    for _ in range(INSERT_WORKERS):
        await queue.put(sentinel)
    await asyncio.gather(*inserters)
    await redis.aclose()
    log.info("phase 1: done")


# ── Phase 2: NLP extraction ──────────────────────────────────────────────────


async def phase_nlp() -> None:
    log.info("phase 2: nlp extraction")

    async with SessionLocal() as session:
        result = await session.execute(select(AtomModel.id, AtomModel.value, AtomModel.clean_offset))
        rows = result.fetchall()

    total = len(rows)
    for batch_start in range(0, total, NLP_BATCH_SIZE):
        batch = rows[batch_start : batch_start + NLP_BATCH_SIZE]
        atom_ids = [r.id for r in batch]
        texts = [r.value for r in batch]
        clean_starts = [r.clean_offset["start"] for r in batch]

        async with SessionLocal() as session:
            async for atom_id, attrs in extract_batch(atom_ids, texts, clean_starts):
                await session.execute(
                    update(AtomModel)
                    .where(AtomModel.id == atom_id)
                    .values(nlp_attributes=[a.model_dump() for a in attrs])
                )
            await session.commit()

        log.info("phase 2: %d/%d atoms", min(batch_start + NLP_BATCH_SIZE, total), total)


# ── Phase 3: LLM pronoun resolution ─────────────────────────────────────────


async def _process_llm_batch(batch: list[tuple[int, str, list]]) -> list[dict]:
    atom_data = []
    for atom_id, value, nlp_attrs in batch:
        attrs = [TokenAttributes(**a) for a in (nlp_attrs or [])]
        entry = build_pronoun_prompt(atom_id, value, attrs)
        if entry:
            atom_data.append(entry)
    if not atom_data:
        return []
    prompt = build_batch_prompt(atom_data)
    raw = await async_groq_complete(prompt, max_tokens=max(500, len(atom_data) * 50))
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

    # build token-budget batches, skipping already-disambiguated atoms
    batches: list[list[tuple[int, str, list]]] = []
    current: list[tuple[int, str, list]] = []
    current_tokens = 0
    for row in rows:
        if row.disambiguation is not None:
            continue
        entry = build_pronoun_prompt(row.id, row.value, [TokenAttributes(**a) for a in (row.nlp_attributes or [])])
        tokens = _token_count(json.dumps(entry)) if entry else _token_count(row.value)
        if current_tokens + tokens > LLM_BATCH_TOKEN_LIMIT and current:
            batches.append(current)
            current, current_tokens = [], 0
        current.append((row.id, row.value, row.nlp_attributes))
        current_tokens += tokens
    if current:
        batches.append(current)

    log.info("phase 3: %d batches", len(batches))

    sem = asyncio.Semaphore(1)

    async def _run_batch(i: int, batch: list) -> None:
        async with sem:
            while True:
                try:
                    pronoun_map = await _process_llm_batch(batch)
                    async with SessionLocal() as session:
                        for atom_id, _value, nlp_attrs in batch:
                            disambiguation = {"pronoun_map": []}
                            for entry in pronoun_map:
                                if entry.get("atom_id") != atom_id:
                                    continue
                                attrs = [TokenAttributes(**a) for a in (nlp_attrs or [])]
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
                    return
                except Exception as e:
                    log.warning("phase 3: batch %d failed (%s), retrying in 10s", i + 1, e)
                    await asyncio.sleep(10)

    await asyncio.gather(*[_run_batch(i, batch) for i, batch in enumerate(batches)])

    log.info("phase 3: done")


# ── Phase 4: Disambiguation + embedding ─────────────────────────────────────


async def _process_doc_disambiguation(doc_id: str) -> tuple[str, dict]:
    async with SessionLocal() as session:
        result = await session.execute(select(AtomModel).where(AtomModel.doc_id == doc_id))
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
    clusters = await resolve_local(index)
    return doc_id, clusters


async def phase_disambiguation() -> None:
    log.info("phase 4: disambiguation + embedding")

    async with SessionLocal() as session:
        result = await session.execute(select(AtomModel.doc_id).distinct())
        doc_ids = [r.doc_id for r in result.fetchall()]

    # process all docs concurrently
    per_doc = list(await asyncio.gather(*[_process_doc_disambiguation(doc_id) for doc_id in doc_ids]))
    global_entities = await merge_across_documents(per_doc)
    log.info("phase 4: %d canonical entities", len(global_entities))

    # stream embedding in batches
    async with SessionLocal() as session:
        result = await session.execute(select(AtomModel.id, AtomModel.value))
        rows = result.fetchall()

    atom_ids = [r.id for r in rows]
    texts = [r.value for r in rows]
    total = len(atom_ids)

    for batch_start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch_ids = atom_ids[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        batch_texts = texts[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        vecs = await asyncio.to_thread(
            load_embedder().encode,
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        async with SessionLocal() as session:
            for atom_id, vec in zip(batch_ids, vecs, strict=False):
                await session.execute(update(AtomModel).where(AtomModel.id == atom_id).values(embedding=vec.tolist()))
            await session.commit()
        log.info("phase 4: embedded %d/%d", min(batch_start + EMBEDDING_BATCH_SIZE, total), total)

    async with SessionLocal() as session:
        await insert_canonical_entities(global_entities, "global", session)

    async with SessionLocal() as session:
        await session.execute(text(
            "CREATE INDEX ix_atoms_embedding ON atoms USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        ))
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
