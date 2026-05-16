import asyncio
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq
from sqlalchemy import select, update

from stardust.config import EMBEDDING_BATCH_SIZE, LLM_BATCH_TOKEN_LIMIT, NLP_BATCH_SIZE
from stardust.db import SessionLocal
from stardust.extract import extract_batch
from stardust.llm import async_groq_complete
from stardust.models import AtomModel
from stardust.parse import normalize_crag, normalize_hotpotqa, normalize_qasper
from stardust.query import insert_canonical_entities, insert_index
from stardust.registry import embedder as load_embedder
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
N = 100

DATASETS: list[tuple[str, Path, str]] = [
    ("hotpotqa", DATA_DIR / "hotpotqa" / "corpus.parquet", "hotpotqa"),
    # ("qasper", DATA_DIR / "qasper" / "train.parquet", "qasper"),  # TODO: fix chunking
    ("crag_open", DATA_DIR / "crag" / "open" / "train.parquet", "crag_open"),
]


def _token_count(text: str) -> int:
    return len(text.split())


# ── Phase 1: Normalize and persist ──────────────────────────────────────────


async def phase_normalize() -> None:
    log.info("phase 1: normalize + persist")
    normalizer_map = {
        "hotpotqa": normalize_hotpotqa,
        "qasper": normalize_qasper,
        "crag_open": normalize_crag,
    }
    global_counter = 0
    for dataset_name, parquet_path, id_prefix in DATASETS:
        table = pq.read_table(parquet_path)
        records = table.to_pylist()
        if N:
            records = records[:N]
        normalizer = normalizer_map[dataset_name]
        total = len(records)
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
            async with SessionLocal() as session:
                await insert_index(nodes, atoms, doc_id, session)
            if i % 10 == 0:
                log.info("phase 1 [%s]: %d/%d records", dataset_name, i + 1, total)


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
    raw = await async_groq_complete(prompt)
    try:
        start, end = raw.find("["), raw.rfind("]") + 1
        return json.loads(raw[start:end]) if start != -1 and end > 0 else []
    except json.JSONDecodeError:
        return []


async def phase_llm() -> None:
    log.info("phase 3: llm pronoun resolution")

    async with SessionLocal() as session:
        result = await session.execute(select(AtomModel.id, AtomModel.value, AtomModel.nlp_attributes))
        rows = result.fetchall()

    # build token-budget batches
    batches: list[list[tuple[int, str, list]]] = []
    current: list[tuple[int, str, list]] = []
    current_tokens = 0
    for row in rows:
        tokens = _token_count(row.value)
        if current_tokens + tokens > LLM_BATCH_TOKEN_LIMIT and current:
            batches.append(current)
            current, current_tokens = [], 0
        current.append((row.id, row.value, row.nlp_attributes))
        current_tokens += tokens
    if current:
        batches.append(current)

    log.info("phase 3: %d batches", len(batches))

    # process batches concurrently
    results = await asyncio.gather(*[_process_llm_batch(b) for b in batches])

    async with SessionLocal() as session:
        for batch, pronoun_map in zip(batches, results, strict=False):
            if not pronoun_map:
                continue
            atom_lookup = {r[0]: r for r in batch}
            for entry in pronoun_map:
                atom_id = entry.get("atom_id")
                if atom_id not in atom_lookup:
                    continue
                _, _, nlp_attrs = atom_lookup[atom_id]
                attrs = [TokenAttributes(**a) for a in (nlp_attrs or [])]
                offset_val = entry.get("offset", [])
                matched = next(
                    (a for a in _pronoun_spans(attrs) if len(offset_val) == 2 and a.offset.start == offset_val[0]),
                    None,
                )
                if matched is None:
                    continue
                resolution = {
                    "offset": matched.offset.model_dump(),
                    "pronoun": entry.get("pronoun", matched.text),
                    "local_entity": entry.get("local_entity", ""),
                    "confidence": float(entry.get("confidence", 1.0)),
                }
                result = await session.execute(select(AtomModel.disambiguation).where(AtomModel.id == atom_id))
                existing = result.scalar_one_or_none() or {"pronoun_map": []}
                existing["pronoun_map"].append(resolution)
                await session.execute(update(AtomModel).where(AtomModel.id == atom_id).values(disambiguation=existing))
        await session.commit()

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

    log.info("phase 4: done")


# ── Main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    await phase_normalize()
    await phase_nlp()
    await phase_llm()
    await phase_disambiguation()


if __name__ == "__main__":
    asyncio.run(main())
