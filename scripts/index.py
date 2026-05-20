import asyncio
import gc
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import cupy
import torch
from sqlalchemy import select, text

from stardust.db import SessionLocal
from stardust.extract import embed_sentences
from stardust.models import Atom, Sentence, TreeNode
from stardust.parse import normalize_hotpotqa
from stardust.query import insert_index, insert_sentences, update_sentence_embedding
from stardust.registry import embedder as load_embedder
from stardust.registry import nlp as load_nlp
from stardust.registry import unload_embedder, unload_nlp, unload_reranker

TUNING_PATH = Path("tuning.json")


def _load_token_budget() -> int | None:
    if not TUNING_PATH.exists():
        return None
    data = json.loads(TUNING_PATH.read_text(encoding="utf-8"))
    return data.get("embedding_token_budget")


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


# ── Phase 1: Normalize + persist atoms + sentences ───────────────────────────


async def phase_normalize() -> None:
    log.info("phase 1: normalize + persist")
    done = 0

    for dataset, data_path, id_prefix in DATASETS:
        i = 0
        with Path(data_path).open(encoding="utf-8") as f:
            records = json.load(f)
        for record in records:
            if N and i >= N:
                break
            record_id = record.get("_id", f"{id_prefix}_{i}")
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
                    atom_db_ids = await insert_index([(record_id, all_nodes, atom_indices)], session)
                    for atom_node, atom_db_id in zip(atoms, atom_db_ids, strict=False):
                        await insert_sentences(atom_db_id, atom_node.sentences, session)
                    await session.commit()
            i += 1
            done += 1
        log.info("phase 1: %s done (%d records)", dataset, i)

    log.info("phase 1: done (%d total records)", done)


async def _flush_batch(sent_ids: list[int], texts: list[str], embedder: Any) -> int:
    if not texts:
        return 0
    vecs = embed_sentences(texts, embedder)
    async with SessionLocal() as session:
        for sent_id, embed_text, vec in zip(sent_ids, texts, vecs, strict=False):
            new_hash = hashlib.sha256(embed_text.encode()).hexdigest()
            tc = len(embed_text.split())
            await update_sentence_embedding(sent_id, embed_text, new_hash, vec, tc, session)
        await session.commit()
    torch.cuda.empty_cache()
    return len(texts)


# ── Phase 2: spaCy + embed + pronoun resolution ──────────────────────────────


async def phase_nlp_embed() -> None:
    log.info("phase 2: nlp + embed + resolve")
    log.info("phase 2: VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    atom_data: list[tuple[int, str, list[tuple[int, int, str, int]]]] = []
    async with SessionLocal() as session, session.begin():
        stream = await session.stream(
            select(
                Atom.id,
                TreeNode.value.label("ancestry"),
                Sentence.id.label("sent_id"),
                Sentence.sentence_idx,
                Sentence.raw_text,
                Sentence.token_count,
            )
            .join(TreeNode, TreeNode.id == Atom.parent_id)
            .join(Sentence, Sentence.atom_id == Atom.id)
            .order_by(Atom.id, Sentence.sentence_idx)
        )
        current_atom_id = None
        current_ancestry = ""
        current_sents: list[tuple[int, int, str, int]] = []
        async for row in stream:
            if row.id != current_atom_id:
                if current_atom_id is not None:
                    atom_data.append((current_atom_id, current_ancestry, current_sents))
                current_atom_id = row.id
                current_ancestry = row.ancestry or ""
                current_sents = []
            current_sents.append((row.sent_id, row.sentence_idx, row.raw_text, row.token_count))
        if current_atom_id is not None:
            atom_data.append((current_atom_id, current_ancestry, current_sents))

    log.info("phase 2: running spaCy on %d atoms", len(atom_data))
    log.info("phase 2: spaCy done, VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    unload_nlp()
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    log.info("phase 2: spaCy unloaded, VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    embedder = load_embedder()
    token_budget = _load_token_budget()
    if token_budget is None:
        raise RuntimeError("embedding_token_budget required in tuning.json")

    async with SessionLocal() as session:
        result = await session.stream(select(Sentence.id, Sentence.resolved_text, Sentence.token_count))
        all_sentences = [
            (row.id, embed_text, row.token_count)
            async for row in result
            if row.resolved_text and (embed_text := row.resolved_text.strip())
        ]

    embedded = 0
    batch_texts: list[str] = []
    batch_sent_ids: list[int] = []
    current_tokens = 0

    for sent_id, embed_text, token_count in all_sentences:
        if current_tokens + token_count > token_budget and batch_texts:
            embedded += await _flush_batch(batch_sent_ids, batch_texts, embedder)
            batch_texts, batch_sent_ids, current_tokens = [], [], 0
        batch_texts.append(embed_text)
        batch_sent_ids.append(sent_id)
        current_tokens += token_count

    embedded += await _flush_batch(batch_sent_ids, batch_texts, embedder)
    log.info("phase 2: %d sentences embedded", embedded)

    async with SessionLocal() as session:
        await session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_sentences_embedding ON sentences "
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
