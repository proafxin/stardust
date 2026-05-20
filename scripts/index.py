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
from sqlalchemy import select, text

from stardust.config import EMBEDDING_BATCH_SIZE, EMBEDDING_INTERNAL_BATCH_SIZE
from stardust.db import SessionLocal
from stardust.extract import embed_sentences, extract_sentences, resolve_pronouns
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
    data = json.loads(TUNING_PATH.read_text())
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


# ── Phase 2: spaCy + embed + pronoun resolution ──────────────────────────────


async def phase_nlp_embed() -> None:
    log.info("phase 2: nlp + embed + resolve")
    log.info("phase 2: VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    atom_data: list[tuple[int, str, list[tuple[int, int, str]]]] = []
    async with SessionLocal() as session, session.begin():
        stream = await session.stream(
            select(
                Atom.id,
                TreeNode.value.label("ancestry"),
                Sentence.id.label("sent_id"),
                Sentence.sentence_idx,
                Sentence.raw_text,
            )
            .join(TreeNode, TreeNode.id == Atom.parent_id)
            .join(Sentence, Sentence.atom_id == Atom.id)
            .order_by(Atom.id, Sentence.sentence_idx)
        )
        current_atom_id = None
        current_ancestry = ""
        current_sents: list[tuple[int, int, str]] = []
        async for row in stream:
            if row.id != current_atom_id:
                if current_atom_id is not None:
                    atom_data.append((current_atom_id, current_ancestry, current_sents))
                current_atom_id = row.id
                current_ancestry = row.ancestry or ""
                current_sents = []
            current_sents.append((row.sent_id, row.sentence_idx, row.raw_text))
        if current_atom_id is not None:
            atom_data.append((current_atom_id, current_ancestry, current_sents))

    log.info("phase 2: running spaCy on %d atoms", len(atom_data))
    atom_sentences = [[s[2] for s in sents] for _, _, sents in atom_data]
    all_sent_results = extract_sentences(atom_sentences)
    log.info("phase 2: spaCy done, VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    unload_nlp()
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    log.info("phase 2: spaCy unloaded, VRAM free %.2fGB", torch.cuda.mem_get_info()[0] / 1024**3)

    embedder = load_embedder()
    done = embedded = updated = 0

    for batch_start in range(0, len(atom_data), EMBEDDING_BATCH_SIZE):
        batch_atoms = atom_data[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        batch_sent_results = all_sent_results[batch_start : batch_start + EMBEDDING_BATCH_SIZE]

        all_embed_texts: list[str] = []
        for (_, ancestry, sents), _ in zip(batch_atoms, batch_sent_results, strict=False):
            for _, _, raw_text in sents:
                all_embed_texts.append(f"{ancestry} | {raw_text}" if ancestry else raw_text)

        token_budget = _load_token_budget()
        if token_budget is not None:
            tokenizer = embedder.tokenizer
            batches: list[list[str]] = []
            current: list[str] = []
            current_tokens = 0
            for t in all_embed_texts:
                tc = len(tokenizer.encode(t, add_special_tokens=True))
                if current and current_tokens + tc > token_budget:
                    batches.append(current)
                    current, current_tokens = [], 0
                current.append(t)
                current_tokens += tc
            if current:
                batches.append(current)
        else:
            batches = [all_embed_texts[i : i + EMBEDDING_INTERNAL_BATCH_SIZE] for i in range(0, len(all_embed_texts), EMBEDDING_INTERNAL_BATCH_SIZE)]
        vecs_list = [embed_sentences(b, embedder) for b in batches]
        vecs = np.concatenate(vecs_list, axis=0)

        vec_idx = 0
        sent_vecs_map: dict[int, np.ndarray] = {}
        async with SessionLocal() as write_session:
            for (_, ancestry, sents), sent_results in zip(batch_atoms, batch_sent_results, strict=False):
                for (sent_id, _sent_idx, raw_text), _ in zip(sents, sent_results, strict=False):
                    embed_text = all_embed_texts[vec_idx]
                    new_hash = hashlib.sha256(embed_text.encode()).hexdigest()
                    await update_sentence_embedding(sent_id, embed_text, new_hash, vecs[vec_idx], write_session)
                    sent_vecs_map[sent_id] = vecs[vec_idx]
                    vec_idx += 1
            await write_session.commit()
        embedded += vec_idx

        # pronoun resolution within batch
        async with SessionLocal() as write_session:
            for (atom_id, ancestry, sents), sent_results in zip(batch_atoms, batch_sent_results, strict=False):
                has_unresolved = any(r[2] for r in sent_results)
                if not has_unresolved:
                    continue
                sent_vecs = np.array([sent_vecs_map[s[0]] for s in sents], dtype=np.float32)
                sentences_text = [s[2] for s in sents]
                resolved = resolve_pronouns(atom_id, sentences_text, sent_results, sent_vecs, ancestry)
                for (sent_id, _, _), (_, _, resolved_text, changed) in zip(sents, resolved, strict=False):
                    if not changed:
                        continue
                    vec = embed_sentences([resolved_text], embedder)[0]
                    new_hash = hashlib.sha256(resolved_text.encode()).hexdigest()
                    await update_sentence_embedding(sent_id, resolved_text, new_hash, vec, write_session)
                    updated += 1
            await write_session.commit()

        torch.cuda.empty_cache()
        done += sum(len(sents) for _, _, sents in batch_atoms)
        log.info(
            "phase 2: %d sentences embedded, %d resolved, VRAM free %.2fGB",
            done,
            updated,
            torch.cuda.mem_get_info()[0] / 1024**3,
        )

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
