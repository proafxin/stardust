import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc

from stardust.config import EMBEDDING_INTERNAL_BATCH_SIZE, NLP_BATCH_SIZE
from stardust.registry import nlp as load_nlp

log = logging.getLogger(__name__)


def run_nlp(sentences: list[str]) -> list[tuple[bool, list[str]]]:
    nlp_model = load_nlp()
    results = [_analyze_doc(doc) for doc in nlp_model.pipe(sentences, batch_size=NLP_BATCH_SIZE)]
    return results


def _analyze_doc(doc: Doc) -> tuple[bool, list[str]]:
    propn_texts = [t.text for t in doc if t.pos_ == "PROPN"]
    propn_ids = {t.i for t in doc if t.pos_ == "PROPN"}
    has_unresolved = any(
        t.pos_ == "PRON"
        and t.dep_ != "expl"
        and not (propn_ids & ({t.head.i} | {c.i for c in t.children} | {c.i for c in t.head.children}))
        for t in doc
    )
    return has_unresolved, propn_texts


def resolve_atoms(
    atoms: list[tuple[list[str], list[str], list[tuple[bool, list[str]]], list[int]]],
    all_vecs: np.ndarray,
) -> tuple[list[list[str]], int]:
    resolvable = 0
    results: list[list[str]] = []
    offset = 0
    for raw_texts, resolved_texts, nlp_results, embed_indices in atoms:
        n = len(embed_indices)
        sent_vecs = all_vecs[offset : offset + n]
        offset += n
        # map embed position -> original sentence index
        embed_pos = {orig_i: pos for pos, orig_i in enumerate(embed_indices)}
        propn_positions = [pos for pos, orig_i in enumerate(embed_indices) if nlp_results[orig_i][1]]
        final_resolved = list(resolved_texts)
        for orig_i, (has_unresolved, _) in enumerate(nlp_results):
            if not has_unresolved or orig_i not in embed_pos:
                continue
            candidates = [pos for pos in propn_positions if pos != embed_pos[orig_i]]
            if not candidates:
                continue
            i_pos = embed_pos[orig_i]
            scores = np.array([float(sent_vecs[i_pos] @ sent_vecs[pos]) for pos in candidates])
            best_pos = candidates[int(np.argmax(scores))]
            best_orig = embed_indices[best_pos]
            final_resolved[orig_i] = f"{resolved_texts[orig_i]} {' '.join(nlp_results[best_orig][1])}"
            resolvable += 1
        results.append(final_resolved)
    return results, resolvable


def embed_sentences(texts: list[str], embedder: SentenceTransformer) -> np.ndarray:
    return embedder.encode(
        texts,
        batch_size=EMBEDDING_INTERNAL_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def embed_with_token_budget(texts: list[str], embedder: SentenceTransformer, token_budget: int) -> np.ndarray:
    token_counts = [len(ids) for ids in embedder.tokenizer(texts, add_special_tokens=True)["input_ids"]]
    all_vecs: list[np.ndarray] = [None] * len(texts)  # type: ignore[list-item]
    batch_indices: list[int] = []
    batch_tokens = 0
    batches_done = 0
    for i, tc in enumerate(token_counts):
        if batch_tokens + tc > token_budget and batch_indices:
            vecs = embed_sentences([texts[j] for j in batch_indices], embedder)
            for j, vec in zip(batch_indices, vecs, strict=False):
                all_vecs[j] = vec
            batches_done += 1
            log.info("embed: batch %d done (%d sentences, %d tokens)", batches_done, len(batch_indices), batch_tokens)
            batch_indices, batch_tokens = [], 0
        batch_indices.append(i)
        batch_tokens += tc
    if batch_indices:
        vecs = embed_sentences([texts[j] for j in batch_indices], embedder)
        for j, vec in zip(batch_indices, vecs, strict=False):
            all_vecs[j] = vec
        batches_done += 1
        log.info("embed: batch %d done (%d sentences, %d tokens)", batches_done, len(batch_indices), batch_tokens)
    return np.array(all_vecs)
