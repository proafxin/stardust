import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc

from stardust.config import NLP_BATCH_SIZE
from stardust.registry import nlp as load_nlp

log = logging.getLogger(__name__)


# per sentence: (unresolved_pron_morphs, propns)
# unresolved_pron_morphs: list of frozenset of morph features for each unresolved PRON
# propns: list of (text, morph_frozenset)
_SentNLP = tuple[list[frozenset[str]], list[tuple[str, frozenset[str]]]]


def run_nlp(sentences: list[str]) -> list[_SentNLP]:
    nlp_model = load_nlp()
    return [_analyze_doc(doc) for doc in nlp_model.pipe(sentences, batch_size=NLP_BATCH_SIZE)]


def _morph(token) -> frozenset[str]:  # type: ignore[no-untyped-def]
    return frozenset(token.morph.to_dict().items())


def _morph_compatible(pron_morph: frozenset[str], propn_morph: frozenset[str]) -> bool:
    pron_gender = {v for k, v in pron_morph if k == "Gender"}
    pron_number = {v for k, v in pron_morph if k == "Number"}
    propn_gender = {v for k, v in propn_morph if k == "Gender"}
    propn_number = {v for k, v in propn_morph if k == "Number"}
    if pron_gender and propn_gender and not pron_gender & propn_gender:
        return False
    if pron_number and propn_number and not pron_number & propn_number:
        return False
    return True


def _analyze_doc(doc: Doc) -> _SentNLP:
    propn_ids = {t.i for t in doc if t.pos_ == "PROPN"}
    unresolved_morphs: list[frozenset[str]] = [
        _morph(t) for t in doc
        if t.pos_ == "PRON"
        and t.dep_ != "expl"
        and not (propn_ids & ({t.head.i} | {c.i for c in t.children} | {c.i for c in t.head.children}))
    ]
    propns: list[tuple[str, frozenset[str]]] = [(t.text, _morph(t)) for t in doc if t.pos_ == "PROPN"]
    return unresolved_morphs, propns


def resolve_atoms(
    atoms: list[tuple[list[str], list[str], list[_SentNLP], list[int]]],
    all_vecs: np.ndarray,
) -> tuple[list[list[str]], int]:
    resolvable = 0
    results: list[list[str]] = []
    offset = 0
    for raw_texts, resolved_texts, nlp_results, embed_indices in atoms:
        n = len(embed_indices)
        sent_vecs = all_vecs[offset : offset + n]
        offset += n
        embed_pos = {orig_i: pos for pos, orig_i in enumerate(embed_indices)}
        final_resolved = list(resolved_texts)
        for orig_i, (unresolved_morphs, _) in enumerate(nlp_results):
            if not unresolved_morphs or orig_i not in embed_pos:
                continue
            i_pos = embed_pos[orig_i]
            # for each unresolved pron, find best morph-compatible candidate sentence
            appended: set[str] = set()
            for pron_morph in unresolved_morphs:
                candidates = [
                    pos for pos, orig_j in enumerate(embed_indices)
                    if pos != i_pos and any(_morph_compatible(pron_morph, pm) for _, pm in nlp_results[orig_j][1])
                ]
                if not candidates:
                    # fallback: any sentence with PROPNs
                    candidates = [pos for pos, orig_j in enumerate(embed_indices) if pos != i_pos and nlp_results[orig_j][1]]
                if not candidates:
                    continue
                scores = sent_vecs[i_pos] @ sent_vecs[np.array(candidates)].T
                best_pos = candidates[int(np.argmax(scores))]
                best_orig = embed_indices[best_pos]
                compatible = [
                    text for text, pm in nlp_results[best_orig][1]
                    if _morph_compatible(pron_morph, pm)
                ] or [text for text, _ in nlp_results[best_orig][1]]
                appended.update(compatible)
            if appended:
                final_resolved[orig_i] = f"{resolved_texts[orig_i]} {' '.join(appended)}"
                resolvable += 1
        results.append(final_resolved)
    return results, resolvable


def embed_sentences(texts: list[str], embedder: SentenceTransformer) -> np.ndarray:
    return embedder.encode(
        texts,
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
