import logging
import re

import numpy as np
from fastcoref import LingMessCoref
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

_PRON = re.compile(r"\b(he|she|it|they|him|her|them|his|hers|its|their|theirs|himself|herself|itself|themselves)\b", re.IGNORECASE)


def resolve_atoms_coref(
    atoms: list[tuple[list[str], str]],
    coref_model: LingMessCoref,
    max_tokens_in_batch: int = 10000,
) -> list[list[tuple[str, str]]]:
    atom_texts = [" ".join(sents) for sents, _ in atoms]
    preds = coref_model.predict(texts=atom_texts, max_tokens_in_batch=max_tokens_in_batch)

    results: list[list[tuple[str, str]]] = []
    for atom_text, (raw_sents, ancestry), pred in zip(atom_texts, atoms, preds, strict=False):
        clusters = pred.get_clusters(as_strings=False)

        pron_to_canonical: dict[tuple[int, int], str] = {}
        if clusters:
            for cluster in clusters:
                canonical = atom_text[cluster[0][0]:cluster[0][1]]
                for start, end in cluster[1:]:
                    if _PRON.fullmatch(atom_text[start:end].strip()):
                        pron_to_canonical[(start, end)] = canonical

        resolved_sents: list[tuple[str, str]] = []
        offset = 0
        for raw in raw_sents:
            sent_start = offset
            sent_end = offset + len(raw)
            offset = sent_end + 1

            replacements = [
                (start - sent_start, end - sent_start, canonical)
                for (start, end), canonical in pron_to_canonical.items()
                if sent_start <= start < sent_end
            ]

            resolved = raw
            for r_start, r_end, canonical in sorted(replacements, reverse=True):
                resolved = resolved[:r_start] + canonical + resolved[r_end:]

            resolved_sents.append((raw, f"{ancestry} | {resolved}"))
        results.append(resolved_sents)
    return results


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
