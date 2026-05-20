import logging
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from spacy.language import Language

log = logging.getLogger(__name__)

_PRON = re.compile(r"\b(he|she|it|they|him|her|them|his|hers|its|their|theirs|himself|herself|itself|themselves)\b", re.IGNORECASE)


def resolve_atom_coref(sentences: list[tuple[str, str]], nlp: Language) -> list[tuple[str, str]]:
    atom_text = " ".join(raw for raw, _ in sentences)
    doc = nlp(atom_text)
    clusters = doc._.coref_clusters
    if not clusters:
        return sentences

    # build map: char_start -> canonical text (first mention in cluster)
    pron_to_canonical: dict[tuple[int, int], str] = {}
    for cluster in clusters:
        canonical = atom_text[cluster[0][0]:cluster[0][1]]
        for start, end in cluster[1:]:
            span_text = atom_text[start:end]
            if _PRON.fullmatch(span_text.strip()):
                pron_to_canonical[(start, end)] = canonical

    if not pron_to_canonical:
        return sentences

    # map each sentence to its char offset in atom_text
    updated: list[tuple[str, str]] = []
    offset = 0
    for raw, resolved in sentences:
        sent_start = offset
        sent_end = offset + len(raw)
        offset = sent_end + 1  # +1 for the space separator

        # find pronouns in this sentence that have a canonical resolution
        replacements = [
            (start - sent_start, end - sent_start, canonical)
            for (start, end), canonical in pron_to_canonical.items()
            if sent_start <= start < sent_end
        ]

        if not replacements:
            updated.append((raw, resolved))
            continue

        # apply replacements right-to-left to preserve offsets
        new_resolved = resolved
        # resolved_text has ancestry prefix + raw, find where raw starts in resolved
        raw_in_resolved = resolved.rfind(raw)
        if raw_in_resolved == -1:
            updated.append((raw, resolved))
            continue

        for r_start, r_end, canonical in sorted(replacements, reverse=True):
            abs_start = raw_in_resolved + r_start
            abs_end = raw_in_resolved + r_end
            new_resolved = new_resolved[:abs_start] + canonical + new_resolved[abs_end:]

        updated.append((raw, new_resolved))

    return updated


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
