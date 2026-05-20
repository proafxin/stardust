import numpy as np
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc

from stardust.config import EMBEDDING_INTERNAL_BATCH_SIZE, NLP_BATCH_SIZE
from stardust.registry import nlp as load_nlp


def run_nlp(sentences: list[str]) -> list[tuple[bool, list[str]]]:
    nlp_model = load_nlp()
    return [_analyze_doc(doc) for doc in nlp_model.pipe(sentences, batch_size=NLP_BATCH_SIZE)]


def _analyze_doc(doc: Doc) -> tuple[bool, list[str]]:
    propn_texts = [t.text for t in doc if t.pos_ == "PROPN"]
    has_unresolved = any(
        t.pos_ == "PRON" and not any(t2.pos_ == "PROPN" and (t.head in {t2, t2.head}) for t2 in doc) for t in doc
    )
    return has_unresolved, propn_texts


def resolve_atoms(
    atoms: list[tuple[list[str], list[str], list[tuple[bool, list[str]]]]],
    all_vecs: np.ndarray,
) -> tuple[list[list[str]], int]:
    resolvable = 0
    results: list[list[str]] = []
    offset = 0
    for raw_texts, resolved_texts, nlp_results in atoms:
        n = len(raw_texts)
        sent_vecs = all_vecs[offset : offset + n]
        offset += n
        propn_indices = [i for i, (_, propns) in enumerate(nlp_results) if propns]
        final_resolved = list(resolved_texts)
        for i, (has_unresolved, _) in enumerate(nlp_results):
            if not has_unresolved or not propn_indices:
                continue
            scores = np.array([float(sent_vecs[i] @ sent_vecs[j]) for j in propn_indices])
            best_j = propn_indices[int(np.argmax(scores))]
            if best_j == i:
                continue
            final_resolved[i] = f"{resolved_texts[i]} {' '.join(nlp_results[best_j][1])}"
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
    for i, tc in enumerate(token_counts):
        if batch_tokens + tc > token_budget and batch_indices:
            vecs = embed_sentences([texts[j] for j in batch_indices], embedder)
            for j, vec in zip(batch_indices, vecs, strict=False):
                all_vecs[j] = vec
            batch_indices, batch_tokens = [], 0
        batch_indices.append(i)
        batch_tokens += tc
    if batch_indices:
        vecs = embed_sentences([texts[j] for j in batch_indices], embedder)
        for j, vec in zip(batch_indices, vecs, strict=False):
            all_vecs[j] = vec
    return np.array(all_vecs)
