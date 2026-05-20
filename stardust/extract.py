import numpy as np
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc

from stardust.config import EMBEDDING_INTERNAL_BATCH_SIZE, NLP_BATCH_SIZE
from stardust.registry import nlp as load_nlp


def run_nlp(sentences: list[str]) -> list[tuple[bool, list[str]]]:
    nlp_model = load_nlp()
    results: list[tuple[bool, list[str]]] = [
        _analyze_doc(doc) for doc in nlp_model.pipe(sentences, batch_size=NLP_BATCH_SIZE)
    ]
    return results


def _analyze_doc(doc: Doc) -> tuple[bool, list[str]]:
    propn_texts = [t.text for t in doc if t.pos_ == "PROPN"]
    has_unresolved = any(
        t.pos_ == "PRON" and not any(t2.pos_ == "PROPN" and (t.head in {t2, t2.head}) for t2 in doc) for t in doc
    )
    return has_unresolved, propn_texts


def resolve_atom(
    raw_texts: list[str],
    resolved_texts: list[str],
    nlp_results: list[tuple[bool, list[str]]],
    sent_vecs: np.ndarray,
) -> list[str]:
    propn_indices = [i for i, (_, propns) in enumerate(nlp_results) if propns]
    final_resolved = list(resolved_texts)
    for i, (has_unresolved, _) in enumerate(nlp_results):
        if not has_unresolved or not propn_indices:
            continue
        scores = np.array([float(sent_vecs[i] @ sent_vecs[j]) for j in propn_indices])
        best_j = propn_indices[int(np.argmax(scores))]
        if best_j == i:
            continue
        referent_propns = nlp_results[best_j][1]
        final_resolved[i] = f"{resolved_texts[i]} {' '.join(referent_propns)}"
    return final_resolved


def embed_sentences(texts: list[str], embedder: SentenceTransformer) -> np.ndarray:
    return embedder.encode(
        texts,
        batch_size=EMBEDDING_INTERNAL_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
