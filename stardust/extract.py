import numpy as np
from sentence_transformers import SentenceTransformer

from stardust.config import EMBEDDING_INTERNAL_BATCH_SIZE, NLP_BATCH_SIZE
from stardust.registry import nlp as load_nlp


def extract_sentences(
    atom_sentences: list[list[str]],
) -> list[list[tuple[int, bool, bool, list[str]]]]:
    nlp_model = load_nlp()
    flat_texts = [s for sents in atom_sentences for s in sents]
    flat_results = []
    for doc in nlp_model.pipe(flat_texts, batch_size=NLP_BATCH_SIZE):
        has_propn = any(t.pos_ == "PROPN" for t in doc)
        propn_texts = [t.text for t in doc if t.pos_ == "PROPN"]
        has_unresolved_pron = any(
            t.pos_ == "PRON" and not any(
                t2.pos_ == "PROPN" and t2.sent == t.sent for t2 in doc
            )
            for t in doc
        )
        flat_results.append((has_propn, has_unresolved_pron, propn_texts))

    results = []
    idx = 0
    for sents in atom_sentences:
        atom_result = []
        for sent_idx, _ in enumerate(sents):
            has_propn, has_unresolved_pron, propn_texts = flat_results[idx]
            atom_result.append((sent_idx, has_propn, has_unresolved_pron, propn_texts))
            idx += 1
        results.append(atom_result)
    return results


def resolve_pronouns(
    atom_id: int,
    sentences: list[str],
    sent_results: list[tuple[int, bool, bool, list[str]]],
    sent_vecs: np.ndarray,
    ancestry: str,
) -> list[tuple[int, str, str, bool]]:
    propn_indices = [i for i, (_, has_propn, _, _) in enumerate(sent_results) if has_propn]
    unresolved_indices = [i for i, (_, _, has_unresolved, _) in enumerate(sent_results) if has_unresolved]

    resolved: list[tuple[int, str, str, bool]] = []
    for i, (sent_idx, has_propn, has_unresolved, propn_texts) in enumerate(sent_results):
        raw = sentences[sent_idx]
        embed_text = f"{ancestry} | {raw}" if ancestry else raw
        if not has_unresolved or not propn_indices:
            resolved.append((sent_idx, raw, embed_text, False))
            continue
        pron_vec = sent_vecs[i]
        scores = np.array([float(pron_vec @ sent_vecs[j]) for j in propn_indices])
        best_j = propn_indices[int(np.argmax(scores))]
        if best_j == i:
            resolved.append((sent_idx, raw, embed_text, False))
            continue
        referent_propns = sent_results[best_j][3]
        resolved_text = f"{embed_text} {' '.join(referent_propns)}"
        resolved.append((sent_idx, raw, resolved_text, True))
    return resolved


def embed_sentences(texts: list[str], embedder: SentenceTransformer) -> np.ndarray:
    return embedder.encode(
        texts,
        batch_size=len(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
