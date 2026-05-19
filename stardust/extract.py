import asyncio
from collections.abc import AsyncGenerator

import numpy as np
from sentence_transformers import SentenceTransformer

from stardust.config import CANONICALIZATION_THRESHOLD, EMBEDDING_INTERNAL_BATCH_SIZE, NLP_BATCH_SIZE
from stardust.registry import nlp as load_nlp
from stardust.resolution.global_resolution import canonicalize

_RELEVANT_POS = {"PROPN", "PRON"}


def _enrich(value: str, doc, embedder: SentenceTransformer) -> str:
    propns: list[tuple[int, str, str]] = []  # (token.i, text, sent_context)
    prons: list[tuple[int, str, str]] = []

    for sent in doc.sents:
        context = sent.text
        for token in sent:
            if token.pos_ == "PROPN":
                propns.append((token.i, token.text, context))
            elif token.pos_ == "PRON":
                prons.append((token.i, token.text, context))

    if not propns:
        return value

    # global canonicalization of PROPNs via embedding clustering
    propn_dicts = [{"id": i, "text": t, "context": c} for i, (_, t, c) in enumerate(propns)]
    clusters = canonicalize(propn_dicts, embedder)
    # build alias map: each propn index -> canonical surface form + all aliases
    idx_to_aliases: dict[int, list[str]] = {}
    for canonical_id, member_ids in clusters.items():
        canonical_text = propn_dicts[canonical_id]["text"]
        aliases = list({propn_dicts[m]["text"] for m in member_ids})
        for m in member_ids:
            idx_to_aliases[m] = [canonical_text] + [a for a in aliases if a != canonical_text]

    # resolve PRONs to nearest PROPN by sentence embedding similarity
    pron_aliases: list[str] = []
    if prons:
        pron_contexts = [c for _, _, c in prons]
        propn_contexts = [c for _, _, c in propns]
        all_contexts = pron_contexts + propn_contexts
        vecs = embedder.encode(
            all_contexts, batch_size=EMBEDDING_INTERNAL_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False
        )
        pron_vecs = vecs[:len(prons)].astype(np.float32)
        propn_vecs = vecs[len(prons):].astype(np.float32)
        scores = pron_vecs @ propn_vecs.T
        for i in range(len(prons)):
            best_j = int(np.argmax(scores[i]))
            if float(scores[i][best_j]) >= 0.5:
                pron_aliases.extend(idx_to_aliases.get(best_j, [propns[best_j][1]]))

    all_aliases = list({a for aliases in idx_to_aliases.values() for a in aliases} | set(pron_aliases))
    if not all_aliases:
        return value
    return value + " " + " ".join(all_aliases)


async def extract_batch(
    atom_ids: list[int], texts: list[str], clean_starts: list[int], embedder: SentenceTransformer,
) -> AsyncGenerator[tuple[int, str]]:
    nlp_model = load_nlp()

    def _run() -> list[tuple[int, str]]:
        return [
            (atom_id, _enrich(text, doc, embedder))
            for atom_id, text, doc in zip(
                atom_ids,
                texts,
                nlp_model.pipe(texts, batch_size=NLP_BATCH_SIZE),
                strict=False,
            )
        ]

    for atom_id, enriched in await asyncio.to_thread(_run):
        yield atom_id, enriched
