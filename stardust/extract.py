import asyncio
from collections.abc import AsyncGenerator

import numpy as np
from sentence_transformers import SentenceTransformer

from stardust.config import EMBEDDING_INTERNAL_BATCH_SIZE, NLP_BATCH_SIZE
from stardust.registry import nlp as load_nlp
from stardust.resolution.global_resolution import canonicalize

_RELEVANT_POS = {"PROPN", "PRON"}


def _extract_tokens(texts: list[str], clean_starts: list[int]) -> list[tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]]:
    nlp_model = load_nlp()
    results = []
    for doc, clean_start in zip(nlp_model.pipe(texts, batch_size=NLP_BATCH_SIZE), clean_starts, strict=False):
        propns: list[tuple[int, str, str]] = []
        prons: list[tuple[int, str, str]] = []
        for sent in doc.sents:
            context = sent.text
            for token in sent:
                if token.pos_ == "PROPN":
                    propns.append((token.i, token.text, context))
                elif token.pos_ == "PRON":
                    prons.append((token.i, token.text, context))
        results.append((propns, prons))
    return results


async def extract_batch(
    atom_ids: list[int], texts: list[str], clean_starts: list[int], embedder: SentenceTransformer,
) -> AsyncGenerator[tuple[int, str]]:
    token_results = await asyncio.to_thread(_extract_tokens, texts, clean_starts)

    # collect all unique contexts needing embedding across all atoms
    all_contexts: list[str] = []
    context_index: dict[str, int] = {}

    for propns, prons in token_results:
        for _, _, ctx in propns + prons:
            if ctx not in context_index:
                context_index[ctx] = len(all_contexts)
                all_contexts.append(ctx)

    if not all_contexts:
        for atom_id, text in zip(atom_ids, texts, strict=False):
            yield atom_id, text
        return

    vecs = embedder.encode(
        all_contexts,
        batch_size=EMBEDDING_INTERNAL_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    for atom_id, text, (propns, prons) in zip(atom_ids, texts, token_results, strict=False):
        if not propns:
            yield atom_id, text
            continue

        propn_vecs = np.array([vecs[context_index[c]] for _, _, c in propns])

        # cluster PROPNs by context similarity to get canonical aliases
        propn_dicts = [{"id": i, "text": t, "context": c} for i, (_, t, c) in enumerate(propns)]
        clusters = canonicalize(propn_dicts, propn_vecs)

        idx_to_aliases: dict[int, list[str]] = {}
        for canonical_id, member_ids in clusters.items():
            canonical_text = propn_dicts[canonical_id]["text"]
            aliases = list({propn_dicts[m]["text"] for m in member_ids})
            for m in member_ids:
                idx_to_aliases[m] = [canonical_text] + [a for a in aliases if a != canonical_text]

        # resolve PRONs to nearest PROPN
        pron_aliases: list[str] = []
        if prons:
            pron_vecs = np.array([vecs[context_index[c]] for _, _, c in prons])
            scores = pron_vecs @ propn_vecs.T
            for i in range(len(prons)):
                best_j = int(np.argmax(scores[i]))
                if float(scores[i][best_j]) >= 0.5:
                    pron_aliases.extend(idx_to_aliases.get(best_j, [propns[best_j][1]]))

        all_aliases = list({a for aliases in idx_to_aliases.values() for a in aliases} | set(pron_aliases))
        yield atom_id, (text + " " + " ".join(all_aliases) if all_aliases else text)
