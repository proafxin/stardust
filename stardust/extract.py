import numpy as np
from sentence_transformers import SentenceTransformer

from stardust.config import EMBEDDING_INTERNAL_BATCH_SIZE, NLP_BATCH_SIZE
from stardust.registry import nlp as load_nlp
from stardust.resolution.global_resolution import canonicalize

_RELEVANT_POS = {"PROPN", "PRON"}


def extract_batch(
    texts: list[str],
) -> list[tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]]:
    nlp_model = load_nlp()
    results = []
    for doc in nlp_model.pipe(texts, batch_size=NLP_BATCH_SIZE):
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


def enrich_batch(
    texts: list[str],
    token_results: list[tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]],
    embedder: SentenceTransformer,
) -> list[str]:
    all_contexts: list[str] = []
    context_index: dict[str, int] = {}
    for propns, prons in token_results:
        for _, _, ctx in propns + prons:
            if ctx not in context_index:
                context_index[ctx] = len(all_contexts)
                all_contexts.append(ctx)

    if not all_contexts:
        return list(texts)

    vecs = embedder.encode(
        all_contexts,
        batch_size=EMBEDDING_INTERNAL_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    enriched_texts = []
    for text, (propns, prons) in zip(texts, token_results, strict=False):
        if not propns:
            enriched_texts.append(text)
            continue

        propn_vecs = np.array([vecs[context_index[c]] for _, _, c in propns])
        propn_dicts = [{"id": i, "text": t, "context": c} for i, (_, t, c) in enumerate(propns)]
        clusters = canonicalize(propn_dicts, propn_vecs)

        idx_to_aliases: dict[int, list[str]] = {}
        for canonical_id, member_ids in clusters.items():
            canonical_text = propn_dicts[canonical_id]["text"]
            aliases = list({propn_dicts[m]["text"] for m in member_ids})
            for m in member_ids:
                idx_to_aliases[m] = [canonical_text] + [a for a in aliases if a != canonical_text]

        pron_aliases: list[str] = []
        if prons:
            pron_vecs = np.array([vecs[context_index[c]] for _, _, c in prons])
            scores = pron_vecs @ propn_vecs.T
            for i in range(len(prons)):
                best_j = int(np.argmax(scores[i]))
                if float(scores[i][best_j]) >= 0.5:
                    pron_aliases.extend(idx_to_aliases.get(best_j, [propns[best_j][1]]))

        all_aliases = list({a for aliases in idx_to_aliases.values() for a in aliases} | set(pron_aliases))
        enriched_texts.append(text + " " + " ".join(all_aliases) if all_aliases else text)

    return enriched_texts
