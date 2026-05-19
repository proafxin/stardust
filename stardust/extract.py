import asyncio
from collections.abc import AsyncGenerator

from stardust.config import NLP_BATCH_SIZE
from stardust.registry import nlp as load_nlp

_NOMINAL_POS = {"NOUN", "PROPN", "PRON"}


async def extract_batch(
    atom_ids: list[int], texts: list[str], clean_starts: list[int]
) -> AsyncGenerator[tuple[int, list[dict]]]:
    nlp_model = load_nlp()

    def _run() -> list[tuple[int, list[dict]]]:
        results = []
        for atom_id, doc, clean_start in zip(
            atom_ids, nlp_model.pipe(texts, batch_size=NLP_BATCH_SIZE), clean_starts, strict=False
        ):
            tokens = []
            for sent in doc.sents:
                context = sent.text
                for token in sent:
                    if token.pos_ not in _NOMINAL_POS:
                        continue
                    tokens.append({
                        "token_index": token.i,
                        "start": clean_start + token.idx,
                        "end": clean_start + token.idx + len(token.text),
                        "text": token.text,
                        "pos": token.pos_,
                        "dep": token.dep_,
                        "morph": {str(k): str(v) for k, v in token.morph.to_dict().items()},
                        "ent_type": token.ent_type_ or None,
                        "ent_iob": token.ent_iob_ or None,
                        "context": context,
                    })
            results.append((atom_id, tokens))
        return results

    for atom_id, tokens in await asyncio.to_thread(_run):
        yield atom_id, tokens
