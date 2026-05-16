import asyncio
from collections.abc import AsyncGenerator

from stardust.registry import nlp as load_nlp
from stardust.tree.atom import SpanOffset, TokenAttributes


async def extract_batch(atom_ids: list[int], texts: list[str], clean_starts: list[int]) -> AsyncGenerator[tuple[int, list[TokenAttributes]], None]:
    nlp_model = load_nlp()

    def _run() -> list[tuple[int, list[TokenAttributes]]]:
        results = []
        for atom_id, doc, clean_start in zip(atom_ids, nlp_model.pipe(texts), clean_starts, strict=False):
            attrs = [
                TokenAttributes(
                    text=token.text,
                    pos_=token.pos_,
                    dep_=token.dep_,
                    morph={str(k): str(v) for k, v in token.morph.to_dict().items()},
                    ent_type_=token.ent_type_,
                    ent_iob_=token.ent_iob_,
                    offset=SpanOffset(
                        start=clean_start + token.idx,
                        end=clean_start + token.idx + len(token.text),
                    ),
                )
                for token in doc
            ]
            results.append((atom_id, attrs))
        return results

    results = await asyncio.to_thread(_run)
    for atom_id, attrs in results:
        yield atom_id, attrs
