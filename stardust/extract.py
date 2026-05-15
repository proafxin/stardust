from __future__ import annotations

from stardust.registry import get_nlp
from stardust.tree.atom import AtomIndex, SpanOffset, TokenAttributes


def extract(index: AtomIndex) -> None:
    nlp = get_nlp()
    for atom_id in index.atoms:
        node = index.nodes[atom_id]
        doc = nlp(node.value)
        attrs: list[TokenAttributes] = []
        for token in doc:
            attrs.append(
                TokenAttributes(
                    text=token.text,
                    pos_=token.pos_,
                    dep_=token.dep_,
                    morph={str(k): str(v) for k, v in token.morph.to_dict().items()},
                    ent_type_=token.ent_type_,
                    ent_iob_=token.ent_iob_,
                    offset=SpanOffset(
                        start=node.clean_offset.start + token.idx,
                        end=node.clean_offset.start + token.idx + len(token.text),
                    ),
                )
            )
        node.nlp_attributes = attrs
