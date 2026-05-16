from stardust.registry import nlp as load_nlp
from stardust.tree.atom import AtomIndex, SpanOffset, TokenAttributes


def extract(index: AtomIndex) -> None:
    nlp_model = load_nlp()
    atom_ids = index.atoms
    texts = [index.nodes[a].value for a in atom_ids]

    for atom_id, doc in zip(atom_ids, nlp_model.pipe(texts), strict=False):
        node = index.nodes[atom_id]
        node.nlp_attributes = [
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
            for token in doc
        ]
