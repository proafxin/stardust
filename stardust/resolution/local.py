from collections import defaultdict

import numpy as np

from stardust.config import ENTITY_MERGE_THRESHOLD
from stardust.registry import embedder as load_embedder
from stardust.tree.atom import AtomIndex, DisambiguationMetadata, PronounResolution, SpanOffset, TokenAttributes

_EQUIVALENT_TYPES: dict[str, str] = {
    "ORG": "ORG",
    "COMPANY": "ORG",
    "CORP": "ORG",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "PERSON": "PERSON",
    "PER": "PERSON",
    "NORP": "NORP",
    "FAC": "FAC",
    "PRODUCT": "PRODUCT",
    "EVENT": "EVENT",
    "WORK_OF_ART": "WORK_OF_ART",
    "LAW": "LAW",
    "LANGUAGE": "LANGUAGE",
    "DATE": "DATE",
    "TIME": "TIME",
    "PERCENT": "PERCENT",
    "MONEY": "MONEY",
    "QUANTITY": "QUANTITY",
    "ORDINAL": "ORDINAL",
    "CARDINAL": "CARDINAL",
}


def canonicalize_type(ent_type: str) -> str:
    return _EQUIVALENT_TYPES.get(ent_type.upper(), ent_type.upper())


def _entity_spans(attrs: list[TokenAttributes]) -> list[tuple[str, str, SpanOffset]]:
    spans: list[tuple[str, str, SpanOffset]] = []
    current_tokens: list[TokenAttributes] = []
    for attr in attrs:
        if attr.ent_iob_ == "B":
            if current_tokens:
                text = " ".join(t.text for t in current_tokens)
                spans.append((text, canonicalize_type(current_tokens[0].ent_type_), current_tokens[0].offset))
            current_tokens = [attr]
        elif attr.ent_iob_ == "I":
            current_tokens.append(attr)
        elif current_tokens:
            text = " ".join(t.text for t in current_tokens)
            spans.append((text, canonicalize_type(current_tokens[0].ent_type_), current_tokens[0].offset))
            current_tokens = []
    if current_tokens:
        text = " ".join(t.text for t in current_tokens)
        spans.append((text, canonicalize_type(current_tokens[0].ent_type_), current_tokens[0].offset))
    return spans


def _pronoun_spans(attrs: list[TokenAttributes]) -> list[TokenAttributes]:
    return [a for a in attrs if a.pos_ == "PRON"]


def _cluster_by_embedding(
    items: list[tuple[str, SpanOffset, int]],
    threshold: float,
) -> list[list[tuple[str, SpanOffset, int]]]:
    if not items:
        return []
    vecs = load_embedder().encode(texts, normalize_embeddings=True)
    clusters: list[list[int]] = []
    assigned = [False] * len(items)
    for i in range(len(items)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(items)):
            if not assigned[j] and float(np.dot(vecs[i], vecs[j])) >= threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)
    return [[items[idx] for idx in cluster] for cluster in clusters]


def resolve_local(
    index: AtomIndex, pronoun_map: list[dict] | None = None
) -> dict[str, list[list[tuple[str, SpanOffset, int]]]]:
    by_type: dict[str, list[tuple[str, SpanOffset, int]]] = defaultdict(list)

    for atom_id in index.atoms:
        node = index.nodes[atom_id]
        for text, ent_type, offset in _entity_spans(node.nlp_attributes):
            by_type[ent_type].append((text, offset, atom_id))

    clusters: dict[str, list[list[tuple[str, SpanOffset, int]]]] = {}
    for ent_type, mentions in by_type.items():
        clusters[ent_type] = _cluster_by_embedding(mentions, ENTITY_MERGE_THRESHOLD)

    if pronoun_map:
        for entry in pronoun_map:
            atom_id = entry["atom_id"]
            node = index.nodes[atom_id]
            pronoun_attrs = _pronoun_spans(node.nlp_attributes)
            matched = next(
                (a for a in pronoun_attrs if a.offset.start == entry["offset"][0]),
                None,
            )
            if matched is None:
                continue
            resolution = PronounResolution(
                offset=matched.offset,
                pronoun=entry["pronoun"],
                local_entity=entry["local_entity"],
                confidence=entry.get("confidence", 1.0),
            )
            if node.disambiguation is None:
                node.disambiguation = DisambiguationMetadata(pronoun_map=[])
            node.disambiguation.pronoun_map.append(resolution)

    return clusters


def build_pronoun_prompt(index: AtomIndex) -> str:
    lines: list[str] = [
        (
            "You are a coreference resolver. For each pronoun in the atoms below, "
            "identify which named entity it refers to within this batch. "
            "Return a JSON array of objects with fields: "
            "atom_id (int), offset ([start, end]), pronoun (str), local_entity (str), confidence (float 0-1).\n"
        )
    ]
    for atom_id in index.atoms:
        node = index.nodes[atom_id]
        pronouns = _pronoun_spans(node.nlp_attributes)
        if not pronouns:
            continue
        lines.append(f"atom_id={atom_id}: {node.value}")
    return "\n".join(lines)
