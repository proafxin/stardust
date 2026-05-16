import asyncio
import json
from collections import defaultdict

import numpy as np

from stardust.config import ENTITY_MERGE_THRESHOLD
from stardust.registry import embedder as load_embedder
from stardust.tree.atom import AtomIndex, DisambiguationMetadata, PronounResolution, SpanOffset, TokenAttributes

_EQUIVALENT_TYPES: dict[str, str] = {
    "ORG": "ORG", "COMPANY": "ORG", "CORP": "ORG",
    "GPE": "LOCATION", "LOC": "LOCATION", "LOCATION": "LOCATION",
    "PERSON": "PERSON", "PER": "PERSON",
    "NORP": "NORP", "FAC": "FAC", "PRODUCT": "PRODUCT",
    "EVENT": "EVENT", "WORK_OF_ART": "WORK_OF_ART", "LAW": "LAW",
    "LANGUAGE": "LANGUAGE", "DATE": "DATE", "TIME": "TIME",
    "PERCENT": "PERCENT", "MONEY": "MONEY", "QUANTITY": "QUANTITY",
    "ORDINAL": "ORDINAL", "CARDINAL": "CARDINAL",
}


def canonicalize_type(ent_type: str) -> str:
    return _EQUIVALENT_TYPES.get(ent_type.upper(), ent_type.upper())


def _entity_spans(attrs: list[TokenAttributes]) -> list[tuple[str, str, SpanOffset]]:
    spans: list[tuple[str, str, SpanOffset]] = []
    current: list[TokenAttributes] = []
    for attr in attrs:
        if attr.ent_iob_ == "B":
            if current:
                spans.append((" ".join(t.text for t in current), canonicalize_type(current[0].ent_type_), current[0].offset))
            current = [attr]
        elif attr.ent_iob_ == "I":
            current.append(attr)
        elif current:
            spans.append((" ".join(t.text for t in current), canonicalize_type(current[0].ent_type_), current[0].offset))
            current = []
    if current:
        spans.append((" ".join(t.text for t in current), canonicalize_type(current[0].ent_type_), current[0].offset))
    return spans


def _pronoun_spans(attrs: list[TokenAttributes]) -> list[TokenAttributes]:
    return [a for a in attrs if a.pos_ == "PRON"]


def _cluster_sync(items: list[tuple[str, SpanOffset, int]], threshold: float) -> list[list[tuple[str, SpanOffset, int]]]:
    if not items:
        return []
    texts = [item[0] for item in items]
    vecs = load_embedder().encode(texts, normalize_embeddings=True)
    assigned = [False] * len(items)
    clusters: list[list[int]] = []
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


async def cluster_by_embedding(
    items: list[tuple[str, SpanOffset, int]],
    threshold: float,
) -> list[list[tuple[str, SpanOffset, int]]]:
    return await asyncio.to_thread(_cluster_sync, items, threshold)


async def resolve_local(
    index: AtomIndex,
) -> dict[str, list[list[tuple[str, SpanOffset, int]]]]:
    by_type: dict[str, list[tuple[str, SpanOffset, int]]] = defaultdict(list)
    for atom_id in index.atoms:
        node = index.nodes[atom_id]
        for text, ent_type, offset in _entity_spans(node.nlp_attributes):
            by_type[ent_type].append((text, offset, atom_id))

    clusters: dict[str, list[list[tuple[str, SpanOffset, int]]]] = {}
    for ent_type, mentions in by_type.items():
        clusters[ent_type] = await cluster_by_embedding(mentions, ENTITY_MERGE_THRESHOLD)

    return clusters


def attach_pronoun_resolutions(
    index: AtomIndex,
    pronoun_map: list[dict],
) -> None:
    for entry in pronoun_map:
        atom_id = entry.get("atom_id")
        if atom_id not in index.nodes:
            continue
        node = index.nodes[atom_id]
        offset_val = entry.get("offset", [])
        matched = next(
            (a for a in _pronoun_spans(node.nlp_attributes)
             if len(offset_val) == 2 and a.offset.start == offset_val[0]),
            None,
        )
        if matched is None:
            continue
        if node.disambiguation is None:
            node.disambiguation = DisambiguationMetadata(pronoun_map=[])
        node.disambiguation.pronoun_map.append(PronounResolution(
            offset=matched.offset,
            pronoun=entry.get("pronoun", matched.text),
            local_entity=entry.get("local_entity", ""),
            confidence=float(entry.get("confidence", 1.0)),
        ))


def build_pronoun_prompt(atom_id: int, value: str, attrs: list[TokenAttributes]) -> dict | None:
    pronouns = _pronoun_spans(attrs)
    if not pronouns:
        return None
    entities = _entity_spans(attrs)
    return {
        "atom_id": atom_id,
        "text": value,
        "entities": [{"text": t, "type": et, "offset": [o.start, o.end]} for t, et, o in entities],
        "pronouns": [{"text": p.text, "offset": [p.offset.start, p.offset.end], "dep": p.dep_, "morph": p.morph} for p in pronouns],
    }


def build_batch_prompt(atom_data: list[dict]) -> str:
    if not atom_data:
        return ""
    return (
        "You are a coreference resolver. For each pronoun below, identify which named entity "
        "within this batch it refers to. Use the entity list, dependency relation (dep), and "
        "morphological features (morph) as signals.\n"
        "Return a JSON array only, no other text. Each object must have:\n"
        "  atom_id (int), offset ([start, end]), pronoun (str), local_entity (str), confidence (float 0-1)\n\n"
        f"Atoms:\n{json.dumps(atom_data, indent=2)}"
    )
