import json

from stardust.tree.atom import AtomIndex, DisambiguationMetadata, PronounResolution, SpanOffset, TokenAttributes


def _entity_spans(attrs: list[TokenAttributes]) -> list[tuple[str, str, SpanOffset]]:
    spans: list[tuple[str, str, SpanOffset]] = []
    current: list[TokenAttributes] = []
    for attr in attrs:
        if attr.ent_iob_ == "B":
            if current:
                spans.append((" ".join(t.text for t in current), current[0].ent_type_, current[0].offset))
            current = [attr]
        elif attr.ent_iob_ == "I":
            current.append(attr)
        elif current:
            spans.append((" ".join(t.text for t in current), current[0].ent_type_, current[0].offset))
            current = []
    if current:
        spans.append((" ".join(t.text for t in current), current[0].ent_type_, current[0].offset))
    return spans


def _pronoun_spans(attrs: list[TokenAttributes]) -> list[TokenAttributes]:
    return [a for a in attrs if a.pos_ == "PRON"]


def _relation_triples(attrs: list[TokenAttributes], entity_start: int, entity_end: int) -> str:
    entity_indices = {i for i, a in enumerate(attrs) if entity_start <= a.offset.start < entity_end}
    triples: list[str] = []
    for i, token in enumerate(attrs):
        if i not in entity_indices:
            continue
        if token.dep_ in ("nsubj", "nsubjpass"):
            head = next((a for a in attrs if a.dep_ == "ROOT"), None)
            if head:
                obj = next((a for a in attrs if a.dep_ in ("attr", "dobj", "pobj")), None)
                if obj:
                    triples.append(f"{token.text} {head.text} {obj.text}")
        elif token.dep_ in ("attr", "appos"):
            subj = next((a for a in attrs if a.dep_ in ("nsubj", "nsubjpass")), None)
            if subj:
                triples.append(f"{subj.text} is {token.text}")
    return " ".join(triples)


def _context_window(attrs: list[TokenAttributes], entity_start: int, entity_end: int, window: int = 5) -> str:
    entity_indices = [i for i, a in enumerate(attrs) if entity_start <= a.offset.start < entity_end]
    if not entity_indices:
        return ""
    lo = max(0, entity_indices[0] - window)
    hi = min(len(attrs), entity_indices[-1] + window + 1)
    return " ".join(a.text for a in attrs[lo:hi])


def collect_entity_mentions(
    record_id: str, index: AtomIndex
) -> list[tuple[str, str, SpanOffset, int, str]]:
    mentions = []
    for atom_id in index.atoms:
        node = index.nodes[atom_id]
        for surface, ent_type, offset in _entity_spans(node.nlp_attributes):
            if not ent_type:
                continue
            span_tokens = [a for a in node.nlp_attributes if offset.start <= a.offset.start < offset.end]
            if not any(t.pos_ in ("NOUN", "PROPN") for t in span_tokens):
                continue
            window = _context_window(node.nlp_attributes, offset.start, offset.end)
            relations = _relation_triples(node.nlp_attributes, offset.start, offset.end)
            context = " ".join(filter(None, [surface, window, relations]))
            mentions.append((surface, ent_type, offset, atom_id, context))
    return mentions


def attach_pronoun_resolutions(index: AtomIndex, pronoun_map: list[dict]) -> None:
    for entry in pronoun_map:
        atom_id = entry.get("atom_id")
        if atom_id not in index.nodes:
            continue
        node = index.nodes[atom_id]
        offset_val = entry.get("offset", [])
        matched = next(
            (
                a
                for a in _pronoun_spans(node.nlp_attributes)
                if len(offset_val) == 2 and a.offset.start == offset_val[0]
            ),
            None,
        )
        if matched is None:
            continue
        if node.disambiguation is None:
            node.disambiguation = DisambiguationMetadata(pronoun_map=[])
        node.disambiguation.pronoun_map.append(
            PronounResolution(
                offset=matched.offset,
                pronoun=entry.get("pronoun", matched.text),
                local_entity=entry.get("local_entity", ""),
                confidence=float(entry.get("confidence", 1.0)),
            )
        )


def build_pronoun_prompt(atom_id: int, value: str, attrs: list[TokenAttributes]) -> dict | None:
    pronouns = _pronoun_spans(attrs)
    if not pronouns:
        return None
    entities = _entity_spans(attrs)
    return {
        "atom_id": atom_id,
        "text": value,
        "entities": [{"text": t, "type": et, "offset": [o.start, o.end]} for t, et, o in entities],
        "pronouns": [
            {"text": p.text, "offset": [p.offset.start, p.offset.end], "dep": p.dep_, "morph": p.morph}
            for p in pronouns
        ],
    }


def build_batch_prompt(atom_data: list[dict]) -> str:
    if not atom_data:
        return ""
    lines = []
    for a in atom_data:
        entities = ", ".join(f"{e['text']}({e['type']})" for e in a["entities"])
        pronouns = ", ".join(f"{p['text']}@{p['offset'][0]}" for p in a["pronouns"])
        lines.append(f"[{a['atom_id']}] {a['text']} | entities: {entities} | pronouns: {pronouns}")
    examples = (
        "[12] Sarah joined the firm in 2005. She became partner within three years. | entities: Sarah(PERSON), firm(ORG) | pronouns: She@36\n"
        "=> [{\"atom_id\":12,\"offset\":[36,39],\"pronoun\":\"She\",\"local_entity\":\"Sarah\",\"confidence\":0.99}]\n"
        "[47] The treaty was signed by France and Germany. It came into force in 1920. | entities: France(GPE), Germany(GPE) | pronouns: It@50\n"
        "=> [{\"atom_id\":47,\"offset\":[50,52],\"pronoun\":\"It\",\"local_entity\":\"treaty\",\"confidence\":0.85}]\n"
        "[83] NASA launched the probe. Engineers monitored its trajectory closely. | entities: NASA(ORG), probe(PRODUCT) | pronouns: its@46\n"
        "=> [{\"atom_id\":83,\"offset\":[46,49],\"pronoun\":\"its\",\"local_entity\":\"probe\",\"confidence\":0.97}]"
    )
    return (
        "Resolve pronoun coreference. For each pronoun, output one JSON object: atom_id, offset([start,end]), pronoun, local_entity, confidence.\n\n"
        f"{examples}\n\n"
        + "\n".join(lines) + "\n=>"
    )
