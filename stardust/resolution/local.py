from stardust.config import NUMERIC_ENTITY_TYPES
from stardust.parse import clean_value
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


def _nominal_spans(attrs: list[TokenAttributes]) -> list[TokenAttributes]:
    return [a for a in attrs if a.pos_ in {"NOUN", "PROPN", "PRON"}]


def _relation_triples(attrs: list[TokenAttributes], entity_start: int, entity_end: int) -> str:
    entity_indices = {i for i, a in enumerate(attrs) if entity_start <= a.offset.start < entity_end}
    triples: list[str] = []
    for i, token in enumerate(attrs):
        if i not in entity_indices:
            continue
        if token.dep_ in {"nsubj", "nsubjpass"}:
            head = next((a for a in attrs if a.dep_ == "ROOT"), None)
            if head:
                obj = next((a for a in attrs if a.dep_ in {"attr", "dobj", "pobj"}), None)
                if obj:
                    triples.append(f"{token.text} {head.text} {obj.text}")
        elif token.dep_ in {"attr", "appos"}:
            subj = next((a for a in attrs if a.dep_ in {"nsubj", "nsubjpass"}), None)
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


def collect_entity_mentions(record_id: str, index: AtomIndex) -> list[tuple[str, str, SpanOffset, int, str]]:
    mentions = []
    for atom_id in index.atoms:
        node = index.nodes[atom_id]
        for surface, ent_type, offset in _entity_spans(node.nlp_attributes):
            if not ent_type or ent_type in NUMERIC_ENTITY_TYPES:
                continue
            window = _context_window(node.nlp_attributes, offset.start, offset.end)
            context = " ".join(filter(None, [surface, window]))
            mentions.append((surface, ent_type, offset, atom_id, context))
        if node.disambiguation:
            for resolution in node.disambiguation.pronoun_map:
                if not resolution.referent:
                    continue
                window = _context_window(node.nlp_attributes, resolution.offset.start, resolution.offset.end)
                context = " ".join(filter(None, [resolution.referent, window]))
                mentions.append((resolution.referent, "COREF", resolution.offset, atom_id, context))
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
                for a in _nominal_spans(node.nlp_attributes)
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
                token=entry.get("token", matched.text),
                referent=entry.get("referent", ""),
                confidence=float(entry.get("confidence", 1.0)),
            )
        )


def build_pronoun_prompt(atom_id: int, value: str, attrs: list[TokenAttributes]) -> dict | None:
    nominals = _nominal_spans(attrs)
    if not nominals:
        return None
    return {"atom_id": atom_id, "text": clean_value(value), "nominals": nominals}


def build_batch_prompt(atom_data: list[dict]) -> str:
    if not atom_data:
        return ""
    texts = "\n\n".join(a["text"] for a in atom_data)
    token_counter = 0
    tokens = []
    for a in atom_data:
        for t in a["nominals"]:
            tokens.append(f"{token_counter}:{t.text}")
            token_counter += 1
    examples = (
        "Sarah joined the firm. She became partner.\n\nThe treaty was signed by France. It came into force.\n\n"
        "tokens: 0:Sarah, 1:firm, 2:She, 3:partner, 4:treaty, 5:France, 6:It\n"
        '=> {"2":0,"3":1,"6":4}'
    )
    return (
        "Resolve coreference. Output a JSON object mapping token_id to referent_token_id. Only include tokens where a clear referent exists.\n\n"
        f"{examples}\n\n{texts}\n\ntokens: {', '.join(tokens)}\n=>"
    )
