from collections.abc import AsyncGenerator

from stardust.config import LLM_BATCH_TOKEN_LIMIT, NUMERIC_ENTITY_TYPES
from stardust.registry import llm_tokenizer
from stardust.tree.atom import SpanOffset, TokenAttributes


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


def collect_entity_mentions(
    record_id: str, rows: list[tuple[int, list, dict | None]]
) -> list[tuple[str, str, SpanOffset, int, str]]:
    mentions = []
    for atom_id, nlp_attrs, disambiguation in rows:
        attrs = [TokenAttributes(**a) for a in (nlp_attrs or [])]
        for surface, ent_type, offset in _entity_spans(attrs):
            if not ent_type or ent_type in NUMERIC_ENTITY_TYPES:
                continue
            window = _context_window(attrs, offset.start, offset.end)
            context = " ".join(filter(None, [surface, window]))
            mentions.append((surface, ent_type, offset, atom_id, context))
        if disambiguation:
            for resolution in disambiguation.get("pronoun_map") or []:
                referent = resolution.get("referent", "")
                if not referent:
                    continue
                offset = SpanOffset(**resolution["offset"])
                window = _context_window(attrs, offset.start, offset.end)
                context = " ".join(filter(None, [referent, window]))
                mentions.append((referent, "COREF", offset, atom_id, context))
    return mentions


_PREAMBLE = (
    "Resolve coreference. Output a single JSON object mapping token_id to referent_token_id. "
    "Only include pronouns or nominals that clearly refer to another token. "
    "Do not map predicate nominals or role descriptions.\n"
    "Example: [0] Sarah joined the firm. She became partner.\ntokens: 0:Sarah, 1:firm, 2:She, 3:partner\n"
    "[1] The treaty was signed by France. It came into force.\ntokens: 4:treaty, 5:France, 6:It\n"
    '=> {"2":0,"6":4}\n\n'
)


async def build_batch_prompts(
    rows: AsyncGenerator[tuple[int, str, list]],
) -> AsyncGenerator[tuple[str, list[tuple[int, str, list]]]]:
    tokenizer = llm_tokenizer()
    preamble_tokens = len(tokenizer.encode(_PREAMBLE, add_special_tokens=False))
    budget = LLM_BATCH_TOKEN_LIMIT - preamble_tokens
    sections: list[str] = []
    batch_rows: list[tuple[int, str, list]] = []
    batch_tokens = 0
    atom_counter = 0
    token_counter = 0
    async for atom_id, value, nlp_attrs in rows:
        attrs = [TokenAttributes(**a) for a in (nlp_attrs or [])]
        nominals = _nominal_spans(attrs)
        if not nominals:
            continue
        leaf = value.rsplit(" | ", 1)[-1] if " | " in value else value
        tokens_str = ", ".join(f"{token_counter + j}:{t.text}" for j, t in enumerate(nominals))
        section = f"[{atom_counter}] {leaf}\ntokens: {tokens_str}"
        section_tokens = len(tokenizer.encode(section, add_special_tokens=False))
        if sections and batch_tokens + section_tokens > budget:
            yield _PREAMBLE + "\n\n".join(sections), batch_rows
            sections = []
            batch_rows = []
            batch_tokens = 0
            atom_counter = 0
            token_counter = 0
            tokens_str = ", ".join(f"{j}:{t.text}" for j, t in enumerate(nominals))
            section = f"[0] {leaf}\ntokens: {tokens_str}"
            section_tokens = len(tokenizer.encode(section, add_special_tokens=False))
        sections.append(section)
        batch_rows.append((atom_id, value, nlp_attrs))
        batch_tokens += section_tokens
        atom_counter += 1
        token_counter += len(nominals)
    if sections:
        yield _PREAMBLE + "\n\n".join(sections), batch_rows
