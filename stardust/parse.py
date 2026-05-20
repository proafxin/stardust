import html
import re
import unicodedata
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from stardust.atom import Modality, Node

HOTPOTQA_LEVELS = ["corpus", "record", "document", "text"]


@dataclass
class ParsedNode:
    node: Node
    is_atom: bool
    sentences: list[tuple[str, str]]


def _clean(raw: str) -> str:
    result: list[str] = []
    raw = html.unescape(raw)
    for segment in re.split(r"(\n+)", unicodedata.normalize("NFKC", raw)):
        if not segment:
            continue
        cleaned = re.sub(r"[ \t]+", " ", segment).strip(" \t")
        cleaned = re.sub(r"\.{3,}", "...", cleaned)
        cleaned = re.sub(r"-{3,}", "—", cleaned)
        if cleaned:
            result.append(cleaned)
    return "".join(result)


def _make_node(
    node_type: str, value: str, parent_index: int | None, modality: Modality = Modality.TEXT, terminal: bool = False
) -> Node:
    return Node(parent_index=parent_index, node_type=node_type, modality=modality, value=value, terminal=terminal)


async def normalize_hotpotqa(record: dict[str, Any], record_id: str) -> AsyncGenerator[ParsedNode]:
    corpus_level, record_level, doc_level, atom_level = HOTPOTQA_LEVELS
    index = 0

    corpus_text = "hotpotqa"
    corpus_clean = _clean(corpus_text)
    corpus_index = index
    yield ParsedNode(node=_make_node(corpus_level, corpus_clean, None), is_atom=False, sentences=[])
    index += 1

    record_clean = _clean(record_id)
    record_value = f"{corpus_clean} | {record_clean}"
    record_index = index
    yield ParsedNode(node=_make_node(record_level, record_value, corpus_index), is_atom=False, sentences=[])
    index += 1

    for title, sentences in record.get("context", []):
        title_clean = _clean(title)
        doc_value = f"{record_value} | {title_clean}"
        doc_index = index
        yield ParsedNode(node=_make_node(doc_level, doc_value, record_index), is_atom=False, sentences=[])
        index += 1

        text = " ".join(s.strip() for s in sentences if s.strip())
        if not text:
            continue
        text_clean = _clean(text)
        clean_sentences = [(s.strip(), f"{doc_value} | {s.strip()}") for s in sentences if s.strip()]
        yield ParsedNode(
            node=_make_node(atom_level, text_clean, doc_index, terminal=True),
            is_atom=True,
            sentences=clean_sentences,
        )
        index += 1
