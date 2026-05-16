import re
import unicodedata
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from stardust.config import ATOM_TOKEN_LIMIT
from stardust.index import OffsetMap
from stardust.tree.atom import Modality, Node, SpanOffset

HOTPOTQA_LEVELS = ["corpus", "document", "sentence"]
QASPER_LEVELS = ["document", "paragraph"]
CRAG_LEVELS = ["corpus", "page", "section", "paragraph"]


@dataclass
class ParsedNode:
    node: Node
    is_atom: bool


@dataclass
class _State:
    counter: int = 0
    raw_pos: int = 0
    clean_pos: int = 0


def _clean(raw: str) -> tuple[str, OffsetMap]:
    offset_map = OffsetMap()
    result: list[str] = []
    src_pos = dst_pos = 0
    for segment in re.split(r"(\n+)", unicodedata.normalize("NFKC", raw)):
        if not segment:
            continue
        cleaned = re.sub(r"[ \t]+", " ", segment).strip(" \t")
        if not cleaned:
            src_pos += len(segment)
            continue
        offset_map.add(src_pos, dst_pos, len(cleaned))
        result.append(cleaned)
        dst_pos += len(cleaned)
        src_pos += len(segment)
    return "".join(result), offset_map


def _token_count(text: str) -> int:
    return len(text.split())


def _make_node(
    nid: int,
    node_type: str,
    value: str,
    raw_start: int,
    raw_len: int,
    clean_start: int,
    clean_len: int,
    parent_id: int | None,
) -> Node:
    return Node(
        id=nid,
        node_type=node_type,
        modality=Modality.TEXT,
        value=value,
        raw_offset=SpanOffset(start=raw_start, end=raw_start + raw_len),
        clean_offset=SpanOffset(start=clean_start, end=clean_start + clean_len),
        parent_id=parent_id,
        nlp_attributes=[],
        disambiguation=None,
    )


async def _flush_buffer(
    buf: list[str],
    node_type: str,
    prefix: str,
    parent_id: int,
    state: _State,
) -> AsyncGenerator[ParsedNode]:
    if not buf:
        return
    combined = " ".join(buf)
    clean_text, _ = _clean(combined)
    value = f"{prefix} | {clean_text}"
    node = _make_node(
        state.counter, node_type, value, state.raw_pos, len(combined), state.clean_pos, len(clean_text), parent_id
    )
    state.counter += 1
    state.raw_pos += len(combined)
    state.clean_pos += len(clean_text)
    yield ParsedNode(node=node, is_atom=True)
    buf.clear()


async def normalize_hotpotqa(record: dict[str, Any]) -> AsyncGenerator[ParsedNode]:
    corpus_level, doc_level, atom_level = HOTPOTQA_LEVELS
    state = _State()

    corpus_text = "hotpotqa"
    corpus_clean, _ = _clean(corpus_text)
    corpus_node = _make_node(
        state.counter,
        corpus_level,
        corpus_clean,
        state.raw_pos,
        len(corpus_text),
        state.clean_pos,
        len(corpus_clean),
        None,
    )
    state.counter += 1
    state.raw_pos += len(corpus_text)
    state.clean_pos += len(corpus_clean)
    yield ParsedNode(node=corpus_node, is_atom=False)

    for title, sentences in zip(record["context"]["title"], record["context"]["sentences"], strict=False):
        doc_clean, _ = _clean(title)
        doc_value = f"{corpus_clean} | {doc_clean}"
        doc_node = _make_node(
            state.counter,
            doc_level,
            doc_value,
            state.raw_pos,
            len(title),
            state.clean_pos,
            len(doc_clean),
            corpus_node.id,
        )
        state.counter += 1
        state.raw_pos += len(title)
        state.clean_pos += len(doc_clean)
        yield ParsedNode(node=doc_node, is_atom=False)

        buffer: list[str] = []
        buffer_tokens = 0
        for sentence in (s for s in sentences if s.strip()):
            tokens = _token_count(sentence)
            if buffer_tokens + tokens > ATOM_TOKEN_LIMIT and buffer:
                async for item in _flush_buffer(buffer, atom_level, doc_value, doc_node.id, state):
                    yield item
                buffer_tokens = 0
            buffer.append(sentence)
            buffer_tokens += tokens
        async for item in _flush_buffer(buffer, atom_level, doc_value, doc_node.id, state):
            yield item


async def normalize_qasper(record: dict[str, Any]) -> AsyncGenerator[ParsedNode]:
    doc_level, atom_level = QASPER_LEVELS
    state = _State()

    context: str = record.get("context", "")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", context) if p.strip()]

    doc_clean, _ = _clean("qasper_paper")
    doc_node = _make_node(
        state.counter, doc_level, doc_clean, state.raw_pos, len("qasper_paper"), state.clean_pos, len(doc_clean), None
    )
    state.counter += 1
    state.raw_pos += len("qasper_paper")
    state.clean_pos += len(doc_clean)
    yield ParsedNode(node=doc_node, is_atom=False)

    buffer: list[str] = []
    buffer_tokens = 0
    for para in paragraphs:
        tokens = _token_count(para)
        if buffer_tokens + tokens > ATOM_TOKEN_LIMIT and buffer:
            async for item in _flush_buffer(buffer, atom_level, doc_clean, doc_node.id, state):
                yield item
            buffer_tokens = 0
        buffer.append(para)
        buffer_tokens += tokens
    async for item in _flush_buffer(buffer, atom_level, doc_clean, doc_node.id, state):
        yield item


def _is_markdown(text: str) -> bool:
    return any(line.startswith("#") for line in text.splitlines())


async def normalize_crag(record: dict[str, Any]) -> AsyncGenerator[ParsedNode]:
    corpus_level, page_level, section_level, atom_level = CRAG_LEVELS
    state = _State()

    domain = record.get("domain", "unknown")
    corpus_text = f"crag_{domain}"
    corpus_clean, _ = _clean(corpus_text)
    corpus_node = _make_node(
        state.counter,
        corpus_level,
        corpus_clean,
        state.raw_pos,
        len(corpus_text),
        state.clean_pos,
        len(corpus_clean),
        None,
    )
    state.counter += 1
    state.raw_pos += len(corpus_text)
    state.clean_pos += len(corpus_clean)
    yield ParsedNode(node=corpus_node, is_atom=False)

    for sr in record.get("search_results", []):
        snippet = sr.get("page_snippet", "").strip()
        if not snippet:
            continue
        page_name = sr.get("page_name", "")[:200]
        if _is_markdown(snippet):
            async for item in _normalize_crag_markdown(
                page_name, snippet, corpus_clean, corpus_node.id, page_level, section_level, atom_level, state
            ):
                yield item
        else:
            async for item in _normalize_crag_search_results(
                page_name, snippet, corpus_clean, corpus_node.id, page_level, atom_level, state
            ):
                yield item


async def _normalize_crag_search_results(
    page_name: str,
    snippet: str,
    corpus_value: str,
    corpus_id: int,
    page_level: str,
    atom_level: str,
    state: _State,
) -> AsyncGenerator[ParsedNode]:
    page_clean, _ = _clean(page_name)
    page_value = f"{corpus_value} | {page_clean}"
    page_node = _make_node(
        state.counter,
        page_level,
        page_value,
        state.raw_pos,
        len(page_name),
        state.clean_pos,
        len(page_clean),
        corpus_id,
    )
    state.counter += 1
    state.raw_pos += len(page_name)
    state.clean_pos += len(page_clean)
    yield ParsedNode(node=page_node, is_atom=False)

    clean_snippet, _ = _clean(snippet)
    atom_value = f"{page_value} | {clean_snippet}"
    atom_node = _make_node(
        state.counter,
        atom_level,
        atom_value,
        state.raw_pos,
        len(snippet),
        state.clean_pos,
        len(clean_snippet),
        page_node.id,
    )
    state.counter += 1
    state.raw_pos += len(snippet)
    state.clean_pos += len(clean_snippet)
    yield ParsedNode(node=atom_node, is_atom=True)


async def _normalize_crag_markdown(
    page_name: str,
    content: str,
    corpus_value: str,
    corpus_id: int,
    page_level: str,
    section_level: str,
    atom_level: str,
    state: _State,
) -> AsyncGenerator[ParsedNode]:
    filename = page_name[:200]
    page_clean, _ = _clean(filename)
    page_value = f"{corpus_value} | {page_clean}"
    page_node = _make_node(
        state.counter, page_level, page_value, state.raw_pos, len(filename), state.clean_pos, len(page_clean), corpus_id
    )
    state.counter += 1
    state.raw_pos += len(filename)
    state.clean_pos += len(page_clean)
    yield ParsedNode(node=page_node, is_atom=False)

    current_section_id = page_node.id
    current_section_value = page_value
    buffer: list[str] = []
    buffer_tokens = 0

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            async for item in _flush_buffer(buffer, atom_level, current_section_value, current_section_id, state):
                yield item
            buffer_tokens = 0
            heading = stripped.lstrip("#").strip()
            sec_clean, _ = _clean(heading)
            current_section_value = f"{page_value} | {sec_clean}"
            sec_node = _make_node(
                state.counter,
                section_level,
                current_section_value,
                state.raw_pos,
                len(heading),
                state.clean_pos,
                len(sec_clean),
                page_node.id,
            )
            current_section_id = sec_node.id
            state.counter += 1
            state.raw_pos += len(heading)
            state.clean_pos += len(sec_clean)
            yield ParsedNode(node=sec_node, is_atom=False)
        else:
            tokens = _token_count(stripped)
            if buffer_tokens + tokens > ATOM_TOKEN_LIMIT and buffer:
                async for item in _flush_buffer(buffer, atom_level, current_section_value, current_section_id, state):
                    yield item
                buffer_tokens = 0
            buffer.append(stripped)
            buffer_tokens += tokens

    async for item in _flush_buffer(buffer, atom_level, current_section_value, current_section_id, state):
        yield item
