import html
import json
import re
import unicodedata
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import unquote

import mistletoe
from mistletoe.ast_renderer import AstRenderer

from stardust.config import ATOM_TOKEN_LIMIT
from stardust.index import OffsetMap
from stardust.registry import embedder


def clean_value(value: str) -> str:
    raw = value.rsplit(" | ", 1)[-1] if " | " in value else value
    raw = re.sub(r"\b[0-9a-f]{32}\b", "", raw)
    raw = re.sub(r"https?%3A%2F%2F\S+", "", raw)
    raw = re.sub(r"https?://\S+", "", raw)
    return re.sub(r"\s+", " ", raw).strip()


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
    raw = html.unescape(raw)
    for segment in re.split(r"(\n+)", unicodedata.normalize("NFKC", raw)):
        if not segment:
            continue
        cleaned = re.sub(r"[ \t]+", " ", segment).strip(" \t")
        cleaned = re.sub(r"\.{3,}", "...", cleaned)
        cleaned = re.sub(r"-{3,}", "—", cleaned)
        if not cleaned:
            src_pos += len(segment)
            continue
        offset_map.add(src_pos, dst_pos, len(cleaned))
        result.append(cleaned)
        dst_pos += len(cleaned)
        src_pos += len(segment)
    return "".join(result), offset_map


@lru_cache(maxsize=65536)
def _token_count(text: str) -> int:
    return len(embedder().tokenizer.encode(text, add_special_tokens=False, verbose=False))


def _buffer_tokens(buf: list[str]) -> int:
    if not buf:
        return 0
    combined = " ".join(buf)
    cleaned, _ = _clean(combined)
    return _token_count(cleaned)


def _extract_text(node: dict) -> str:
    t = node.get("type", "")
    if t in {"BlockCode", "CodeFence", "ThematicBreak", "LineBreak", "EscapeSequence", "AutoLink", "InlineCode"}:
        return ""
    if t == "Image":
        return " ".join(_extract_text(c) for c in (node.get("children") or []))
    if t == "Link":
        return " ".join(_extract_text(c) for c in (node.get("children") or []))
    if t == "RawText":
        content = html.unescape(node.get("content", ""))
        content = re.sub(r"[\[\]!*_`#>]", "", content)
        content = re.sub(r"\S+\.svg\S*", "", content)
        content = re.sub(r"/assets/\S*", "", content)
        return re.sub(r"\s+", " ", content).strip()
    return " ".join(_extract_text(c) for c in (node.get("children") or []))


def _md_blocks(markdown: str) -> list[tuple[str, str]]:
    with AstRenderer() as renderer:
        ast = json.loads(renderer.render(mistletoe.Document(markdown)))
    blocks: list[tuple[str, str]] = []
    for node in ast.get("children") or []:
        t = node.get("type", "")
        if t in {"Heading", "SetextHeading"}:
            text = _extract_text(node).strip()
            if text:
                blocks.append(("heading", text))
        elif t in {"Paragraph", "Quote"} or t in {"List", "Quote"}:
            text = _extract_text(node).strip()
            if text:
                blocks.append(("paragraph", text))
        elif t == "Table":
            for row in node.get("children") or []:
                cells = [_extract_text(cell).strip() for cell in (row.get("children") or [])]
                text = " | ".join(c for c in cells if c)
                if text:
                    blocks.append(("paragraph", text))
    return blocks


def _clean_filename(filename: str) -> str:
    # strip leading hash prefix: "<hex>-https%3A%2F%2F..."
    s = re.sub(r"^[0-9a-f]{32}-", "", filename)
    s = unquote(s)
    # extract just the path portion after the domain
    m = re.search(r"https?://[^/]+(/.*)", s)
    if m:
        s = m.group(1)
    # replace slashes, dashes, underscores with spaces and clean up
    s = re.sub(r"[/_-]+", " ", s)
    s = re.sub(r"\.[a-z]{2,4}$", "", s)  # strip file extension
    return re.sub(r"\s+", " ", s).strip()


def _make_node(
    nid: int,
    node_type: str,
    value: str,
    raw_start: int,
    raw_len: int,
    clean_start: int,
    clean_len: int,
    parent_id: int | None,
    terminal: bool = False,
) -> Node:
    return Node(
        transient_id=nid,
        transient_parent_id=parent_id,
        node_type=node_type,
        modality=Modality.TEXT,
        value=value,
        raw_offset=SpanOffset(start=raw_start, end=raw_start + raw_len),
        clean_offset=SpanOffset(start=clean_start, end=clean_start + clean_len),
        terminal=terminal,
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
        state.counter, node_type, value, state.raw_pos, len(combined), state.clean_pos, len(clean_text), parent_id, True
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

    title: str = record.get("title", "") or ""
    text: str = record.get("text", "") or ""
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
        corpus_node.transient_id,
    )
    state.counter += 1
    state.raw_pos += len(title)
    state.clean_pos += len(doc_clean)
    yield ParsedNode(node=doc_node, is_atom=False)

    buffer: list[str] = []
    content_limit = ATOM_TOKEN_LIMIT - _token_count(f"{doc_value} | ")
    for sentence in (s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()):
        if _buffer_tokens([*buffer, sentence]) > content_limit and buffer:
            async for item in _flush_buffer(buffer, atom_level, doc_value, doc_node.transient_id, state):
                yield item
        buffer.append(sentence)
    async for item in _flush_buffer(buffer, atom_level, doc_value, doc_node.transient_id, state):
        yield item


async def normalize_qasper(record: dict[str, Any]) -> AsyncGenerator[ParsedNode]:
    doc_level, atom_level = QASPER_LEVELS
    state = _State()

    context: str = record.get("context", "")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]

    doc_clean, _ = _clean("qasper_paper")
    doc_node = _make_node(
        state.counter, doc_level, doc_clean, state.raw_pos, len("qasper_paper"), state.clean_pos, len(doc_clean), None
    )
    state.counter += 1
    state.raw_pos += len("qasper_paper")
    state.clean_pos += len(doc_clean)
    yield ParsedNode(node=doc_node, is_atom=False)

    buffer: list[str] = []
    content_limit = ATOM_TOKEN_LIMIT - _token_count(f"{doc_clean} | ")
    for para in sentences:
        if _buffer_tokens([*buffer, para]) > content_limit and buffer:
            async for item in _flush_buffer(buffer, atom_level, doc_clean, doc_node.transient_id, state):
                yield item
        buffer.append(para)
    async for item in _flush_buffer(buffer, atom_level, doc_clean, doc_node.transient_id, state):
        yield item


async def normalize_crag(record: dict[str, Any]) -> AsyncGenerator[ParsedNode]:
    corpus_level, page_level, section_level, atom_level = CRAG_LEVELS
    state = _State()

    corpus_text = "crag_open"
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

    content: str = record.get("markdown", "") or ""
    filename: str = (record.get("filename", "") or "")[:200]
    if not content.strip():
        return

    page_clean, _ = _clean(_clean_filename(filename))
    page_value = f"{corpus_clean} | {page_clean}"
    page_node = _make_node(
        state.counter,
        page_level,
        page_value,
        state.raw_pos,
        len(filename),
        state.clean_pos,
        len(page_clean),
        corpus_node.transient_id,
    )
    state.counter += 1
    state.raw_pos += len(filename)
    state.clean_pos += len(page_clean)
    yield ParsedNode(node=page_node, is_atom=False)

    blocks = _md_blocks(content)
    if not blocks:
        return

    current_section_id = page_node.transient_id
    current_section_value = page_value
    buffer: list[str] = []

    for block_type, text in blocks:
        if block_type == "heading":
            sec_clean, _ = _clean(text)
            current_section_value = f"{page_value} | {sec_clean}"
            sec_node = _make_node(
                state.counter,
                section_level,
                current_section_value,
                state.raw_pos,
                len(text),
                state.clean_pos,
                len(sec_clean),
                page_node.transient_id,
            )
            current_section_id = sec_node.transient_id
            state.counter += 1
            state.raw_pos += len(text)
            state.clean_pos += len(sec_clean)
            yield ParsedNode(node=sec_node, is_atom=False)
            content_limit = ATOM_TOKEN_LIMIT - _token_count(f"{current_section_value} | ")
            if _buffer_tokens([*buffer, text]) > content_limit and buffer:
                async for item in _flush_buffer(buffer, atom_level, current_section_value, current_section_id, state):
                    yield item
            buffer.append(text)
        else:
            content_limit = ATOM_TOKEN_LIMIT - _token_count(f"{current_section_value} | ")
            if _buffer_tokens([*buffer, text]) > content_limit and buffer:
                async for item in _flush_buffer(buffer, atom_level, current_section_value, current_section_id, state):
                    yield item
            buffer.append(text)

    async for item in _flush_buffer(buffer, atom_level, current_section_value, current_section_id, state):
        yield item
