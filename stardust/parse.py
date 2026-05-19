import html
import json
import re
import unicodedata
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

import mistletoe
from mistletoe.ast_renderer import AstRenderer

from stardust.index import OffsetMap
from stardust.tree.atom import Modality, Node, SpanOffset


def clean_value(value: str) -> str:
    raw = value.rsplit(" | ", 1)[-1] if " | " in value else value
    raw = re.sub(r"\b[0-9a-f]{32}\b", "", raw)
    raw = re.sub(r"https?%3A%2F%2F\S+", "", raw)
    raw = re.sub(r"https?://\S+", "", raw)
    return re.sub(r"\s+", " ", raw).strip()


HOTPOTQA_LEVELS = ["corpus", "record", "document", "sentence"]
QASPER_LEVELS = ["record", "document", "paragraph"]
CRAG_LEVELS = ["corpus", "record", "page", "section", "paragraph"]


@dataclass
class ParsedNode:
    node: Node
    is_atom: bool


@dataclass
class _State:
    index: int = 0
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

    return blocks


def _clean_filename(filename: str) -> str:
    s = re.sub(r"^[0-9a-f]{32}-", "", filename)
    s = unquote(s)
    m = re.search(r"https?://[^/]+(/.*)", s)
    if m:
        s = m.group(1)
    s = re.sub(r"[/_-]+", " ", s)
    s = re.sub(r"\.[a-z]{2,4}$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _make_node(
    node_type: str,
    value: str,
    raw_start: int,
    raw_len: int,
    clean_start: int,
    clean_len: int,
    parent_index: int | None,
    modality: Modality = Modality.TEXT,
    terminal: bool = False,
) -> Node:
    return Node(
        parent_index=parent_index,
        node_type=node_type,
        modality=modality,
        value=value,
        raw_offset=SpanOffset(start=raw_start, end=raw_start + raw_len),
        clean_offset=SpanOffset(start=clean_start, end=clean_start + clean_len),
        terminal=terminal,
    )


@dataclass
class TableData:
    title: str
    col_names: list[str]
    row_count: int
    rows: list[str]


_MD_NOISE = re.compile(r"!?\[[^\]]*\]\([^)]*\)")


def _cell_text(cell: str) -> str:
    return re.sub(r"\s+", " ", _MD_NOISE.sub("", cell)).strip()


def _parse_md_tables(markdown: str, last_heading: str = "") -> list[TableData]:
    tables: list[TableData] = []
    lines = markdown.splitlines()
    i = 0
    current_heading = last_heading
    while i < len(lines):
        line = lines[i]
        if re.match(r"#+\s+", line):
            current_heading = re.sub(r"^#+\s+", "", line).strip()
            i += 1
            continue
        if not re.match(r"\s*\|", line):
            i += 1
            continue
        block: list[str] = []
        while i < len(lines) and re.match(r"\s*\|", lines[i]):
            block.append(lines[i])
            i += 1
        if len(block) < 2:
            continue
        sep_idx = next((j for j, l in enumerate(block) if re.match(r"\s*\|[\s\-:|]+\|", l)), None)
        if sep_idx is None:
            continue
        header_line = block[sep_idx - 1] if sep_idx > 0 else block[0]
        col_names = [_cell_text(c) for c in header_line.strip("|\n ").split("|")]
        col_names = [c for c in col_names if c]
        if not col_names:
            continue
        data_rows = [l for j, l in enumerate(block) if j != sep_idx and (sep_idx == 0 or j != sep_idx - 1)]
        rows: list[str] = []
        for row_line in data_rows:
            cells = [_cell_text(c) for c in row_line.strip("|\n ").split("|")]
            cells = cells[: len(col_names)]
            while len(cells) < len(col_names):
                cells.append("")
            if not any(cells):
                continue
            rows.append(" | ".join(f"{col_names[k]}: {cells[k]}" for k in range(len(col_names)) if cells[k]))
        if not rows:
            continue
        tables.append(TableData(title=current_heading, col_names=col_names, row_count=len(rows), rows=rows))
    return tables


async def normalize_hotpotqa(record: dict[str, Any], record_id: str) -> AsyncGenerator[ParsedNode]:
    corpus_level, record_level, doc_level, atom_level = HOTPOTQA_LEVELS
    state = _State()

    corpus_text = "hotpotqa"
    corpus_clean, _ = _clean(corpus_text)
    corpus_index = state.index
    yield ParsedNode(
        node=_make_node(corpus_level, corpus_clean, state.raw_pos, len(corpus_text), state.clean_pos, len(corpus_clean), None),
        is_atom=False,
    )
    state.index += 1
    state.raw_pos += len(corpus_text)
    state.clean_pos += len(corpus_clean)

    record_clean, _ = _clean(record_id)
    record_value = f"{corpus_clean} | {record_clean}"
    record_index = state.index
    yield ParsedNode(
        node=_make_node(record_level, record_value, state.raw_pos, len(record_id), state.clean_pos, len(record_clean), corpus_index),
        is_atom=False,
    )
    state.index += 1
    state.raw_pos += len(record_id)
    state.clean_pos += len(record_clean)

    title: str = record.get("title", "") or ""
    text: str = record.get("text", "") or ""
    doc_clean, _ = _clean(title)
    doc_value = f"{record_value} | {doc_clean}"
    doc_index = state.index
    yield ParsedNode(
        node=_make_node(doc_level, doc_value, state.raw_pos, len(title), state.clean_pos, len(doc_clean), record_index),
        is_atom=False,
    )
    state.index += 1
    state.raw_pos += len(title)
    state.clean_pos += len(doc_clean)

    if not text.strip():
        return
    text_clean, _ = _clean(text)
    value = f"{doc_value} | {text_clean}"
    yield ParsedNode(
        node=_make_node(atom_level, value, state.raw_pos, len(text), state.clean_pos, len(text_clean), doc_index, terminal=True),
        is_atom=True,
    )


async def normalize_qasper(record: dict[str, Any], record_id: str) -> AsyncGenerator[ParsedNode]:
    record_level, doc_level, atom_level = QASPER_LEVELS
    state = _State()

    context: str = record.get("context", "")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]

    record_clean, _ = _clean(record_id)
    record_index = state.index
    yield ParsedNode(
        node=_make_node(record_level, record_clean, state.raw_pos, len(record_id), state.clean_pos, len(record_clean), None),
        is_atom=False,
    )
    state.index += 1
    state.raw_pos += len(record_id)
    state.clean_pos += len(record_clean)

    doc_clean, _ = _clean("qasper_paper")
    doc_value = f"{record_clean} | {doc_clean}"
    doc_index = state.index
    yield ParsedNode(
        node=_make_node(doc_level, doc_value, state.raw_pos, len("qasper_paper"), state.clean_pos, len(doc_clean), record_index),
        is_atom=False,
    )
    state.index += 1
    state.raw_pos += len("qasper_paper")
    state.clean_pos += len(doc_clean)

    buffer: list[str] = []
    content_limit = ATOM_TOKEN_LIMIT - _token_count(f"{doc_value} | ")
    for para in sentences:
        if _buffer_tokens([*buffer, para]) > content_limit and buffer:
            async for item in _flush_buffer(buffer, atom_level, doc_value, doc_index, state):
                yield item
        buffer.append(para)
    async for item in _flush_buffer(buffer, atom_level, doc_value, doc_index, state):
        yield item


async def normalize_crag(record: dict[str, Any], record_id: str) -> AsyncGenerator[ParsedNode]:
    corpus_level, record_level, page_level, section_level, atom_level = CRAG_LEVELS
    state = _State()

    corpus_text = "crag_open"
    corpus_clean, _ = _clean(corpus_text)
    corpus_index = state.index
    yield ParsedNode(
        node=_make_node(corpus_level, corpus_clean, state.raw_pos, len(corpus_text), state.clean_pos, len(corpus_clean), None),
        is_atom=False,
    )
    state.index += 1
    state.raw_pos += len(corpus_text)
    state.clean_pos += len(corpus_clean)

    record_clean, _ = _clean(record_id)
    record_value = f"{corpus_clean} | {record_clean}"
    record_index = state.index
    yield ParsedNode(
        node=_make_node(record_level, record_value, state.raw_pos, len(record_id), state.clean_pos, len(record_clean), corpus_index),
        is_atom=False,
    )
    state.index += 1
    state.raw_pos += len(record_id)
    state.clean_pos += len(record_clean)

    content: str = record.get("markdown", "") or ""
    filename: str = (record.get("filename", "") or "")[:200]
    if not content.strip():
        return

    page_clean, _ = _clean(_clean_filename(filename))
    page_value = f"{record_value} | {page_clean}"
    page_index = state.index
    yield ParsedNode(
        node=_make_node(page_level, page_value, state.raw_pos, len(filename), state.clean_pos, len(page_clean), record_index),
        is_atom=False,
    )
    state.index += 1
    state.raw_pos += len(filename)
    state.clean_pos += len(page_clean)

    blocks = _md_blocks(content)
    if not blocks:
        return

    current_section_index = page_index
    current_section_value = page_value
    buffer: list[str] = []

    for block_type, text in blocks:
        if block_type == "heading":
            sec_clean, _ = _clean(text)
            current_section_value = f"{page_value} | {sec_clean}"
            current_section_index = state.index
            yield ParsedNode(
                node=_make_node(
                    section_level,
                    current_section_value,
                    state.raw_pos,
                    len(text),
                    state.clean_pos,
                    len(sec_clean),
                    page_index,
                ),
                is_atom=False,
            )
            state.index += 1
            state.raw_pos += len(text)
            state.clean_pos += len(sec_clean)
            content_limit = ATOM_TOKEN_LIMIT - _token_count(f"{current_section_value} | ")
            if _buffer_tokens([*buffer, text]) > content_limit and buffer:
                async for item in _flush_buffer(
                    buffer, atom_level, current_section_value, current_section_index, state
                ):
                    yield item
            buffer.append(text)
        else:
            content_limit = ATOM_TOKEN_LIMIT - _token_count(f"{current_section_value} | ")
            if _buffer_tokens([*buffer, text]) > content_limit and buffer:
                async for item in _flush_buffer(
                    buffer, atom_level, current_section_value, current_section_index, state
                ):
                    yield item
            buffer.append(text)

    async for item in _flush_buffer(buffer, atom_level, current_section_value, current_section_index, state):
        yield item
