import re
import unicodedata
from collections.abc import AsyncGenerator
from typing import Any

from stardust.config import ATOM_TOKEN_LIMIT
from stardust.index import OffsetMap
from stardust.tree.atom import Modality, Node, SpanOffset

HOTPOTQA_LEVELS = ["corpus", "document", "sentence"]
QASPER_LEVELS = ["document", "paragraph"]
CRAG_LEVELS = ["corpus", "page", "section", "paragraph"]


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


async def normalize_hotpotqa(record: dict[str, Any]) -> AsyncGenerator[tuple[Node, bool], None]:
    corpus_level, doc_level, atom_level = HOTPOTQA_LEVELS
    counter = 0
    raw_pos = clean_pos = 0

    corpus_text = "hotpotqa"
    corpus_clean, _ = _clean(corpus_text)
    corpus_node = _make_node(counter, corpus_level, corpus_clean, raw_pos, len(corpus_text), clean_pos, len(corpus_clean), None)
    counter += 1
    raw_pos += len(corpus_text)
    clean_pos += len(corpus_clean)
    yield corpus_node, False

    for title, sentences in zip(record["context"]["title"], record["context"]["sentences"], strict=False):
        doc_clean, _ = _clean(title)
        doc_value = f"{corpus_clean} | {doc_clean}"
        doc_node = _make_node(counter, doc_level, doc_value, raw_pos, len(title), clean_pos, len(doc_clean), corpus_node.id)
        counter += 1
        raw_pos += len(title)
        clean_pos += len(doc_clean)
        yield doc_node, False

        buffer: list[str] = []
        buffer_tokens = 0

        async def flush_hotpotqa(buf: list[str], parent_id: int) -> AsyncGenerator[tuple[Node, bool], None]:
            nonlocal counter, raw_pos, clean_pos
            if not buf:
                return
            combined = " ".join(buf)
            clean_text, _ = _clean(combined)
            value = f"{doc_value} | {clean_text}"
            node = _make_node(counter, atom_level, value, raw_pos, len(combined), clean_pos, len(clean_text), parent_id)
            counter += 1
            raw_pos += len(combined)
            clean_pos += len(clean_text)
            yield node, True
            buf.clear()

        for sentence in (s for s in sentences if s.strip()):
            tokens = _token_count(sentence)
            if buffer_tokens + tokens > ATOM_TOKEN_LIMIT and buffer:
                async for item in flush_hotpotqa(buffer, doc_node.id):
                    yield item
                buffer_tokens = 0
            buffer.append(sentence)
            buffer_tokens += tokens
        async for item in flush_hotpotqa(buffer, doc_node.id):
            yield item


async def normalize_qasper(record: dict[str, Any]) -> AsyncGenerator[tuple[Node, bool], None]:
    doc_level, atom_level = QASPER_LEVELS
    counter = 0
    raw_pos = clean_pos = 0

    context: str = record.get("context", "")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", context) if p.strip()]

    doc_clean, _ = _clean("qasper_paper")
    doc_node = _make_node(counter, doc_level, doc_clean, raw_pos, len("qasper_paper"), clean_pos, len(doc_clean), None)
    counter += 1
    raw_pos += len("qasper_paper")
    clean_pos += len(doc_clean)
    yield doc_node, False

    buffer: list[str] = []
    buffer_tokens = 0

    async def flush_qasper(buf: list[str]) -> AsyncGenerator[tuple[Node, bool], None]:
        nonlocal counter, raw_pos, clean_pos
        if not buf:
            return
        combined = " ".join(buf)
        clean_text, _ = _clean(combined)
        value = f"{doc_clean} | {clean_text}"
        node = _make_node(counter, atom_level, value, raw_pos, len(combined), clean_pos, len(clean_text), doc_node.id)
        counter += 1
        raw_pos += len(combined)
        clean_pos += len(clean_text)
        yield node, True
        buf.clear()

    for para in paragraphs:
        tokens = _token_count(para)
        if buffer_tokens + tokens > ATOM_TOKEN_LIMIT and buffer:
            async for item in flush_qasper(buffer):
                yield item
            buffer_tokens = 0
        buffer.append(para)
        buffer_tokens += tokens
    async for item in flush_qasper(buffer):
        yield item


async def normalize_crag(record: dict[str, Any], groundingdata: dict[str, Any] | None = None) -> AsyncGenerator[tuple[Node, bool], None]:
    corpus_level, page_level, section_level, atom_level = CRAG_LEVELS
    counter = 0
    raw_pos = clean_pos = 0

    domain = record.get("domain", "unknown")
    corpus_text = f"crag_{domain}"
    corpus_clean, _ = _clean(corpus_text)
    corpus_node = _make_node(counter, corpus_level, corpus_clean, raw_pos, len(corpus_text), clean_pos, len(corpus_clean), None)
    counter += 1
    raw_pos += len(corpus_text)
    clean_pos += len(corpus_clean)
    yield corpus_node, False

    if groundingdata:
        async for item in _normalize_crag_markdown(
            groundingdata, corpus_clean, corpus_node.id,
            page_level, section_level, atom_level,
            counter, raw_pos, clean_pos,
        ):
            yield item
    else:
        for sr in record.get("search_results", []):
            snippet = sr.get("page_snippet", "").strip()
            if not snippet:
                continue
            page_name = sr.get("page_name", "")[:200]
            page_clean, _ = _clean(page_name)
            page_value = f"{corpus_clean} | {page_clean}"
            page_node = _make_node(counter, page_level, page_value, raw_pos, len(page_name), clean_pos, len(page_clean), corpus_node.id)
            counter += 1
            raw_pos += len(page_name)
            clean_pos += len(page_clean)
            yield page_node, False

            clean_snippet, _ = _clean(snippet)
            atom_value = f"{page_value} | {clean_snippet}"
            atom_node = _make_node(counter, atom_level, atom_value, raw_pos, len(snippet), clean_pos, len(clean_snippet), page_node.id)
            counter += 1
            raw_pos += len(snippet)
            clean_pos += len(clean_snippet)
            yield atom_node, True


async def _normalize_crag_markdown(
    groundingdata: dict[str, Any],
    corpus_value: str,
    corpus_id: int,
    page_level: str,
    section_level: str,
    atom_level: str,
    counter: int,
    raw_pos: int,
    clean_pos: int,
) -> AsyncGenerator[tuple[Node, bool], None]:
    filename = groundingdata.get("filename", "")[:200]
    page_clean, _ = _clean(filename)
    page_value = f"{corpus_value} | {page_clean}"
    page_node = _make_node(counter, page_level, page_value, raw_pos, len(filename), clean_pos, len(page_clean), corpus_id)
    counter += 1
    raw_pos += len(filename)
    clean_pos += len(page_clean)
    yield page_node, False

    current_section_id = page_node.id
    current_section_value = page_value
    buffer: list[str] = []
    buffer_tokens = 0

    async def flush_md(buf: list[str]) -> AsyncGenerator[tuple[Node, bool], None]:
        nonlocal counter, raw_pos, clean_pos
        if not buf:
            return
        combined = " ".join(buf)
        clean_text, _ = _clean(combined)
        value = f"{current_section_value} | {clean_text}"
        node = _make_node(counter, atom_level, value, raw_pos, len(combined), clean_pos, len(clean_text), current_section_id)
        counter += 1
        raw_pos += len(combined)
        clean_pos += len(clean_text)
        yield node, True
        buf.clear()

    for line in groundingdata.get("markdown", "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            async for item in flush_md(buffer):
                yield item
            buffer_tokens = 0
            heading = stripped.lstrip("#").strip()
            sec_clean, _ = _clean(heading)
            current_section_value = f"{page_value} | {sec_clean}"
            sec_node = _make_node(counter, section_level, current_section_value, raw_pos, len(heading), clean_pos, len(sec_clean), page_node.id)
            current_section_id = sec_node.id
            counter += 1
            raw_pos += len(heading)
            clean_pos += len(sec_clean)
            yield sec_node, False
        else:
            tokens = _token_count(stripped)
            if buffer_tokens + tokens > ATOM_TOKEN_LIMIT and buffer:
                async for item in flush_md(buffer):
                    yield item
                buffer_tokens = 0
            buffer.append(stripped)
            buffer_tokens += tokens

    async for item in flush_md(buffer):
        yield item
