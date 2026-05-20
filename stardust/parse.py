import html
import re
import unicodedata
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

from stardust.tree.atom import Modality, Node


def clean_value(value: str) -> str:
    raw = value.rsplit(" | ", 1)[-1] if " | " in value else value
    raw = re.sub(r"\b[0-9a-f]{32}\b", "", raw)
    raw = re.sub(r"https?%3A%2F%2F\S+", "", raw)
    raw = re.sub(r"https?://\S+", "", raw)
    return re.sub(r"\s+", " ", raw).strip()


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
    node_type: str,
    value: str,
    parent_index: int | None,
    modality: Modality = Modality.TEXT,
    terminal: bool = False,
) -> Node:
    return Node(
        parent_index=parent_index,
        node_type=node_type,
        modality=modality,
        value=value,
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


def _clean_filename(filename: str) -> str:
    s = re.sub(r"^[0-9a-f]{32}-", "", filename)
    s = unquote(s)
    m = re.search(r"https?://[^/]+(/.*)", s)
    if m:
        s = m.group(1)
    s = re.sub(r"[/_-]+", " ", s)
    s = re.sub(r"\.[a-z]{2,4}$", "", s)
    return re.sub(r"\s+", " ", s).strip()


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
        sep_idx = next((j for j, row in enumerate(block) if re.match(r"\s*\|[\s\-:|]+\|", row)), None)
        if sep_idx is None:
            continue
        header_line = block[sep_idx - 1] if sep_idx > 0 else block[0]
        col_names = [_cell_text(c) for c in header_line.strip("|\n ").split("|")]
        col_names = [c for c in col_names if c]
        if not col_names:
            continue
        data_rows = [row for j, row in enumerate(block) if j != sep_idx and (sep_idx == 0 or j != sep_idx - 1)]
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
