import re
import unicodedata
from typing import Any

from pydantic import BaseModel

from stardust.config import ATOM_TOKEN_LIMIT
from stardust.index import OffsetMap
from stardust.tree.atom import AtomIndex, Modality, Node, NodeType, SpanOffset


class Section(BaseModel):
    heading: str | None
    paragraphs: list[str]
    subsections: list["Section"] = []


Section.model_rebuild()


class Document(BaseModel):
    title: str
    sections: list[Section]
    metadata: dict[str, str | int | float | bool | None] = {}


def normalize_hotpotqa(record: dict[str, Any]) -> list[Document]:
    supporting: set[tuple[str, int]] = {
        (t, s)
        for t, s in zip(
            record["supporting_facts"]["title"],
            record["supporting_facts"]["sent_id"],
            strict=False,
        )
    }
    docs: list[Document] = []
    for title, sentences in zip(record["context"]["title"], record["context"]["sentences"], strict=False):
        docs.append(
            Document(
                title=title,
                sections=[Section(heading=None, paragraphs=[s for s in sentences if s.strip()])],
                metadata={
                    "question": record.get("question", ""),
                    "answer": record.get("answer", ""),
                    "is_supporting": any((title, i) in supporting for i in range(len(sentences))),
                },
            )
        )
    return docs


def normalize_qasper(record: dict[str, Any]) -> list[Document]:
    context: str = record.get("context", "")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", context) if p.strip()]
    return [
        Document(
            title="qasper_paper",
            sections=[Section(heading=None, paragraphs=paragraphs)],
            metadata={
                "questions": str(record.get("questions", [])),
                "answers": str(record.get("answers", [])),
            },
        )
    ]


def normalize_crag(qapair: dict[str, Any], groundingdata: dict[str, Any] | None = None) -> list[Document]:
    base_meta: dict[str, str | int | float | bool | None] = {
        "query": qapair.get("query", ""),
        "answer": qapair.get("answer", ""),
        "query_time": qapair.get("query_time", ""),
        "domain": qapair.get("domain", ""),
        "question_type": qapair.get("question_type", ""),
        "static_or_dynamic": qapair.get("static_or_dynamic", ""),
    }
    if groundingdata:
        return [_parse_markdown_doc(groundingdata, base_meta)]
    docs: list[Document] = []
    for sr in qapair.get("search_results", []):
        snippet = sr.get("page_snippet", "").strip()
        if not snippet:
            continue
        docs.append(
            Document(
                title=sr.get("page_name", "")[:200],
                sections=[Section(heading=None, paragraphs=[snippet])],
                metadata={
                    **base_meta,
                    "page_url": sr.get("page_url", ""),
                    "page_last_modified": sr.get("page_last_modified", ""),
                    "_hash": sr.get("_hash", ""),
                },
            )
        )
    return docs


def _parse_markdown_doc(
    groundingdata: dict[str, Any],
    base_meta: dict[str, str | int | float | bool | None]
) -> Document:
    markdown: str = groundingdata.get("markdown", "")
    filename: str = groundingdata.get("filename", "")
    sections: list[Section] = []
    current_section: Section | None = None
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if current_section is not None:
                sections.append(current_section)
            current_section = Section(heading=stripped.lstrip("#").strip(), paragraphs=[])
        else:
            if current_section is None:
                current_section = Section(heading=None, paragraphs=[])
            current_section.paragraphs.append(stripped)
    if current_section is not None:
        sections.append(current_section)
    return Document(title=filename[:200], sections=sections, metadata=base_meta)


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


class _IndexBuilder:
    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.children: dict[int, list[int]] = {}
        self.atoms: list[int] = []
        self._counter = 0
        self._raw_pos = 0
        self._clean_pos = 0

    def add(
        self,
        node_type: NodeType,
        value: str,
        raw_len: int,
        clean_len: int,
        parent_id: int | None,
        metadata: dict[str, str | int | float | bool | None],
    ) -> int:
        nid = self._counter
        self._counter += 1
        self.nodes[nid] = Node(
            id=nid,
            node_type=node_type,
            modality=Modality.TEXT,
            value=value,
            raw_offset=SpanOffset(start=self._raw_pos, end=self._raw_pos + raw_len),
            clean_offset=SpanOffset(start=self._clean_pos, end=self._clean_pos + clean_len),
            parent_id=parent_id,
            nlp_attributes=[],
            disambiguation=None,
            metadata=metadata,
        )
        self.children[nid] = []
        if parent_id is not None:
            self.children[parent_id].append(nid)
        self._raw_pos += raw_len
        self._clean_pos += clean_len
        return nid

    def add_atoms(
        self,
        paragraphs: list[str],
        ancestor_value: str,
        parent_id: int,
        metadata: dict[str, str | int | float | bool | None],
    ) -> None:
        buffer: list[str] = []
        buffer_tokens = 0

        def flush() -> None:
            if not buffer:
                return
            combined = " ".join(buffer)
            clean_text, _ = _clean(combined)
            value = f"{ancestor_value} | {clean_text}" if ancestor_value else clean_text
            nid = self.add(NodeType.PARAGRAPH, value, len(combined), len(clean_text), parent_id, metadata)
            self.atoms.append(nid)
            buffer.clear()

        for para in paragraphs:
            tokens = _token_count(para)
            if buffer_tokens + tokens > ATOM_TOKEN_LIMIT:
                flush()
                buffer_tokens = 0
            buffer.append(para)
            buffer_tokens += tokens
        flush()


def build_atom_index(docs: list[Document]) -> AtomIndex:
    b = _IndexBuilder()
    for doc in docs:
        doc_clean, _ = _clean(doc.title)
        doc_id = b.add(NodeType.DOCUMENT, doc_clean, len(doc.title), len(doc_clean), None, doc.metadata)
        for section in doc.sections:
            sec_clean, _ = _clean(section.heading or "")
            sec_value = " | ".join(filter(None, [doc_clean, section.heading]))
            sec_id = b.add(
                NodeType.SECTION, sec_value, len(section.heading or ""), len(sec_clean), doc_id, doc.metadata
            )
            b.add_atoms(section.paragraphs, sec_value, sec_id, doc.metadata)
            for subsection in section.subsections:
                sub_clean, _ = _clean(subsection.heading or "")
                sub_value = " | ".join(filter(None, [sec_value, subsection.heading]))
                sub_id = b.add(
                    NodeType.SUBSECTION, sub_value, len(subsection.heading or ""), len(sub_clean), sec_id, doc.metadata
                )
                b.add_atoms(subsection.paragraphs, sub_value, sub_id, doc.metadata)
    return AtomIndex(nodes=b.nodes, children=b.children, atoms=b.atoms, embeddings={})
