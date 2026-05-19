from enum import StrEnum

from pydantic import BaseModel


class Modality(StrEnum):
    TEXT = "text"
    TABULAR = "tabular"


class SpanOffset(BaseModel):
    start: int
    end: int


class Node(BaseModel):
    parent_index: int | None
    node_type: str
    modality: Modality
    value: str
    raw_offset: SpanOffset
    clean_offset: SpanOffset
    terminal: bool
