from enum import StrEnum

from pydantic import BaseModel


class Modality(StrEnum):
    TEXT = "text"
    TABULAR = "tabular"


class SpanOffset(BaseModel):
    start: int
    end: int


class TokenAttributes(BaseModel):
    text: str
    pos_: str
    dep_: str
    morph: dict[str, str]
    ent_type_: str
    ent_iob_: str
    offset: SpanOffset


class PronounResolution(BaseModel):
    offset: SpanOffset
    token: str
    referent: str


class DisambiguationMetadata(BaseModel):
    pronoun_map: list[PronounResolution]


class Node(BaseModel):
    id: int
    node_type: str
    modality: Modality
    value: str
    raw_offset: SpanOffset
    clean_offset: SpanOffset
    parent_id: int | None
    terminal: bool
    nlp_attributes: list[TokenAttributes]
    disambiguation: DisambiguationMetadata | None


class AtomIndex(BaseModel):
    nodes: dict[int, Node]
    children: dict[int, list[int]]
    atoms: list[int]
    embeddings: dict[int, list[float]]
