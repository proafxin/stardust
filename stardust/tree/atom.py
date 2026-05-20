from enum import StrEnum

from pydantic import BaseModel


class Modality(StrEnum):
    TEXT = "text"
    TABULAR = "tabular"


class Node(BaseModel):
    parent_index: int | None
    node_type: str
    modality: Modality
    value: str
    terminal: bool
