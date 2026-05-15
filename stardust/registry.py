from __future__ import annotations

import spacy
from sentence_transformers import CrossEncoder, SentenceTransformer
from spacy.language import Language

from config import EMBEDDING_MODEL, RERANKER_MODEL, SPACY_MODEL

_nlp: Language | None = None
_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def get_nlp() -> Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(SPACY_MODEL)
    return _nlp


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def warm_up() -> None:
    get_nlp()
    get_embedder()
    get_reranker()
