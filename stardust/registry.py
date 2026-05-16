from functools import cache

import spacy
from sentence_transformers import CrossEncoder, SentenceTransformer
from spacy.language import Language

from stardust.config import EMBEDDING_MODEL, RERANKER_MODEL, SPACY_MODEL


@cache
def nlp() -> Language:
    return spacy.load(SPACY_MODEL)


@cache
def embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@cache
def reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)
