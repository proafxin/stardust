from functools import cache

import spacy
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from spacy.language import Language

from stardust.config import EMBEDDING_MODEL, RERANKER_MODEL, SPACY_MODEL


@cache
def nlp() -> Language:
    spacy.prefer_gpu()
    return spacy.load(SPACY_MODEL)


def unload_nlp() -> None:
    if nlp.cache_info().currsize:
        nlp.cache_clear()
        torch.cuda.empty_cache()


@cache
def embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, device="cuda")


@cache
def reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL, device="cuda")
