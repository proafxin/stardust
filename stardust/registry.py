import gc
from functools import cache

import cupy
import spacy
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from spacy.language import Language

from stardust.config import EMBEDDING_MODEL, RERANKER_MODEL, SPACY_MODEL


@cache
def nlp() -> Language:
    spacy.prefer_gpu()
    return spacy.load(SPACY_MODEL, disable=["lemmatizer", "ner"])


def unload_nlp() -> None:
    if not nlp.cache_info().currsize:
        return
    model = nlp()
    nlp.cache_clear()
    del model
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()


@cache
def embedder() -> SentenceTransformer:
    model = SentenceTransformer(EMBEDDING_MODEL, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = 512
    model.eval()
    return model


def unload_embedder() -> None:
    if not embedder.cache_info().currsize:
        return
    model = embedder()
    model.cpu()
    del model
    embedder.cache_clear()
    torch.cuda.empty_cache()


@cache
def reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL, device="cuda")


def unload_reranker() -> None:
    if not reranker.cache_info().currsize:
        return
    model = reranker()
    model.model.cpu()
    del model
    reranker.cache_clear()
    torch.cuda.empty_cache()
