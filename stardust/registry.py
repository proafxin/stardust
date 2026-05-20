import gc
from functools import cache

import torch
from fastcoref import LingMessCoref
from sentence_transformers import CrossEncoder, SentenceTransformer

from stardust.config import COREF_MODEL, EMBEDDING_MODEL, RERANKER_MODEL


@cache
def coref() -> LingMessCoref:
    return LingMessCoref(model_name_or_path=COREF_MODEL, device="cuda")


def unload_coref() -> None:
    if not coref.cache_info().currsize:
        return
    model = coref()
    coref.cache_clear()
    del model
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
