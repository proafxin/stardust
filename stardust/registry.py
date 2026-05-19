import gc
from functools import cache

import cupy
import spacy
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from spacy.language import Language
from transformers import AutoTokenizer

from stardust.config import EMBEDDING_MODEL, LLM_TOKENIZER_MODEL, RERANKER_MODEL, SPACY_MODEL


@cache
def nlp() -> Language:
    spacy.prefer_gpu()
    return spacy.load(SPACY_MODEL)


def unload_nlp() -> None:
    if nlp.cache_info().currsize:
        nlp.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()


@cache
def llm_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(LLM_TOKENIZER_MODEL)


def unload_llm_tokenizer() -> None:
    if llm_tokenizer.cache_info().currsize:
        llm_tokenizer.cache_clear()
        gc.collect()


@cache
def embedder() -> SentenceTransformer:
    model = SentenceTransformer(EMBEDDING_MODEL, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = 512
    model.eval()
    return model


def unload_embedder() -> None:
    if embedder.cache_info().currsize:
        model = embedder()
        model.cpu()
        del model
        embedder.cache_clear()
        torch.cuda.empty_cache()


@cache
def reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL, device="cuda")
