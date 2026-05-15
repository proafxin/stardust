import spacy
from sentence_transformers import CrossEncoder, SentenceTransformer
from spacy.language import Language

from stardust.config import EMBEDDING_MODEL, RERANKER_MODEL, SPACY_MODEL


class Registry:
    def __init__(self) -> None:
        self._nlp: Language | None = None
        self._embedder: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None

    def nlp(self) -> Language:
        if self._nlp is None:
            self._nlp = spacy.load(SPACY_MODEL)
        return self._nlp

    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

    def warm_up(self) -> None:
        self.nlp()
        self.embedder()
        self.reranker()


registry = Registry()
