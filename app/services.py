"""Wires settings into ready-to-use components (store, retriever, LLM).

Built lazily so importing the API does not load the embedding model.
"""

from __future__ import annotations

from functools import cached_property

from app.config import Settings, get_settings
from app.retrieval.embeddings import Embedder, build_embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.store import ChunkStore
from app.retrieval.vector import VectorIndex


class Services:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None, llm=None):
        self.settings = settings or get_settings()
        self._embedder = embedder
        self._llm = llm

    @cached_property
    def store(self) -> ChunkStore:
        store = ChunkStore.load(self.settings.index_path)
        if not store.chunks:
            raise RuntimeError(
                f"No index found in {self.settings.index_path}. Run `python -m app.ingest --rebuild` first."
            )
        return store

    @cached_property
    def embedder(self) -> Embedder:
        return self._embedder or build_embedder(self.settings)

    @cached_property
    def retriever(self) -> HybridRetriever:
        vector = VectorIndex(self.settings.index_path, self.embedder)
        return HybridRetriever(self.store, vector, rrf_k=self.settings.rrf_k)

    @cached_property
    def llm(self):
        if self._llm is not None:
            return self._llm
        from app.generation.llm import build_llm

        return build_llm(self.settings)
