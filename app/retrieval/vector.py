"""Dense retrieval over a persistent Chroma collection."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.ingest.chunker import Chunk
from app.retrieval.embeddings import Embedder

COLLECTION = "plc_chunks"


def _client(index_dir: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(index_dir / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
    )


class VectorIndex:
    def __init__(self, index_dir: Path, embedder: Embedder):
        self.embedder = embedder
        self._client = _client(index_dir)
        self._collection = self._client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def reset(self) -> None:
        self._client.delete_collection(COLLECTION)
        self._collection = self._client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def count(self) -> int:
        return self._collection.count()

    def upsert(self, chunks: list[Chunk], batch_size: int = 128) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=self.embedder.embed_documents([c.text for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )

    def delete_except(self, keep_ids: set[str]) -> int:
        """Remove vectors whose chunk no longer exists (keeps re-ingestion idempotent)."""
        existing = set(self._collection.get(include=[])["ids"])
        stale = sorted(existing - keep_ids)
        if stale:
            self._collection.delete(ids=stale)
        return len(stale)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        n = min(k, self.count())
        if n == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self.embedder.embed_query(query)], n_results=n, include=["distances"]
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [(cid, 1.0 - dist) for cid, dist in zip(ids, distances)]
