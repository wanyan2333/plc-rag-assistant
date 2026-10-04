"""Embedding backends behind a small interface so the model can be swapped."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from app.config import Settings


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local sentence-transformers model (default: BAAI/bge-small-en-v1.5, 384-d)."""

    # BGE models are trained with this instruction prefix for retrieval queries.
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self._batch_size = batch_size
        self._query_prefix = self.QUERY_INSTRUCTION if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(self._query_prefix + text, normalize_embeddings=True)
        return vector.tolist()


class HashEmbedder:
    """Deterministic feature-hashing embedder (bag of words + character trigrams).

    No model download and no network — used by the test-suite and for quick offline runs.
    It only captures lexical overlap, so do not use it to judge semantic retrieval quality.
    """

    def __init__(self, dim: int = 384):
        self.name = f"hash-{dim}"
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9#]+", text.lower())
        grams = [w[i : i + 3] for w in words for i in range(max(1, len(w) - 2))]
        return words + grams

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feature in self._features(text):
            digest = hashlib.md5(feature.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            vec[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)
