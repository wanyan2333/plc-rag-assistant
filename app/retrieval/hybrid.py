"""Hybrid retrieval: BM25 + dense vectors fused with Reciprocal Rank Fusion (RRF).

Modes:
  vector  dense retrieval only (baseline)
  bm25    keyword retrieval only (baseline)
  hybrid  RRF(vector, bm25) + exact fault-code routing: if the query mentions a code that
          exists in an ingested fault table, that row's chunk is returned first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import RetrievalMode
from app.fault_codes import find_fault_codes
from app.ingest.chunker import Chunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.store import ChunkStore
from app.retrieval.vector import VectorIndex


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)), rank from 1.

    Ties are broken by the best rank a document achieved in any list, then by id.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    return sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
    sources: dict[str, int] = field(default_factory=dict)  # retriever -> rank in that list
    exact_fault_match: bool = False

    def to_dict(self) -> dict:
        return {
            **self.chunk.to_dict(),
            "score": round(self.score, 6),
            "rank": self.rank,
            "sources": self.sources,
            "exact_fault_match": self.exact_fault_match,
        }


class HybridRetriever:
    def __init__(
        self,
        store: ChunkStore,
        vector: VectorIndex,
        bm25: BM25Index | None = None,
        rrf_k: int = 60,
        candidates: int = 30,
    ):
        self.store = store
        self.vector = vector
        self.bm25 = bm25 or BM25Index(store.chunks)
        self.rrf_k = rrf_k
        self.candidates = candidates

    def exact_fault_matches(self, query: str) -> list[Chunk]:
        hits: list[Chunk] = []
        for code in find_fault_codes(query):
            hits.extend(c for c in self.store.lookup_fault_code(code) if c not in hits)
        return hits

    def search(self, query: str, k: int = 6, mode: RetrievalMode = "hybrid") -> list[RetrievedChunk]:
        n = max(k, self.candidates)
        if mode == "vector":
            return self._wrap(self.vector.search(query, k), "vector")
        if mode == "bm25":
            return self._wrap(self.bm25.search(query, k), "bm25")
        if mode != "hybrid":
            raise ValueError(f"unknown retrieval mode {mode!r}")

        vector_ranked = [cid for cid, _ in self.vector.search(query, n)]
        bm25_ranked = [cid for cid, _ in self.bm25.search(query, n)]
        fused = rrf_fuse([vector_ranked, bm25_ranked], k=self.rrf_k)

        source_ranks = {"vector": vector_ranked, "bm25": bm25_ranked}
        results: list[RetrievedChunk] = []
        exact = self.exact_fault_matches(query)
        exact_ids = {c.chunk_id for c in exact}
        top_score = fused[0][1] if fused else 1.0
        for chunk in exact:
            results.append(
                RetrievedChunk(chunk, top_score + 1.0, 0, self._ranks(chunk.chunk_id, source_ranks), True)
            )
        for cid, score in fused:
            if cid in exact_ids:
                continue
            chunk = self.store.get(cid)
            if chunk is not None:
                results.append(RetrievedChunk(chunk, score, 0, self._ranks(cid, source_ranks)))
        results = results[:k]
        for i, r in enumerate(results, start=1):
            r.rank = i
        return results

    @staticmethod
    def _ranks(cid: str, source_ranks: dict[str, list[str]]) -> dict[str, int]:
        return {name: ranking.index(cid) + 1 for name, ranking in source_ranks.items() if cid in ranking}

    def _wrap(self, ranked: list[tuple[str, float]], source: str) -> list[RetrievedChunk]:
        out = []
        for cid, score in ranked:
            chunk = self.store.get(cid)
            if chunk is not None:
                out.append(RetrievedChunk(chunk, score, len(out) + 1, {source: len(out) + 1}))
        return out
