"""Keyword retrieval with BM25 (rank_bm25).

The tokenizer keeps fault-code-like compounds intact ("16#8085", "e-101", "s7-1200") and
also emits their parts plus a dash-free form, so "E101", "E-101" and "e-101" all match.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.ingest.chunker import Chunk

COMPOUND_RE = re.compile(r"[a-z0-9]+(?:[#\-_.][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it its of on or the this to "
    "what when where which why with you your my me".split()
)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in COMPOUND_RE.finditer(text.lower()):
        token = match.group(0).strip(".-_")
        if not token:
            continue
        parts = [p for p in re.split(r"[#\-_.]", token) if p]
        if len(parts) > 1:
            tokens.append(token)  # exact compound, e.g. "16#8085"
            tokens.append(re.sub(r"[\-_.]", "", token))  # dash-free form, e.g. "e101"
            tokens.extend(p for p in parts if p not in STOPWORDS)
        elif token not in STOPWORDS:
            tokens.append(token)
    return tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.ids = [c.chunk_id for c in chunks]
        corpus = [tokenize(f"{c.section_heading}\n{c.text}") for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [(self.ids[i], float(scores[i])) for i in ranked[:k] if scores[i] > 0]
