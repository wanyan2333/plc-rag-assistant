"""Citation helpers: parse [n] markers and pick a supporting snippet from a chunk."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_WORD_RE = re.compile(r"[a-z0-9#\-]{3,}")


class Citation(BaseModel):
    doc_title: str
    page: int = Field(ge=1)
    snippet: str

    @field_validator("snippet")
    @classmethod
    def _clean_snippet(cls, value: str) -> str:
        # Models sometimes return escaped newlines; keep snippets single-line and readable.
        return " ".join(value.replace("\\n", " ").split())


class NumberedCitation(Citation):
    n: int
    page_end: int
    chunk_id: str
    section_heading: str = ""


def cited_numbers(text: str) -> list[int]:
    seen: list[int] = []
    for match in CITE_RE.finditer(text):
        for n in match.group(1).split(","):
            value = int(n)
            if value not in seen:
                seen.append(value)
    return seen


def best_snippet(chunk_text: str, reference: str, max_chars: int = 280) -> str:
    """The sentence (with a neighbour) in `chunk_text` that best overlaps `reference`."""
    body = chunk_text.split("\n", 1)[-1] if chunk_text.startswith("[") else chunk_text
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", body) if s.strip()]
    if not sentences:
        return chunk_text[:max_chars]
    ref_words = set(_WORD_RE.findall(reference.lower()))
    scores = [len(ref_words & set(_WORD_RE.findall(s.lower()))) for s in sentences]
    i = max(range(len(sentences)), key=lambda k: (scores[k], -k))
    snippet = sentences[i]
    if i + 1 < len(sentences) and len(snippet) < max_chars // 2:
        snippet += " " + sentences[i + 1]
    return snippet if len(snippet) <= max_chars else snippet[: max_chars - 1].rstrip() + "…"
