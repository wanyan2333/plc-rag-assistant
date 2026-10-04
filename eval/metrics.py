"""Pure metric functions (unit-tested in tests/test_eval_metrics.py)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Ref:
    """A retrieved chunk or a citation: document plus the page span it covers."""

    doc_id: str
    page: int
    page_end: int | None = None

    def pages(self) -> set[int]:
        return set(range(self.page, (self.page_end or self.page) + 1))


def is_relevant(ref: Ref, expected_doc: str, expected_pages: list[int]) -> bool:
    return ref.doc_id == expected_doc and bool(ref.pages() & set(expected_pages))


def first_relevant_rank(refs: list[Ref], expected_doc: str, expected_pages: list[int]) -> int | None:
    for rank, ref in enumerate(refs, start=1):
        if is_relevant(ref, expected_doc, expected_pages):
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def citation_accuracy(citations: list[Ref], expected_doc: str, expected_pages: list[int]) -> float | None:
    """Share of citations that point at an expected page. None when nothing was cited."""
    if not citations:
        return None
    return sum(is_relevant(c, expected_doc, expected_pages) for c in citations) / len(citations)


def parse_judge(text: str) -> tuple[int | None, str]:
    """Extract {"score": n, "reason": ...} from a judge reply."""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            score = int(data.get("score"))
            if 1 <= score <= 5:
                return score, str(data.get("reason", ""))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    fallback = re.search(r"score\D{0,5}([1-5])", text, re.I)
    return (int(fallback.group(1)) if fallback else None), text.strip()[:200]


def mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None
