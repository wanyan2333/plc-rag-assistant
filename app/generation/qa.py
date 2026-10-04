"""Mode A: grounded question answering with [n] citations."""

from __future__ import annotations

import re
import time

from pydantic import BaseModel

from app.config import RetrievalMode
from app.generation.citations import NumberedCitation, best_snippet, cited_numbers
from app.generation.llm import LLM, Message, Usage
from app.generation.prompts import NOT_FOUND, QA_SYSTEM, qa_user_prompt
from app.retrieval.hybrid import HybridRetriever


class QAResult(BaseModel):
    question: str
    answer: str
    citations: list[NumberedCitation]
    found_in_manuals: bool
    retrieval_mode: str
    retrieved_chunk_ids: list[str]
    usage: dict
    model: str
    latency_ms: int


def is_not_found(answer: str) -> bool:
    return NOT_FOUND.lower().rstrip(".") in answer.lower()


def answer_question(
    question: str,
    retriever: HybridRetriever,
    llm: LLM,
    k: int = 6,
    mode: RetrievalMode = "hybrid",
) -> QAResult:
    start = time.perf_counter()
    results = retriever.search(question, k=k, mode=mode)
    usage = Usage()
    if not results:
        answer, model = NOT_FOUND, llm.model
    else:
        response = llm.chat(QA_SYSTEM, [Message("user", qa_user_prompt(question, results))])
        usage += response.usage
        answer, model = response.text.strip(), response.model

    citations: list[NumberedCitation] = []
    for n in sorted(cited_numbers(answer)):
        if not 1 <= n <= len(results):
            continue  # ignore hallucinated source numbers
        chunk = results[n - 1].chunk
        # Use the sentences of the answer that cite [n] to choose the snippet.
        citing = " ".join(s for s in re.split(r"(?<=[.!?\]])\s+", answer) if n in cited_numbers(s)) or answer
        citations.append(
            NumberedCitation(
                n=n,
                doc_title=chunk.doc_title,
                page=chunk.page,
                page_end=chunk.page_end,
                chunk_id=chunk.chunk_id,
                section_heading=chunk.section_heading,
                snippet=best_snippet(chunk.text, citing),
            )
        )

    return QAResult(
        question=question,
        answer=answer,
        citations=citations,
        found_in_manuals=not is_not_found(answer),
        retrieval_mode=mode,
        retrieved_chunk_ids=[r.chunk.chunk_id for r in results],
        usage=usage.to_dict(),
        model=model,
        latency_ms=int((time.perf_counter() - start) * 1000),
    )
