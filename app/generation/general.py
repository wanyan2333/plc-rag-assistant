"""General-knowledge fallback for questions the manuals do not cover.

The grounded answer stays the primary output; this answer is returned in a separate field
and must always be shown to users as "not from the manuals".
"""

from __future__ import annotations

from app.generation.llm import LLM, LLMError, Message, Usage
from app.generation.prompts import GENERAL_SYSTEM


def general_answer(question: str, llm: LLM) -> tuple[str | None, Usage]:
    """Return (answer, usage). Errors are swallowed: the grounded result is still valid without it."""
    try:
        reply = llm.chat(GENERAL_SYSTEM.format(model=llm.model), [Message("user", question)])
    except LLMError:
        return None, Usage()
    return (reply.text.strip() or None), reply.usage
