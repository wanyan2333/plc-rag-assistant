"""Mode B: tool-calling troubleshooting agent with schema-validated JSON output.

Loop: the model calls lookup_fault_code / search_manuals (at most `max_tool_rounds` rounds),
then submits its plan through submit_troubleshooting_result. The submission is validated
with Pydantic plus a grounding check (every citation must point at a page the tools
actually returned). On failure the error is fed back to the model once; a second failure
returns an error structure instead of raising.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.config import RetrievalMode, Settings
from app.generation.citations import Citation
from app.generation.general import general_answer
from app.generation.llm import LLM, LLMError, Message, ToolCall, ToolResult, ToolSpec, Usage
from app.generation.prompts import TROUBLESHOOT_SYSTEM
from app.generation.tools import LOOKUP_FAULT_CODE, SEARCH_MANUALS, ToolExecutor
from app.ingest.chunker import Chunk

MAX_SUBMIT_ATTEMPTS = 2  # first try + one retry with the validation error


class TroubleshootResult(BaseModel):
    fault_code: str | None = Field(description="Fault/alarm/event code from the report, or null.")
    summary: str = Field(min_length=1, description="One or two sentences: what the fault means.")
    likely_causes: list[str] = Field(description="Most likely causes, most probable first.")
    diagnostic_steps: list[str] = Field(description="Ordered, concrete diagnostic and corrective steps.")
    safety_notes: list[str] = Field(description="Safety precautions (lockout/tagout, DC bus discharge, ...).")
    citations: list[Citation] = Field(description="Manual references (doc_title, page, verbatim snippet).")
    confidence: Literal["high", "medium", "low"]
    found_in_manuals: bool

    @model_validator(mode="after")
    def _consistent(self) -> "TroubleshootResult":
        if self.found_in_manuals:
            if not self.diagnostic_steps:
                raise ValueError("found_in_manuals is true but diagnostic_steps is empty")
            if not self.citations:
                raise ValueError("found_in_manuals is true but citations is empty")
        return self


SUBMIT_RESULT = ToolSpec(
    name="submit_troubleshooting_result",
    description="Submit the final troubleshooting plan. Call exactly once, after researching with the other tools.",
    parameters=TroubleshootResult.model_json_schema(),
)


class TroubleshootResponse(BaseModel):
    ok: bool
    result: TroubleshootResult | None = None
    error: str | None = None
    general_answer: str | None = None  # general-knowledge answer when the manuals had nothing
    question: str
    attempts: int = 0
    tool_rounds: int = 0
    tool_trace: list[dict] = []
    retrieved_chunk_ids: list[str] = []
    retrieval_mode: str = "hybrid"
    usage: dict = {}
    model: str = ""
    latency_ms: int = 0
    raw_output: dict | None = None


def _norm(title: str) -> str:
    return " ".join(title.casefold().split())


def check_grounding(result: TroubleshootResult, seen: dict[str, Chunk]) -> list[str]:
    """Return problems for citations that do not match a document page returned by the tools."""
    allowed = {(_norm(c.doc_title), p) for c in seen.values() for p in c.pages}
    problems = []
    for i, cit in enumerate(result.citations):
        if (_norm(cit.doc_title), cit.page) not in allowed:
            problems.append(f"citations[{i}] ({cit.doc_title!r}, page {cit.page}) was not returned by any tool call")
    if problems:
        sources = sorted({f"{c.doc_title!r} p.{c.page}" + (f"-{c.page_end}" if c.page_end != c.page else "") for c in seen.values()})
        problems.append("Allowed sources: " + ("; ".join(sources) if sources else "none - call a research tool first"))
    return problems


def validate_submission(args: dict, seen: dict[str, Chunk]) -> tuple[TroubleshootResult | None, str | None]:
    try:
        result = TroubleshootResult.model_validate(args)
    except ValidationError as exc:
        errors = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}" for e in exc.errors())
        return None, f"Schema validation failed: {errors}"
    problems = check_grounding(result, seen)
    if problems:
        return None, "Grounding check failed: " + " | ".join(problems)
    return result, None


def troubleshoot(
    question: str,
    retriever,
    llm: LLM,
    settings: Settings,
    mode: RetrievalMode = "hybrid",
    allow_general: bool = False,
) -> TroubleshootResponse:
    response = _troubleshoot(question, retriever, llm, settings, mode)
    if allow_general and response.ok and not response.result.found_in_manuals:
        start = time.perf_counter()
        response.general_answer, extra = general_answer(question, llm)
        response.usage = {
            "input_tokens": response.usage.get("input_tokens", 0) + extra.input_tokens,
            "output_tokens": response.usage.get("output_tokens", 0) + extra.output_tokens,
        }
        response.latency_ms += int((time.perf_counter() - start) * 1000)
    return response


def _troubleshoot(
    question: str,
    retriever,
    llm: LLM,
    settings: Settings,
    mode: RetrievalMode,
) -> TroubleshootResponse:
    start = time.perf_counter()
    max_rounds = settings.max_tool_rounds
    system = TROUBLESHOOT_SYSTEM.format(max_rounds=max_rounds)
    executor = ToolExecutor(retriever, mode)
    research_tools = [LOOKUP_FAULT_CODE, SEARCH_MANUALS]
    messages: list[Message] = [Message("user", question)]
    usage = Usage()
    rounds = attempts = 0
    last_args: dict | None = None
    last_error = "The model did not submit a result."
    model = llm.model

    def response(ok: bool, result: TroubleshootResult | None = None, error: str | None = None) -> TroubleshootResponse:
        return TroubleshootResponse(
            ok=ok,
            result=result,
            error=error,
            question=question,
            attempts=attempts,
            tool_rounds=rounds,
            tool_trace=executor.trace,
            retrieved_chunk_ids=list(executor.seen),
            retrieval_mode=mode,
            usage=usage.to_dict(),
            model=model,
            latency_ms=int((time.perf_counter() - start) * 1000),
            raw_output=last_args,
        )

    # Hard cap on model calls: research rounds + final submission + one retry.
    for _ in range(max_rounds + MAX_SUBMIT_ATTEMPTS + 1):
        tools = [*research_tools, SUBMIT_RESULT] if rounds < max_rounds else [SUBMIT_RESULT]
        try:
            reply = llm.chat(system, messages, tools)
        except LLMError as exc:
            return response(False, error=f"LLM call failed: {exc}")
        usage += reply.usage
        model = reply.model
        messages.append(reply.as_message())

        if not reply.tool_calls:
            attempts += 1
            last_error = "No tool call: you must call submit_troubleshooting_result with the final plan."
            if attempts >= MAX_SUBMIT_ATTEMPTS:
                return response(False, error=last_error)
            messages.append(Message("user", last_error))
            continue

        results: list[ToolResult] = []
        did_research = False
        for call in reply.tool_calls:
            if call.name == SUBMIT_RESULT.name:
                attempts += 1
                last_args = call.args
                result, error = validate_submission(call.args, executor.seen)
                if result is not None:
                    return response(True, result)
                last_error = error or "invalid submission"
                if attempts >= MAX_SUBMIT_ATTEMPTS:
                    return response(False, error=f"Output failed validation twice. Last error: {last_error}")
                results.append(ToolResult(call.id, call.name, f"{last_error}\nFix the problems and call submit_troubleshooting_result again.", is_error=True))
            elif rounds >= max_rounds:
                results.append(ToolResult(call.id, call.name, "Tool budget exhausted.", is_error=True))
            else:
                did_research = True
                results.append(executor.run(call))
        if did_research:
            rounds += 1
        nudge = ""
        if rounds >= max_rounds:
            nudge = "Research budget exhausted. Call submit_troubleshooting_result now using the information gathered."
        messages.append(Message("user", nudge, tool_results=results))

    return response(False, error=f"No valid result after the maximum number of steps. Last error: {last_error}")


def tool_call(name: str, args: dict, call_id: str = "call") -> ToolCall:
    """Small helper used by tests and scripted demos."""
    return ToolCall(call_id, name, args)
