"""Provider-neutral chat interface with tool calling.

Implementations:
  AnthropicLLM  Claude via the official `anthropic` SDK (primary provider)
  GeminiLLM     Google Gemini via `google-genai` (alternative provider)
  FakeLLM       deterministic offline model for tests and key-less demos

The conversation is kept in a small neutral format (`Message`, `ToolCall`, `ToolResult`).
Assistant turns also carry the provider-native content (`raw`) which is echoed back
unchanged on the next request — required to preserve Claude thinking blocks and Gemini
thought signatures in multi-step tool use.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from app.config import Settings

log = logging.getLogger(__name__)

# USD per million tokens (input, output). Override via PRICE_*_PER_MTOK for other models.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


# --------------------------------------------------------------------------- data types

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema (object)


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: str  # "user" | "assistant"
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    raw: Any = None  # provider-native assistant content, echoed back verbatim


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self

    def to_dict(self) -> dict:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: Usage
    model: str
    raw: Any = None

    def as_message(self) -> Message:
        return Message("assistant", self.text, list(self.tool_calls), raw=self.raw)


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    model: str

    def chat(
        self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None, max_tokens: int | None = None
    ) -> LLMResponse: ...


def estimate_cost(model: str, usage: Usage, settings: Settings | None = None) -> float | None:
    price = PRICES.get(model)
    if settings and settings.price_input_per_mtok is not None and settings.price_output_per_mtok is not None:
        price = (settings.price_input_per_mtok, settings.price_output_per_mtok)
    if price is None:
        return None
    return (usage.input_tokens * price[0] + usage.output_tokens * price[1]) / 1_000_000


def inline_refs(schema: dict) -> dict:
    """Inline `$ref`/`$defs` from a Pydantic JSON schema so every provider accepts it."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                target = defs[node["$ref"].split("/")[-1]]
                merged = {**resolve(target), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


# --------------------------------------------------------------------------- Anthropic

class AnthropicLLM:
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, settings: Settings, model: str | None = None):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
        self.model = model or settings.anthropic_model
        self.effort = settings.anthropic_effort
        self.max_tokens = settings.max_output_tokens
        self.fallbacks = settings.anthropic_fallbacks.strip()

    @staticmethod
    def _to_api(messages: list[Message]) -> list[dict]:
        out = []
        for m in messages:
            if m.role == "assistant":
                if m.raw is not None:
                    out.append({"role": "assistant", "content": m.raw})
                else:
                    content: list[dict] = [{"type": "text", "text": m.text}] if m.text else []
                    content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in m.tool_calls]
                    out.append({"role": "assistant", "content": content})
                continue
            content = [
                {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error}
                for r in m.tool_results
            ]
            if m.text:
                content.append({"type": "text", "text": m.text})
            out.append({"role": "user", "content": content})
        return out

    def chat(self, system, messages, tools=None, max_tokens=None) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "system": system,
            "messages": self._to_api(messages),
            "output_config": {"effort": self.effort},
        }
        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": inline_refs(t.parameters)} for t in tools
            ]
            params["tool_choice"] = {"type": "auto"}
        try:
            if self.fallbacks:
                response = self.client.beta.messages.create(
                    **params, betas=[self.FALLBACK_BETA], extra_body={"fallbacks": self.fallbacks}
                )
            else:
                response = self.client.messages.create(**params)
        except self._anthropic.BadRequestError as exc:
            if self.fallbacks and "fallback" in str(exc).lower():
                log.warning("Server-side fallback rejected (%s); retrying without it.", exc)
                self.fallbacks = ""
                return self.chat(system, messages, tools, max_tokens)
            raise LLMError(f"Anthropic request rejected: {exc}") from exc
        except self._anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("The model declined to answer this request.")
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"]
        usage = Usage(response.usage.input_tokens, response.usage.output_tokens)
        return LLMResponse(text, calls, response.stop_reason or "", usage, response.model, raw=response.content)


# --------------------------------------------------------------------------- Gemini

class GeminiLLM:
    def __init__(self, settings: Settings, model: str | None = None):
        from google import genai
        from google.genai import types

        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self._types = types
        self._errors = genai.errors
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = model or settings.gemini_model
        self.max_tokens = settings.max_output_tokens

    def _to_api(self, messages: list[Message]) -> list:
        types = self._types
        out = []
        for m in messages:
            if m.role == "assistant":
                if m.raw is not None:
                    out.append(m.raw)
                else:
                    parts = [types.Part(text=m.text)] if m.text else []
                    parts += [types.Part(function_call=types.FunctionCall(name=c.name, args=c.args)) for c in m.tool_calls]
                    out.append(types.Content(role="model", parts=parts))
                continue
            parts = [
                types.Part.from_function_response(
                    name=r.name, response={"error": r.content} if r.is_error else {"result": r.content}
                )
                for r in m.tool_results
            ]
            if m.text:
                parts.append(types.Part(text=m.text))
            out.append(types.Content(role="user", parts=parts))
        return out

    def chat(self, system, messages, tools=None, max_tokens=None) -> LLMResponse:
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens or self.max_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if tools:
            config.tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name, description=t.description, parameters_json_schema=inline_refs(t.parameters)
                        )
                        for t in tools
                    ]
                )
            ]

        response = None
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=self._to_api(messages), config=config
                )
                break
            except self._errors.APIError as exc:
                retryable = getattr(exc, "code", None) in (429, 500, 503)
                if not retryable or attempt == 2:
                    raise LLMError(f"Gemini API error: {exc}") from exc
                time.sleep(2 * (attempt + 1))

        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.content is None:
            raise LLMError(f"Gemini returned no content (prompt feedback: {response.prompt_feedback})")
        texts, calls = [], []
        for i, part in enumerate(candidate.content.parts or []):
            if part.function_call is not None:
                fc = part.function_call
                calls.append(ToolCall(fc.id or f"call_{len(messages)}_{i}", fc.name, dict(fc.args or {})))
            elif part.text and not part.thought:
                texts.append(part.text)
        meta = response.usage_metadata
        usage = Usage(
            getattr(meta, "prompt_token_count", 0) or 0,
            (getattr(meta, "candidates_token_count", 0) or 0) + (getattr(meta, "thoughts_token_count", 0) or 0),
        )
        stop = str(candidate.finish_reason.name if candidate.finish_reason else "")
        return LLMResponse("".join(texts), calls, "tool_use" if calls else stop, usage, self.model, raw=candidate.content)


# --------------------------------------------------------------------------- Fake

Policy = Callable[[str, list[Message], list[ToolSpec] | None], LLMResponse]


class FakeLLM:
    """Offline LLM. Either replays a script of responses or follows a heuristic policy.

    The default policy is good enough for demos without an API key: it answers from the
    best-matching source with a citation, refuses when nothing relevant was retrieved, and
    drives the troubleshoot tool loop deterministically.
    """

    model = "fake-llm"

    def __init__(self, script: list[LLMResponse | Callable[..., LLMResponse]] | None = None, policy: Policy | None = None):
        self.script = list(script or [])
        self.policy = policy or default_fake_policy
        self.calls: list[dict] = []

    def chat(self, system, messages, tools=None, max_tokens=None) -> LLMResponse:
        self.calls.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        if self.script:
            item = self.script.pop(0)
            return item(system, messages, tools) if callable(item) else item
        return self.policy(system, messages, tools)


def fake_response(text: str = "", tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    calls = tool_calls or []
    usage = Usage(input_tokens=100, output_tokens=max(1, len(text.split()) + 20 * len(calls)))
    return LLMResponse(text, calls, "tool_use" if calls else "end_turn", usage, FakeLLM.model)


_WORD_RE = re.compile(r"[a-z0-9#\-]{3,}")
_FAKE_STOP = frozenset("what does the and for with when how should which are can from this that mean means".split())


def _keywords(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _FAKE_STOP}


def default_fake_policy(system: str, messages: list[Message], tools: list[ToolSpec] | None) -> LLMResponse:
    from app.generation.prompts import NOT_FOUND  # local import to avoid a cycle

    question = messages[0].text
    if tools and any(t.name == "submit_troubleshooting_result" for t in tools):
        return _fake_troubleshoot(question, messages, tools)

    # Q&A: pick the numbered source with the best keyword overlap.
    sources = re.findall(r"<source id=\"(\d+)\"[^>]*>\n(.*?)\n</source>", question, re.S)
    q_part = question.rsplit("<question>", 1)[-1]
    q_words = _keywords(q_part)
    best, best_score = None, 0.0
    for sid, text in sources:
        overlap = len(q_words & _keywords(text)) / max(1, len(q_words))
        if overlap > best_score:
            best, best_score = (sid, text), overlap
    if best is None or best_score < 0.5:
        return fake_response(NOT_FOUND)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", best[1].split("\n", 1)[-1]) if s.strip()]
    ranked = sorted(sentences, key=lambda s: -len(q_words & _keywords(s)))[:2]
    return fake_response(" ".join(s.strip() for s in ranked) + f" [{best[0]}]")


def _fake_troubleshoot(question: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
    from app.fault_codes import find_fault_codes

    results = [r for m in messages for r in m.tool_results if not r.is_error]
    tool_names = {t.name for t in tools}
    if not results and "lookup_fault_code" in tool_names:
        codes = find_fault_codes(question)
        if codes:
            return fake_response(tool_calls=[ToolCall("fake_1", "lookup_fault_code", {"code": codes[0]})])
        return fake_response(tool_calls=[ToolCall("fake_1", "search_manuals", {"query": question, "top_k": 4})])

    hits: list[dict] = []
    for r in results:
        try:
            hits.extend(json.loads(r.content).get("results", []))
        except json.JSONDecodeError:
            continue
    if not hits:
        payload = {
            "fault_code": None, "summary": "Not found in the provided manuals.", "likely_causes": [],
            "diagnostic_steps": [], "safety_notes": ["Follow lockout/tagout and the manufacturer's safety instructions."],
            "citations": [], "confidence": "low", "found_in_manuals": False,
        }
    else:
        top = hits[0]
        text = top["text"]
        cause = re.search(r"Cause: (.*)", text)
        remedy = re.search(r"Remedy: (.*)", text)
        steps = [s.strip() for s in re.split(r"(?<=[.;])\s+|,\s*then\s+", remedy.group(1)) if s.strip()] if remedy else []
        first_line = text.splitlines()[1] if "\n" in text else text[:200]
        payload = {
            "fault_code": top.get("fault_code") or None,
            "summary": first_line[:300],
            "likely_causes": [cause.group(1)] if cause else ["See the cited manual section."],
            "diagnostic_steps": steps or ["Follow the procedure in the cited manual section."],
            "safety_notes": ["Apply lockout/tagout before working on wiring or the drive."],
            "citations": [{"doc_title": top["doc_title"], "page": top["page"], "snippet": first_line[:200]}],
            "confidence": "high" if top.get("fault_code") else "medium",
            "found_in_manuals": True,
        }
    return fake_response(tool_calls=[ToolCall("fake_submit", "submit_troubleshooting_result", payload)])


# --------------------------------------------------------------------------- factory

def build_llm(settings: Settings, model: str | None = None) -> LLM:
    if settings.llm_provider == "fake":
        return FakeLLM()
    if settings.llm_provider == "gemini":
        return GeminiLLM(settings, model)
    return AnthropicLLM(settings, model)
