"""Tools exposed to the model in Troubleshoot mode, and their executor."""

from __future__ import annotations

import json

from app.config import RetrievalMode
from app.generation.llm import ToolCall, ToolResult, ToolSpec
from app.ingest.chunker import Chunk
from app.retrieval.hybrid import HybridRetriever

LOOKUP_FAULT_CODE = ToolSpec(
    name="lookup_fault_code",
    description=(
        "Exact lookup of a fault, alarm or diagnostic event code in the fault-code tables of the ingested manuals. "
        "Use it whenever the user mentions a code such as E-101, F-0042, A-0501 or 16#8085. "
        "Returns the table row(s) with description, cause and remedy, or found=false."
    ),
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "The code exactly as written, e.g. 'F-0011' or '16#8085'."}},
        "required": ["code"],
        "additionalProperties": False,
    },
)

SEARCH_MANUALS = ToolSpec(
    name="search_manuals",
    description=(
        "Hybrid (keyword + semantic) search over the ingested PLC / drive manuals. Use specific technical queries, "
        "e.g. 'ERROR LED flashing red meaning' or 'DC bus overvoltage deceleration ramp'. Returns ranked excerpts with "
        "document title and page."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 8, "description": "Number of excerpts (default 5)."},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


def _chunk_payload(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_title": chunk.doc_title,
        "page": chunk.page,
        "page_end": chunk.page_end,
        "section_heading": chunk.section_heading,
        "fault_code": chunk.fault_code,
        "text": chunk.text,
    }


class ToolExecutor:
    """Runs research tools and remembers every chunk the model has seen (for grounding checks)."""

    def __init__(self, retriever: HybridRetriever, mode: RetrievalMode = "hybrid"):
        self.retriever = retriever
        self.mode = mode
        self.seen: dict[str, Chunk] = {}
        self.trace: list[dict] = []

    def _remember(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self.seen.setdefault(c.chunk_id, c)

    def lookup_fault_code(self, code: str) -> dict:
        chunks = self.retriever.store.lookup_fault_code(code)
        self._remember(chunks)
        payload = {"code": code, "found": bool(chunks), "results": [_chunk_payload(c) for c in chunks]}
        if not chunks:
            payload["hint"] = "No exact table entry. Try search_manuals with the code and the symptom."
        return payload

    def search_manuals(self, query: str, top_k: int = 5) -> dict:
        top_k = max(1, min(int(top_k or 5), 8))
        results = self.retriever.search(query, k=top_k, mode=self.mode)
        chunks = [r.chunk for r in results]
        self._remember(chunks)
        return {"query": query, "results": [_chunk_payload(c) for c in chunks]}

    def run(self, call: ToolCall) -> ToolResult:
        try:
            if call.name == LOOKUP_FAULT_CODE.name:
                payload = self.lookup_fault_code(str(call.args["code"]))
            elif call.name == SEARCH_MANUALS.name:
                payload = self.search_manuals(str(call.args["query"]), call.args.get("top_k", 5))
            else:
                return ToolResult(call.id, call.name, f"Unknown tool {call.name!r}", is_error=True)
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult(call.id, call.name, f"Invalid arguments: {exc}", is_error=True)
        self.trace.append({"tool": call.name, "args": call.args, "results": [r["chunk_id"] for r in payload["results"]]})
        return ToolResult(call.id, call.name, json.dumps(payload, ensure_ascii=False))
