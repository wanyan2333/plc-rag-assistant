"""Request/response models for the REST API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.generation.qa import QAResult
from app.generation.troubleshoot import TroubleshootResponse


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["What does fault F-0011 mean on the VD-500?"])
    mode: Literal["qa", "troubleshoot"] = "qa"
    retrieval: Literal["hybrid", "vector", "bm25"] = "hybrid"
    top_k: int = Field(default=6, ge=1, le=12)
    allow_general: bool = Field(
        default=True,
        description="If the manuals do not cover the question, also return a clearly separated general-knowledge answer.",
    )


class AskResponse(BaseModel):
    mode: Literal["qa", "troubleshoot"]
    retrieval: str
    qa: QAResult | None = None
    troubleshoot: TroubleshootResponse | None = None


class FaultCodeEntry(BaseModel):
    chunk_id: str
    doc_title: str
    page: int
    section_heading: str
    fault_code: str
    text: str


class FaultCodeResponse(BaseModel):
    code: str
    found: bool
    entries: list[FaultCodeEntry]


class DocumentInfo(BaseModel):
    doc_id: str
    doc_title: str
    pages: int
    chunks: int
    fault_code_chunks: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    index_loaded: bool
    documents: int
    chunks: int
    llm_provider: str
    llm_model: str
    detail: str | None = None
