"""FastAPI application.

    uvicorn app.api.main:app --reload
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    FaultCodeEntry,
    FaultCodeResponse,
    HealthResponse,
)
from app.generation.llm import LLMError
from app.generation.qa import answer_question
from app.generation.troubleshoot import troubleshoot
from app.services import Services

log = logging.getLogger("plc_rag.requests")


def _model_name(services: Services) -> str:
    s = services.settings
    return {"anthropic": s.anthropic_model, "gemini": s.gemini_model, "fake": "fake-llm"}[s.llm_provider]


def create_app(services: Services | None = None, warmup: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if warmup:
            try:  # load the index and embedding model once, not on the first request
                await run_in_threadpool(lambda: app.state.services.retriever)
            except RuntimeError as exc:
                log.warning("Index not ready: %s", exc)
        yield

    app = FastAPI(
        title="PLC Troubleshooting Assistant",
        version="0.1.0",
        description="RAG over PLC / drive manuals: cited answers and validated troubleshooting plans.",
        lifespan=lifespan,
    )
    app.state.services = services or Services()

    def get_services(request: Request) -> Services:
        return request.app.state.services

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        services = get_services(request)
        try:
            store = services.store
            return HealthResponse(
                status="ok",
                index_loaded=True,
                documents=len(store.documents()),
                chunks=len(store.chunks),
                llm_provider=services.settings.llm_provider,
                llm_model=_model_name(services),
            )
        except RuntimeError as exc:
            return HealthResponse(
                status="degraded",
                index_loaded=False,
                documents=0,
                chunks=0,
                llm_provider=services.settings.llm_provider,
                llm_model=_model_name(services),
                detail=str(exc),
            )

    @app.get("/documents", response_model=list[DocumentInfo])
    def documents(request: Request) -> list[DocumentInfo]:
        services = get_services(request)
        try:
            return [DocumentInfo(**d) for d in services.store.documents()]
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/fault-codes/{code}", response_model=FaultCodeResponse)
    def fault_code(code: str, request: Request) -> FaultCodeResponse:
        services = get_services(request)
        try:
            chunks = services.store.lookup_fault_code(code)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not chunks:
            raise HTTPException(status_code=404, detail=f"Fault code {code!r} not found in the ingested manuals")
        entries = [
            FaultCodeEntry(
                chunk_id=c.chunk_id,
                doc_title=c.doc_title,
                page=c.page,
                section_heading=c.section_heading,
                fault_code=c.fault_code,
                text=c.text,
            )
            for c in chunks
        ]
        return FaultCodeResponse(code=code, found=True, entries=entries)

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        services = get_services(request)
        start = time.perf_counter()
        try:
            retriever, llm = services.retriever, services.llm
        except (RuntimeError, LLMError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if body.mode == "qa":
            try:
                result = answer_question(
                    body.question, retriever, llm, k=body.top_k, mode=body.retrieval, allow_general=body.allow_general
                )
            except LLMError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            response = AskResponse(mode="qa", retrieval=body.retrieval, qa=result)
            chunk_ids, usage = result.retrieved_chunk_ids, result.usage
        else:
            result = troubleshoot(
                body.question, retriever, llm, services.settings, mode=body.retrieval, allow_general=body.allow_general
            )
            response = AskResponse(mode="troubleshoot", retrieval=body.retrieval, troubleshoot=result)
            chunk_ids, usage = result.retrieved_chunk_ids, result.usage

        log.info(
            json.dumps(
                {
                    "event": "ask",
                    "mode": body.mode,
                    "retrieval": body.retrieval,
                    "latency_ms": int((time.perf_counter() - start) * 1000),
                    "chunk_ids": chunk_ids,
                    "usage": usage,
                    "general_answer": bool((response.qa or response.troubleshoot).general_answer),
                    "question_chars": len(body.question),
                }
            )
        )
        return response

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
for _noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

app = create_app()
