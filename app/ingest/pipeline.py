"""Ingestion pipeline: PDFs -> parsed pages -> chunks -> Chroma + JSONL store."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.ingest.chunker import Chunk, chunk_document
from app.ingest.pdf_parser import parse_pdf
from app.retrieval.embeddings import Embedder, build_embedder
from app.retrieval.store import load_manifest, save_chunks
from app.retrieval.vector import VectorIndex

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    documents: int = 0
    pages: int = 0
    chunks: int = 0
    fault_code_chunks: int = 0
    skipped: bool = False
    removed_stale: int = 0
    seconds: float = 0.0
    per_document: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        if self.skipped:
            return f"Index is up to date ({self.documents} documents, {self.chunks} chunks) - nothing to do. Use --rebuild to force."
        lines = [
            f"Documents processed : {self.documents}",
            f"Pages               : {self.pages}",
            f"Chunks              : {self.chunks}",
            f"  fault_code chunks : {self.fault_code_chunks}",
            f"  text chunks       : {self.chunks - self.fault_code_chunks}",
            f"Stale vectors removed: {self.removed_stale}",
            f"Elapsed             : {self.seconds:.1f}s",
        ]
        for d in self.per_document:
            lines.append(
                f"  - {d['doc_title']} ({d['file']}): {d['pages']} pages, {d['chunks']} chunks, {d['fault_code_chunks']} fault codes"
            )
        return "\n".join(lines)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest(
    settings: Settings,
    raw_dir: Path | None = None,
    index_dir: Path | None = None,
    rebuild: bool = False,
    embedder: Embedder | None = None,
) -> IngestStats:
    start = time.perf_counter()
    raw_dir = raw_dir or settings.raw_path
    index_dir = index_dir or settings.index_path
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    chunk_cfg = {
        "min": settings.chunk_min_tokens,
        "max": settings.chunk_max_tokens,
        "overlap": settings.chunk_overlap_tokens,
    }
    embedding_name = embedder.name if embedder else (
        f"hash-384" if settings.embedding_backend == "hash" else settings.embedding_model
    )
    fingerprint = {
        "files": {p.name: _file_hash(p) for p in pdfs},
        "chunking": chunk_cfg,
        "embedding": embedding_name,
    }

    manifest = load_manifest(index_dir)
    if not rebuild and manifest.get("fingerprint") == fingerprint:
        stats = IngestStats(**{k: manifest["stats"][k] for k in ("documents", "pages", "chunks", "fault_code_chunks")})
        stats.skipped = True
        return stats

    stats = IngestStats()
    all_chunks: list[Chunk] = []
    for pdf in pdfs:
        log.info("Parsing %s", pdf.name)
        doc = parse_pdf(pdf)
        chunks = chunk_document(doc, chunk_cfg["min"], chunk_cfg["max"], chunk_cfg["overlap"])
        n_fault = sum(c.chunk_type == "fault_code" for c in chunks)
        stats.documents += 1
        stats.pages += len(doc.pages)
        stats.chunks += len(chunks)
        stats.fault_code_chunks += n_fault
        stats.per_document.append(
            {
                "file": pdf.name,
                "doc_id": doc.doc_id,
                "doc_title": doc.doc_title,
                "pages": len(doc.pages),
                "chunks": len(chunks),
                "fault_code_chunks": n_fault,
            }
        )
        all_chunks.extend(chunks)

    ids = [c.chunk_id for c in all_chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk ids - two PDFs share the same file stem?")

    embedder = embedder or build_embedder(settings)
    vectors = VectorIndex(index_dir, embedder)
    if rebuild:
        vectors.reset()
    vectors.upsert(all_chunks)
    stats.removed_stale = vectors.delete_except(set(ids))

    stats.seconds = time.perf_counter() - start
    save_chunks(
        index_dir,
        all_chunks,
        {
            "fingerprint": fingerprint,
            "stats": {
                "documents": stats.documents,
                "pages": stats.pages,
                "chunks": stats.chunks,
                "fault_code_chunks": stats.fault_code_chunks,
            },
            "documents": stats.per_document,
        },
    )
    return stats
