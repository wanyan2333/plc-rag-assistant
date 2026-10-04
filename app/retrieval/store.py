"""On-disk chunk store (JSONL) plus an index manifest.

Chroma holds the vectors; this file is the source of truth for chunk text/metadata and
feeds the BM25 index and the exact fault-code table.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.fault_codes import normalize_code
from app.ingest.chunker import Chunk

CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


def save_chunks(index_dir: Path, chunks: list[Chunk], manifest: dict) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    tmp = index_dir / (CHUNKS_FILE + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(index_dir / CHUNKS_FILE)
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_chunks(index_dir: Path) -> list[Chunk]:
    path = index_dir / CHUNKS_FILE
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]


def load_manifest(index_dir: Path) -> dict:
    path = index_dir / MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@dataclass
class ChunkStore:
    chunks: list[Chunk]

    def __post_init__(self) -> None:
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.by_fault_code: dict[str, list[Chunk]] = {}
        for c in self.chunks:
            if c.chunk_type == "fault_code" and c.fault_code:
                self.by_fault_code.setdefault(c.fault_code, []).append(c)

    @classmethod
    def load(cls, index_dir: Path) -> "ChunkStore":
        return cls(load_chunks(index_dir))

    def get(self, chunk_id: str) -> Chunk | None:
        return self.by_id.get(chunk_id)

    def lookup_fault_code(self, code: str) -> list[Chunk]:
        return list(self.by_fault_code.get(normalize_code(code), []))

    def documents(self) -> list[dict]:
        counts = Counter(c.doc_id for c in self.chunks)
        fault_counts = Counter(c.doc_id for c in self.chunks if c.chunk_type == "fault_code")
        titles = {c.doc_id: c.doc_title for c in self.chunks}
        pages = Counter()
        for c in self.chunks:
            pages[c.doc_id] = max(pages[c.doc_id], c.page_end)
        return [
            {
                "doc_id": doc_id,
                "doc_title": titles[doc_id],
                "chunks": counts[doc_id],
                "fault_code_chunks": fault_counts[doc_id],
                "pages": pages[doc_id],
            }
            for doc_id in sorted(counts)
        ]
