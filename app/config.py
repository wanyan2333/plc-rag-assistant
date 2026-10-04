"""Application settings, loaded from environment variables and an optional .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RetrievalMode = Literal["vector", "bm25", "hybrid"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    llm_provider: Literal["anthropic", "gemini", "openrouter", "fake"] = "anthropic"
    # Backup provider tried when a primary call fails (empty = no fallback)
    llm_fallback_provider: Literal["", "anthropic", "gemini", "openrouter"] = ""
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_effort: Literal["low", "medium", "high"] = "medium"
    anthropic_fallbacks: str = "default"  # "default" enables server-side refusal fallback, "" disables
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-flash-latest"
    openrouter_api_key: str | None = None
    openrouter_model: str = "thinkingmachines/inkling:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    judge_model: str | None = None  # LLM-as-judge model; defaults to the answer model
    max_output_tokens: int = 4096
    # Optional price override (USD per million tokens) for models not in the built-in table
    price_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None

    # Embeddings
    embedding_backend: Literal["sentence-transformers", "hash"] = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Paths (relative paths are resolved against the project root)
    raw_dir: Path = Path("data/raw")
    index_dir: Path = Path("data/index")

    # Chunking, in approximate tokens (see app.ingest.chunker.count_tokens)
    chunk_min_tokens: int = 500
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 80

    # Retrieval
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = 6
    rrf_k: int = 60

    # Troubleshoot agent
    max_tool_rounds: int = 4

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def raw_path(self) -> Path:
        return self.resolve(self.raw_dir)

    @property
    def index_path(self) -> Path:
        return self.resolve(self.index_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
