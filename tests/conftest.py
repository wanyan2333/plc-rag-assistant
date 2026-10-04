"""Shared fixtures. Everything here is offline: synthetic PDF, hash embeddings, FakeLLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.ingest.pdf_parser import ParsedDocument, parse_pdf
from app.retrieval.embeddings import HashEmbedder
from tests.fixtures.make_fixture_pdf import make_fixture_pdf


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("raw")
    make_fixture_pdf(directory / "fx100_manual.pdf")
    return directory


@pytest.fixture(scope="session")
def fixture_pdf(raw_dir) -> Path:
    return raw_dir / "fx100_manual.pdf"


@pytest.fixture(scope="session")
def parsed_doc(fixture_pdf) -> ParsedDocument:
    return parse_pdf(fixture_pdf)


@pytest.fixture(scope="session")
def settings(raw_dir, tmp_path_factory) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="fake",
        embedding_backend="hash",
        raw_dir=raw_dir,
        index_dir=tmp_path_factory.mktemp("index"),
    )


@pytest.fixture(scope="session")
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture(scope="session")
def built_index(settings, embedder) -> Path:
    from app.ingest.pipeline import ingest

    ingest(settings, rebuild=True, embedder=embedder)
    return settings.index_path
