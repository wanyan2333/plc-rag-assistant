from app.ingest.pipeline import ingest
from app.retrieval.store import ChunkStore
from app.retrieval.vector import VectorIndex


def test_ingest_builds_store_and_vectors(built_index, embedder):
    store = ChunkStore.load(built_index)
    assert len(store.chunks) > 0
    assert len(store.lookup_fault_code("e-101")) == 1
    assert VectorIndex(built_index, embedder).count() == len(store.chunks)
    docs = store.documents()
    assert docs[0]["doc_id"] == "fx100-manual"
    assert docs[0]["fault_code_chunks"] == 5


def test_ingest_is_idempotent(settings, embedder, tmp_path):
    index_dir = tmp_path / "index"
    first = ingest(settings, index_dir=index_dir, rebuild=True, embedder=embedder)
    second = ingest(settings, index_dir=index_dir, embedder=embedder)
    assert second.skipped
    third = ingest(settings, index_dir=index_dir, rebuild=True, embedder=embedder)
    assert first.chunks == third.chunks
    assert VectorIndex(index_dir, embedder).count() == first.chunks
    assert len(ChunkStore.load(index_dir).chunks) == first.chunks


def test_ingest_stats(settings, embedder, tmp_path):
    stats = ingest(settings, index_dir=tmp_path / "idx", rebuild=True, embedder=embedder)
    assert (stats.documents, stats.pages, stats.fault_code_chunks) == (1, 4, 5)
    assert "fault_code chunks" in stats.summary()
