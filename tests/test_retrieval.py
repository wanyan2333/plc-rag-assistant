import pytest

from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.hybrid import HybridRetriever, rrf_fuse
from app.retrieval.store import ChunkStore
from app.retrieval.vector import VectorIndex


@pytest.fixture(scope="module")
def retriever(built_index, embedder):
    store = ChunkStore.load(built_index)
    return HybridRetriever(store, VectorIndex(built_index, embedder))


def test_tokenizer_keeps_fault_codes_and_variants():
    tokens = tokenize("Fault 16#8085 and E-101 on S7-1200")
    assert "16#8085" in tokens and "8085" in tokens
    assert "e-101" in tokens and "e101" in tokens
    assert "and" not in tokens


@pytest.mark.parametrize("query,code", [("16#8085", "16#8085"), ("what is F-0042", "F0042"), ("e101 fault", "E101")])
def test_bm25_hits_fault_codes_exactly(retriever, query, code):
    results = BM25Index(retriever.store.chunks).search(query, k=3)
    top = retriever.store.get(results[0][0])
    assert top.chunk_type == "fault_code"
    assert top.fault_code == code


def test_rrf_scores_and_order():
    fused = rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60)
    ids = [doc for doc, _ in fused]
    assert ids == ["b", "c", "a", "d"]
    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["d"] == pytest.approx(1 / 63)


def test_rrf_tie_break_prefers_better_best_rank():
    fused = rrf_fuse([["x", "y"], ["y", "x"]], k=60)
    # Equal scores; both have best rank 1 so the id decides deterministically.
    assert [d for d, _ in fused] == ["x", "y"]
    fused = rrf_fuse([["p"], ["q", "r"]], k=60)
    assert [d for d, _ in fused][:2] == ["p", "q"]


def test_hybrid_puts_exact_fault_code_first(retriever):
    results = retriever.search("ERROR LED flashing, controller shows E-205", k=4, mode="hybrid")
    assert results[0].exact_fault_match
    assert results[0].chunk.fault_code == "E205"
    assert [r.rank for r in results] == [1, 2, 3, 4]


def test_unknown_code_does_not_break_hybrid(retriever):
    results = retriever.search("fault E-999", k=3, mode="hybrid")
    assert results and not any(r.exact_fault_match for r in results)


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_all_modes_return_ranked_chunks(retriever, mode):
    results = retriever.search("lockout tagout before opening the cabinet", k=3, mode=mode)
    assert 1 <= len(results) <= 3
    assert results[0].rank == 1


def test_invalid_mode(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")
