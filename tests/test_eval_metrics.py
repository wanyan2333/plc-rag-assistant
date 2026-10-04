import json
from pathlib import Path

from eval.metrics import Ref, citation_accuracy, first_relevant_rank, hit_at_k, mean, parse_judge, reciprocal_rank
from eval.run import render_report, summarize

DATASET = Path(__file__).resolve().parent.parent / "eval" / "dataset.jsonl"


def test_rank_hit_and_mrr():
    refs = [Ref("a", 1), Ref("b", 3, 4), Ref("b", 7)]
    rank = first_relevant_rank(refs, "b", [4])
    assert rank == 2  # page span 3-4 covers expected page 4
    assert hit_at_k(rank, 1) == 0.0 and hit_at_k(rank, 3) == 1.0
    assert reciprocal_rank(rank) == 0.5
    assert first_relevant_rank(refs, "c", [1]) is None
    assert reciprocal_rank(None) == 0.0


def test_citation_accuracy():
    assert citation_accuracy([Ref("a", 2), Ref("a", 9)], "a", [2]) == 0.5
    assert citation_accuracy([], "a", [2]) is None


def test_parse_judge():
    assert parse_judge('Sure: {"score": 4, "reason": "minor gap"}') == (4, "minor gap")
    assert parse_judge("Score: 2 because it is wrong")[0] == 2
    assert parse_judge('{"score": 9}')[0] is None


def test_mean_ignores_none():
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None]) is None


def test_dataset_is_well_formed():
    items = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert 25 <= len(items) <= 30
    assert len({i["id"] for i in items}) == len(items)
    types = [i["type"] for i in items]
    assert types.count("fault_code") >= 5
    assert types.count("unanswerable") >= 3
    for item in items:
        assert item["type"] in {"fault_code", "procedure", "concept", "unanswerable"}
        assert item["question"] and item["reference_answer"]
        if item["type"] != "unanswerable":
            assert item["expected_doc"] and item["expected_pages"]


def test_report_renders_three_modes():
    rec = lambda mode, rank: {"id": "q1", "type": "fault_code", "mode": mode, "rank": rank, "rr": reciprocal_rank(rank),
                              **{f"hit@{k}": hit_at_k(rank, k) for k in (1, 3, 5)}}
    per_mode = {"vector": [rec("vector", 3)], "bm25": [rec("bm25", 1)], "hybrid": [rec("hybrid", 1)]}
    meta = {"timestamp": "t", "dataset": "d", "n_questions": 1, "n_answerable": 1, "n_unanswerable": 0, "needs_review": 0,
            "documents": "D", "embedding": "e", "chunks": 1, "model": "m", "judge_model": "j", "top_k": 6}
    report = render_report(meta, per_mode)
    assert "| vector | 0.00 | 1.00 | 1.00 | 0.33 |" in report
    assert "| hybrid | 1.00 |" in report
    assert summarize(per_mode["bm25"])["mrr"] == 1.0
