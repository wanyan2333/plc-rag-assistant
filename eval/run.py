"""Evaluation runner.

    python -m eval.run --retrieval all              # retrieval + answers + LLM judge
    python -m eval.run --retrieval all --no-generation   # retrieval metrics only (free, offline)
    python -m eval.run --retrieval hybrid --limit 5

Writes eval/results/<timestamp>.md (report) and .json (raw per-question records).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.generation.llm import FakeLLM, LLMError, Message, Usage, build_llm, estimate_cost
from app.generation.qa import answer_question
from app.services import Services
from eval.metrics import (
    Ref,
    citation_accuracy,
    first_relevant_rank,
    hit_at_k,
    mean,
    parse_judge,
    reciprocal_rank,
)

EVAL_DIR = Path(__file__).resolve().parent
MODES = ["vector", "bm25", "hybrid"]
KS = (1, 3, 5)
JUDGE_SYSTEM = "You are a strict, fair grader of technical answers. Output only JSON."

log = logging.getLogger("eval")


def load_dataset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def judge_answer(judge, template: str, item: dict, answer: str) -> tuple[int | None, str, Usage]:
    prompt = template.format(question=item["question"], reference=item["reference_answer"], candidate=answer)
    try:
        reply = judge.chat(JUDGE_SYSTEM, [Message("user", prompt)], max_tokens=2048)
    except LLMError as exc:
        return None, f"judge error: {exc}", Usage()
    score, reason = parse_judge(reply.text)
    return score, reason, reply.usage


def evaluate_item(item: dict, mode: str, services: Services, args, judge, template: str) -> dict:
    retriever = services.retriever
    answerable = item["type"] != "unanswerable"
    record: dict = {"id": item["id"], "type": item["type"], "mode": mode, "question": item["question"]}

    start = time.perf_counter()
    results = retriever.search(item["question"], k=max(KS), mode=mode)
    record["retrieval_ms"] = int((time.perf_counter() - start) * 1000)
    refs = [Ref(r.chunk.doc_id, r.chunk.page, r.chunk.page_end) for r in results]
    record["retrieved"] = [r.chunk.chunk_id for r in results]
    if answerable:
        rank = first_relevant_rank(refs, item["expected_doc"], item["expected_pages"])
        record["rank"] = rank
        record["rr"] = reciprocal_rank(rank)
        for k in KS:
            record[f"hit@{k}"] = hit_at_k(rank, k)

    if args.no_generation:
        return record

    try:
        qa = answer_question(item["question"], retriever, services.llm, k=args.top_k, mode=mode)
    except LLMError as exc:
        record["error"] = str(exc)
        return record
    usage = Usage(qa.usage["input_tokens"], qa.usage["output_tokens"])
    record.update(
        answer=qa.answer,
        refused=not qa.found_in_manuals,
        latency_ms=qa.latency_ms,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=estimate_cost(qa.model, usage, services.settings),
        model=qa.model,
        cited=[f"{c.chunk_id.split('::')[0]}:p{c.page}" for c in qa.citations],
    )
    citations = [Ref(c.chunk_id.split("::")[0], c.page, c.page_end) for c in qa.citations]
    if answerable:
        record["citation_accuracy"] = citation_accuracy(citations, item["expected_doc"], item["expected_pages"])
    else:
        record["refusal_correct"] = 1.0 if record["refused"] else 0.0

    if judge is not None:
        score, reason, judge_usage = judge_answer(judge, template, item, qa.answer)
        record.update(judge_score=score, judge_reason=reason, judge_tokens=judge_usage.to_dict())
    return record


def summarize(records: list[dict]) -> dict:
    answerable = [r for r in records if r["type"] != "unanswerable"]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]
    generated = [r for r in records if "answer" in r]
    summary = {f"hit@{k}": mean([r.get(f"hit@{k}") for r in answerable]) for k in KS}
    summary["mrr"] = mean([r.get("rr") for r in answerable])
    if generated:
        summary.update(
            judge=mean([r.get("judge_score") for r in generated]),
            citation_accuracy=mean([r.get("citation_accuracy") for r in answerable]),
            refusal_accuracy=mean([r.get("refusal_correct") for r in unanswerable]),
            false_refusal_rate=mean([1.0 if r.get("refused") else 0.0 for r in answerable if "answer" in r]),
            latency_ms=mean([r.get("latency_ms") for r in generated]),
            input_tokens=mean([r.get("input_tokens") for r in generated]),
            output_tokens=mean([r.get("output_tokens") for r in generated]),
            cost_usd=mean([r.get("cost_usd") for r in generated]),
            errors=sum(1 for r in records if "error" in r),
        )
    return summary


def by_type(records: list[dict], metric: str) -> dict[str, float | None]:
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        if r["type"] != "unanswerable":
            groups[r["type"]].append(r.get(metric))
    return {t: mean(v) for t, v in sorted(groups.items())}


def fmt(value, kind: str = "ratio") -> str:
    if value is None:
        return "n/a"
    if kind == "ratio":
        return f"{value:.2f}"
    if kind == "score":
        return f"{value:.2f} / 5"
    if kind == "ms":
        return f"{value / 1000:.1f} s"
    if kind == "int":
        return f"{value:,.0f}"
    if kind == "usd":
        return f"${value:.4f}"
    return str(value)


def render_report(meta: dict, per_mode: dict[str, list[dict]]) -> str:
    modes = list(per_mode)
    summaries = {m: summarize(per_mode[m]) for m in modes}
    lines = [
        f"# Evaluation results - {meta['timestamp']}",
        "",
        f"- Dataset: `{meta['dataset']}` - {meta['n_questions']} questions "
        f"({meta['n_answerable']} answerable, {meta['n_unanswerable']} unanswerable; {meta['needs_review']} still marked needs_review)",
        f"- Documents: {meta['documents']}",
        f"- Embeddings: `{meta['embedding']}` - chunks: {meta['chunks']}",
        f"- Answer model: `{meta['model']}` - judge: `{meta['judge_model']}` - top_k for answers: {meta['top_k']}",
        "",
        "## Retrieval (answerable questions)",
        "",
        "A hit means a retrieved chunk from the expected document covers one of the expected pages.",
        "",
        "| Mode | Hit@1 | Hit@3 | Hit@5 | MRR |",
        "|---|---|---|---|---|",
    ]
    for m in modes:
        s = summaries[m]
        lines.append(f"| {m} | {fmt(s['hit@1'])} | {fmt(s['hit@3'])} | {fmt(s['hit@5'])} | {fmt(s['mrr'])} |")

    types = sorted({r["type"] for r in per_mode[modes[0]] if r["type"] != "unanswerable"})
    lines += ["", "### Hit@1 by question type", "", "| Mode | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for m in modes:
        h = by_type(per_mode[m], "hit@1")
        lines.append(f"| {m} | " + " | ".join(fmt(h.get(t)) for t in types) + " |")

    if "judge" in summaries[modes[0]]:
        lines += [
            "",
            "## Answer quality",
            "",
            "| Mode | LLM judge (1-5) | Citation accuracy | Refusal accuracy (unanswerable) | False refusals (answerable) | Avg latency | Avg tokens in / out | Avg cost / question |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for m in modes:
            s = summaries[m]
            lines.append(
                f"| {m} | {fmt(s['judge'], 'score')} | {fmt(s['citation_accuracy'])} | {fmt(s['refusal_accuracy'])} | "
                f"{fmt(s['false_refusal_rate'])} | {fmt(s['latency_ms'], 'ms')} | "
                f"{fmt(s['input_tokens'], 'int')} / {fmt(s['output_tokens'], 'int')} | {fmt(s['cost_usd'], 'usd')} |"
            )
        errors = sum(summaries[m].get("errors", 0) for m in modes)
        if errors:
            lines += ["", f"> {errors} generation call(s) failed (API errors) and are excluded from answer metrics."]

    lines += ["", "## Per-question detail", ""]
    header = "| ID | Type | " + " | ".join(f"{m} rank" for m in modes)
    has_answers = "answer" in per_mode[modes[-1]][0] if per_mode[modes[-1]] else False
    if has_answers:
        header += f" | {modes[-1]} judge | {modes[-1]} refused | {modes[-1]} cited |"
    else:
        header += " |"
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for i, r in enumerate(per_mode[modes[0]]):
        ranks = []
        for m in modes:
            rec = per_mode[m][i]
            ranks.append("-" if rec["type"] == "unanswerable" else str(rec.get("rank") or "miss"))
        row = f"| {r['id']} | {r['type']} | " + " | ".join(ranks)
        if has_answers:
            last = per_mode[modes[-1]][i]
            row += f" | {last.get('judge_score', 'n/a')} | {'yes' if last.get('refused') else 'no'} | {', '.join(last.get('cited', [])) or '-'} |"
        else:
            row += " |"
        lines.append(row)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run", description="Evaluate retrieval and answer quality.")
    parser.add_argument("--retrieval", default="all", choices=["all", *MODES])
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.jsonl")
    parser.add_argument("--no-generation", action="store_true", help="retrieval metrics only (no LLM calls)")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge score")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--top-k", type=int, default=None, help="chunks passed to the answer model (default TOP_K)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    settings = get_settings()
    args.top_k = args.top_k or settings.top_k
    services = Services(settings)
    dataset = load_dataset(args.dataset)[: args.limit]
    modes = MODES if args.retrieval == "all" else [args.retrieval]

    judge = None
    judge_model = "n/a"
    if not args.no_generation and not args.no_judge and not isinstance(services.llm, FakeLLM):
        judge = build_llm(settings, model=settings.judge_model) if settings.judge_model else services.llm
        judge_model = judge.model
    template = (EVAL_DIR / "judge_prompt.md").read_text(encoding="utf-8")

    _ = services.retriever  # load index + embedding model once
    per_mode: dict[str, list[dict]] = {}
    for mode in modes:
        print(f"[{mode}] evaluating {len(dataset)} questions...", flush=True)
        with ThreadPoolExecutor(max_workers=1 if args.no_generation else args.workers) as pool:
            per_mode[mode] = list(pool.map(lambda item: evaluate_item(item, mode, services, args, judge, template), dataset))
        s = summarize(per_mode[mode])
        print(f"[{mode}] Hit@1={fmt(s['hit@1'])} Hit@5={fmt(s['hit@5'])} MRR={fmt(s['mrr'])}"
              + (f" judge={fmt(s.get('judge'))} refusal={fmt(s.get('refusal_accuracy'))}" if "judge" in s else ""), flush=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {
        "timestamp": timestamp,
        "dataset": args.dataset.name,
        "n_questions": len(dataset),
        "n_answerable": sum(1 for d in dataset if d["type"] != "unanswerable"),
        "n_unanswerable": sum(1 for d in dataset if d["type"] == "unanswerable"),
        "needs_review": sum(1 for d in dataset if d.get("needs_review")),
        "documents": ", ".join(d["doc_title"] for d in services.store.documents()),
        "chunks": len(services.store.chunks),
        "embedding": services.embedder.name,
        "model": "n/a (retrieval only)" if args.no_generation else services.llm.model,
        "judge_model": judge_model,
        "top_k": args.top_k,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    report = render_report(meta, per_mode)
    md_path = args.out / f"{timestamp}.md"
    md_path.write_text(report, encoding="utf-8")
    (args.out / f"{timestamp}.json").write_text(
        json.dumps({"meta": meta, "summary": {m: summarize(r) for m, r in per_mode.items()}, "records": per_mode},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nReport written to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
