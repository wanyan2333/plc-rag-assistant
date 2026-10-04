"""CLI: python -m app.generation "question" [--mode qa|troubleshoot] [--retrieval hybrid|vector|bm25]"""

from __future__ import annotations

import argparse
import json
import logging

from app.services import Services


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.generation", description="Ask the manuals a question.")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["qa", "troubleshoot"], default="qa")
    parser.add_argument("--retrieval", choices=["hybrid", "vector", "bm25"], default="hybrid")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    services = Services()
    if args.mode == "qa":
        from app.generation.qa import answer_question

        result = answer_question(args.question, services.retriever, services.llm, k=services.settings.top_k, mode=args.retrieval)
        print(result.answer, "\n")
        for c in result.citations:
            print(f"[{c.n}] {c.doc_title}, p.{c.page}: {c.snippet}")
        print(f"\n{result.model} | {result.latency_ms} ms | tokens {result.usage}")
    else:
        from app.generation.troubleshoot import troubleshoot

        result = troubleshoot(args.question, services.retriever, services.llm, services.settings, mode=args.retrieval)
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
