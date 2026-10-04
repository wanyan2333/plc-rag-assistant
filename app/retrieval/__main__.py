"""CLI: python -m app.retrieval "E-101 overcurrent" [--mode hybrid|vector|bm25] [-k 5]"""

from __future__ import annotations

import argparse
import textwrap

from app.services import Services


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.retrieval", description="Query the manual index.")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)

    services = Services()
    results = services.retriever.search(args.query, k=args.k, mode=args.mode)
    if not results:
        print("No results.")
    for r in results:
        c = r.chunk
        flag = " [exact fault code]" if r.exact_fault_match else ""
        print(f"#{r.rank} score={r.score:.4f} {c.chunk_type}{flag} | {c.doc_title} p.{c.page} | {c.section_heading}")
        print(textwrap.indent(textwrap.shorten(c.text, 300), "    "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
