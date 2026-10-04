"""CLI: python -m app.ingest [--rebuild] [--raw-dir DIR] [--index-dir DIR]"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.ingest.pipeline import ingest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ingest", description="Ingest PLC manuals into the search index.")
    parser.add_argument("--rebuild", action="store_true", help="drop the existing index and rebuild from scratch")
    parser.add_argument("--raw-dir", type=Path, help="folder with PDF manuals (default: RAW_DIR)")
    parser.add_argument("--index-dir", type=Path, help="index output folder (default: INDEX_DIR)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    try:
        stats = ingest(settings, raw_dir=args.raw_dir, index_dir=args.index_dir, rebuild=args.rebuild)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
