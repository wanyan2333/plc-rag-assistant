"""Publish the app to a Hugging Face Space (Docker SDK).

    hf auth login                                   # once, with a *write* token
    python -m scripts.deploy_hf_space               # -> <your-hf-user>/plc-rag-assistant
    python -m scripts.deploy_hf_space --space someone/other-name --set-gemini-secret

Uploads the git-tracked files (so .env, data/ and personal notes are never included),
with a Space-specific README header. Images are linked from GitHub instead of uploaded.
The Gemini key must exist as a Space secret named GEMINI_API_KEY: add it in the Space
settings, or pass --set-gemini-secret to copy it from your local .env.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
GITHUB_RAW = "https://raw.githubusercontent.com/wanyan2333/plc-rag-assistant/main/"
SKIP = re.compile(r"^(docs/|\.gitattributes$)")  # binaries are served from GitHub; keep the Space's LFS rules

SPACE_HEADER = """---
title: PLC Troubleshooting Assistant
emoji: 🔧
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: RAG over PLC/drive manuals - cited answers, JSON plans
---

"""


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f and not SKIP.match(f)]


def stage(target: Path) -> int:
    files = tracked_files()
    for name in files:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme = re.sub(r"\]\((docs/[^)]+)\)", lambda m: f"]({GITHUB_RAW}{m.group(1)})", readme)
    (target / "README.md").write_text(SPACE_HEADER + readme, encoding="utf-8")
    return len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", help="Space id <user>/<name> (default: <your-hf-user>/plc-rag-assistant)")
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    parser.add_argument("--set-gemini-secret", action="store_true", help="copy GEMINI_API_KEY from .env into the Space secrets")
    parser.add_argument("--dry-run", type=Path, metavar="DIR", help="only stage the upload into DIR (no login, no upload)")
    args = parser.parse_args(argv)

    if args.dry_run:
        count = stage(args.dry_run)
        print(f"Staged {count} files into {args.dry_run}")
        return 0

    api = HfApi()
    user = api.whoami()["name"]
    space_id = args.space or f"{user}/plc-rag-assistant"

    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    if args.set_gemini_secret:
        key = dotenv_values(ROOT / ".env").get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("GEMINI_API_KEY not found in .env")
        api.add_space_secret(space_id, "GEMINI_API_KEY", key, description="Gemini API key for answers")
        print("Space secret GEMINI_API_KEY set.")

    with tempfile.TemporaryDirectory() as tmp:
        count = stage(Path(tmp))
        api.upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=tmp,
            commit_message="Deploy from GitHub main",
            delete_patterns=["*.py", "*.md", "*.json", "*.jsonl", "*.toml", "*.lock", "Dockerfile", ".dockerignore", ".gitignore", ".env.example"],
        )
    print(f"Uploaded {count} files to https://huggingface.co/spaces/{space_id}")
    print("The Space now builds the Docker image (about 5-10 minutes). Watch progress on the Space page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
