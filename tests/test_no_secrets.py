"""Guard rail: no API keys or manuals may ever be committed."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SECRET_RE = re.compile(
    r"sk-ant-[A-Za-z0-9_\-]{20,}"  # Anthropic
    r"|sk-or-v1-[A-Za-z0-9]{20,}"  # OpenRouter
    r"|AIza[0-9A-Za-z_\-]{30,}"  # Google API key
    r"|AQ\.[A-Za-z0-9_\-]{30,}"  # Google auth token
)


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [line for line in out.splitlines() if line]


def test_env_file_is_not_tracked():
    files = tracked_files()
    assert ".env" not in files
    assert not [f for f in files if f.startswith("data/raw/") or f.endswith(".pdf")]


def test_no_api_keys_in_tracked_files():
    leaks = []
    for name in tracked_files():
        path = ROOT / name
        if not path.is_file() or path.suffix in {".lock", ".pdf", ".png", ".jpg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if SECRET_RE.search(text):
            leaks.append(name)
    assert not leaks, f"possible API key committed in: {leaks}"
