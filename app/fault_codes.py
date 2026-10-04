"""Fault-code pattern detection and normalisation, shared by ingestion and retrieval.

Manuals use several notations for the same idea:
  * Siemens diagnostic event IDs:    16#8085
  * Rockwell hexadecimal fault codes: 0xF0A3
  * Vendor-prefixed codes:           E-101, F-0042, AL-12, F0042
"""

from __future__ import annotations

import re

FAULT_CODE_RE = re.compile(
    r"""
    (?<![\w#])(?:
        16\#[0-9A-Fa-f]{4}              # Siemens 16#xxxx
      | 0x[0-9A-Fa-f]{2,8}              # hex codes
      | [A-Z]{1,3}-\d{2,5}              # E-101, F-0042, AL-12
      | [A-Z]{1,2}\d{3,5}               # F0042, E101
    )(?![\w#-])
    """,
    re.VERBOSE,
)


def normalize_code(code: str) -> str:
    """Canonical key used for exact matching (case/dash/space insensitive)."""
    return re.sub(r"[\s\-_]", "", code.strip().upper())


def find_fault_codes(text: str) -> list[str]:
    """Return fault codes mentioned in `text`, in order of appearance, without duplicates."""
    seen: dict[str, str] = {}
    for match in FAULT_CODE_RE.finditer(text):
        raw = match.group(0)
        seen.setdefault(normalize_code(raw), raw)
    return list(seen.values())


def is_fault_code(text: str) -> bool:
    match = FAULT_CODE_RE.fullmatch(text.strip())
    return match is not None
