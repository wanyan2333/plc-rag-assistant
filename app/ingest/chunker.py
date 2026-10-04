"""Chunking strategy.

Text: packed greedily from paragraphs, preferring heading boundaries. A new chunk starts
at a heading once the current chunk reaches `min_tokens`; otherwise a chunk is closed when
the next paragraph would exceed `max_tokens`, and the next chunk then starts with roughly
`overlap_tokens` of trailing context from the previous one.

Fault-code tables: every row becomes its own chunk (`chunk_type="fault_code"`) so an exact
code lookup returns one precise, self-contained answer instead of a slice of a big table.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.fault_codes import FAULT_CODE_RE, is_fault_code, normalize_code
from app.ingest.pdf_parser import ParsedDocument, Table, TextBlock

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
MAX_UNIT_TOKENS = 150  # paragraphs are pre-split into units no larger than this


def count_tokens(text: str) -> int:
    """Approximate token count (word pieces + punctuation).

    Deliberately tokenizer-free so chunking is deterministic and offline; it tracks
    BPE token counts for English technical prose closely enough for sizing chunks.
    """
    return len(TOKEN_RE.findall(text))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int  # 1-based page where the chunk starts
    page_end: int
    section_heading: str
    chunk_type: str  # "text" | "fault_code"
    text: str
    fault_code: str = ""  # normalised code for fault_code chunks

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def metadata(self) -> dict:
        """Chroma-compatible metadata (scalars only, no None)."""
        data = self.to_dict()
        data.pop("text")
        return data

    @property
    def pages(self) -> range:
        return range(self.page, self.page_end + 1)


# --------------------------------------------------------------------------- fault codes

_CODE_HEADERS = ("code", "fault", "error", "alarm", "event", "id")
_DESC_HEADERS = ("description", "meaning", "name", "message", "fault text", "text")
_CAUSE_HEADERS = ("cause", "reason", "possible cause")
_REMEDY_HEADERS = ("remedy", "action", "solution", "corrective", "resolution", "fix")


def _find_col(header: list[str], keywords: tuple[str, ...], exclude: set[int]) -> int | None:
    for i, name in enumerate(header):
        if i in exclude:
            continue
        lowered = name.lower()
        if any(k in lowered for k in keywords):
            return i
    return None


def fault_rows(table: Table) -> list[dict[str, str]]:
    """Interpret a table as a fault-code table. Returns [] if it is not one."""
    header = [h.strip() for h in table.header]
    code_col = _find_col(header, _CODE_HEADERS, set())
    if code_col is None or not table.rows:
        # Headerless tables: accept if most first-column cells look like fault codes.
        first = [r[0] for r in table.rows if r]
        if first and sum(is_fault_code(c) for c in first) >= 0.6 * len(first):
            code_col = 0
            if is_fault_code(header[0] if header else ""):
                table = Table(page=table.page, header=[""] * len(header), rows=[header, *table.rows])
        else:
            return []

    rows = [r for r in table.rows if len(r) > code_col]
    hits = sum(bool(FAULT_CODE_RE.search(r[code_col])) for r in rows)
    if not rows or hits < 0.6 * len(rows):
        return []

    used = {code_col}
    desc_col = _find_col(header, _DESC_HEADERS, used)
    used |= {desc_col} if desc_col is not None else set()
    cause_col = _find_col(header, _CAUSE_HEADERS, used)
    used |= {cause_col} if cause_col is not None else set()
    remedy_col = _find_col(header, _REMEDY_HEADERS, used)
    used |= {remedy_col} if remedy_col is not None else set()

    def cell(row: list[str], col: int | None) -> str:
        return row[col].strip() if col is not None and col < len(row) else ""

    out = []
    for row in rows:
        match = FAULT_CODE_RE.search(row[code_col])
        if not match:
            continue
        extra = [
            f"{header[i] or 'Info'}: {v}"
            for i, v in enumerate(row)
            if i not in used and v.strip() and i < len(header)
        ]
        out.append(
            {
                "code": match.group(0),
                "description": cell(row, desc_col),
                "cause": cell(row, cause_col),
                "remedy": cell(row, remedy_col),
                "extra": "; ".join(extra),
            }
        )
    return out


def _fault_chunk_text(row: dict[str, str], doc_title: str, heading: str) -> str:
    parts = [f"Fault code {row['code']}"]
    if row["description"]:
        parts[0] += f": {row['description']}"
    if row["cause"]:
        parts.append(f"Cause: {row['cause']}")
    if row["remedy"]:
        parts.append(f"Remedy: {row['remedy']}")
    if row["extra"]:
        parts.append(row["extra"])
    context = f"[{doc_title}" + (f" - {heading}]" if heading else "]")
    return context + "\n" + "\n".join(parts)


# --------------------------------------------------------------------------- text

def _split_units(text: str, max_tokens: int = MAX_UNIT_TOKENS) -> list[str]:
    """Split a paragraph into sentence-based units of at most `max_tokens`."""
    if count_tokens(text) <= max_tokens:
        return [text]
    units: list[str] = []
    current = ""
    for sentence in SENTENCE_SPLIT_RE.split(text):
        if count_tokens(sentence) > max_tokens:
            # Very long sentence: fall back to word windows.
            words = sentence.split()
            piece: list[str] = []
            for word in words:
                piece.append(word)
                if count_tokens(" ".join(piece)) >= max_tokens:
                    units.append(" ".join(piece))
                    piece = []
            sentence = " ".join(piece)
            if not sentence:
                continue
        candidate = f"{current} {sentence}".strip()
        if current and count_tokens(candidate) > max_tokens:
            units.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        units.append(current)
    return units


def _tail(text: str, n_tokens: int) -> str:
    """Trailing words of `text` that add up to at least `n_tokens` tokens."""
    if n_tokens <= 0:
        return ""
    words = text.split()
    taken: list[str] = []
    total = 0
    for word in reversed(words):
        taken.append(word)
        total += count_tokens(word)
        if total >= n_tokens:
            break
    return " ".join(reversed(taken))


@dataclass
class _Unit:
    text: str
    page: int
    heading: str
    starts_section: bool


def chunk_document(
    doc: ParsedDocument,
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap_tokens: int = 80,
) -> list[Chunk]:
    if not 0 <= overlap_tokens < min_tokens <= max_tokens:
        raise ValueError("expected 0 <= overlap < min <= max")

    units: list[_Unit] = []
    fault_chunks: list[tuple[int, dict, str]] = []
    heading = ""
    pending_heading_start = True

    for page in doc.pages:
        for element in page.elements:
            if isinstance(element, TextBlock):
                if element.is_heading:
                    heading = element.text
                    pending_heading_start = True
                    continue
                for i, unit in enumerate(_split_units(element.text)):
                    prefix = f"{heading}\n" if pending_heading_start and heading and i == 0 else ""
                    units.append(_Unit(prefix + unit, page.number, heading, pending_heading_start and i == 0))
                pending_heading_start = False
            elif isinstance(element, Table):
                rows = fault_rows(element)
                if rows:
                    fault_chunks.extend((page.number, row, heading) for row in rows)
                else:
                    # Non-fault tables are kept as text rows so their content stays searchable.
                    lines = [" | ".join(element.header)] + [" | ".join(r) for r in element.rows]
                    for i, unit in enumerate(_split_units("\n".join(lines))):
                        units.append(_Unit(unit, page.number, heading, False))

    chunks: list[Chunk] = []

    def new_chunk(text: str, page: int, page_end: int, section: str, kind: str, code: str = "") -> None:
        seq = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{seq:04d}",
                doc_id=doc.doc_id,
                doc_title=doc.doc_title,
                page=page,
                page_end=page_end,
                section_heading=section,
                chunk_type=kind,
                text=text,
                fault_code=code,
            )
        )

    # Greedy packing of text units.
    buf: list[str] = []
    buf_tokens = 0
    buf_page = buf_end = 0
    buf_heading = ""

    def flush() -> str:
        nonlocal buf, buf_tokens
        text = "\n".join(buf).strip()
        if text:
            new_chunk(text, buf_page, buf_end, buf_heading, "text")
        buf, buf_tokens = [], 0
        return text

    for unit in units:
        unit_tokens = count_tokens(unit.text)
        if buf and unit.starts_section and buf_tokens >= min_tokens:
            flush()  # heading boundary, no overlap across sections
        elif buf and buf_tokens + unit_tokens > max_tokens:
            previous = flush()
            overlap = _tail(previous, overlap_tokens)
            if overlap and count_tokens(overlap) + unit_tokens <= max_tokens:
                buf, buf_tokens = [overlap], count_tokens(overlap)
                buf_page = buf_end  # overlap text comes from the end of the previous chunk
        if not buf:
            buf_page, buf_heading = unit.page, unit.heading
        elif not buf_heading:
            buf_heading = unit.heading
        buf.append(unit.text)
        buf_tokens = count_tokens("\n".join(buf))
        buf_end = unit.page
    flush()

    for page_number, row, section in fault_chunks:
        new_chunk(
            _fault_chunk_text(row, doc.doc_title, section),
            page_number,
            page_number,
            section,
            "fault_code",
            normalize_code(row["code"]),
        )
    return chunks
