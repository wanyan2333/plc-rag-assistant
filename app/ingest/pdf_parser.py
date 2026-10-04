"""PDF parsing with PyMuPDF.

Produces an ordered stream of text blocks and tables per page while
  * keeping 1-based page numbers,
  * removing running headers/footers (lines repeated on >50% of pages),
  * re-joining words hyphenated across line breaks,
  * flagging headings (larger or bold short lines) so the chunker can split on them,
  * excluding table regions from the text flow (tables are emitted separately).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz

HEADER_FOOTER_ZONE = 0.08  # top/bottom fraction of the page treated as header/footer area
REPEAT_THRESHOLD = 0.5  # a line must appear on more than this share of pages


@dataclass
class TextBlock:
    text: str
    is_heading: bool
    page: int
    y0: float = 0.0


@dataclass
class Table:
    page: int
    header: list[str]
    rows: list[list[str]]
    y0: float = 0.0


@dataclass
class ParsedPage:
    number: int  # 1-based
    elements: list[TextBlock | Table] = field(default_factory=list)

    @property
    def blocks(self) -> list[TextBlock]:
        return [e for e in self.elements if isinstance(e, TextBlock)]

    @property
    def tables(self) -> list[Table]:
        return [e for e in self.elements if isinstance(e, Table)]

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


@dataclass
class ParsedDocument:
    doc_id: str
    doc_title: str
    source_path: str
    pages: list[ParsedPage]


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    y0: float
    y1: float
    block_no: int


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "document"


def _clean_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    return _join_lines(cell.splitlines())


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines into a paragraph, merging words split by a trailing hyphen."""
    out = ""
    for raw in lines:
        line = re.sub(r"\s+", " ", raw.replace("\xad", "-")).strip()
        if not line:
            continue
        if not out:
            out = line
        elif re.search(r"[A-Za-z]-$", out) and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}"
    return out


def _repeat_key(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text)).strip().lower()


def _extract_lines(page: fitz.Page, table_rects: list[fitz.Rect]) -> list[_Line]:
    lines: list[_Line] = []
    data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    for block_no, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            bbox = fitz.Rect(line["bbox"])
            center = fitz.Point((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)
            if any(rect.contains(center) for rect in table_rects):
                continue
            text = "".join(s["text"] for s in spans)
            bold = all((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower() for s in spans)
            lines.append(
                _Line(
                    text=text,
                    size=max(s["size"] for s in spans),
                    bold=bold,
                    y0=bbox.y0,
                    y1=bbox.y1,
                    block_no=block_no,
                )
            )
    return lines


def _extract_tables(page: fitz.Page, page_number: int) -> tuple[list[Table], list[fitz.Rect]]:
    tables: list[Table] = []
    rects: list[fitz.Rect] = []
    try:
        found = page.find_tables()
    except Exception:  # table detection is best effort
        return tables, rects
    for tab in found.tables:
        rows = [[_clean_cell(c) for c in row] for row in tab.extract()]
        rows = [r for r in rows if any(r)]
        if len(rows) < 2:
            continue
        rects.append(fitz.Rect(tab.bbox))
        tables.append(Table(page=page_number, header=rows[0], rows=rows[1:], y0=tab.bbox[1]))
    return tables, rects


def _is_heading(lines: list[_Line], body_size: float) -> bool:
    text = " ".join(l.text.strip() for l in lines)
    if not text or len(text) > 100 or len(lines) > 2 or not re.search(r"[A-Za-z]{3,}", text):
        return False
    if text.endswith((".", ",", ";", ":")):
        return False
    larger = max(l.size for l in lines) >= body_size + 1.0
    bold = all(l.bold for l in lines)
    return larger or bold


def parse_pdf(path: str | Path) -> ParsedDocument:
    path = Path(path)
    doc = fitz.open(path)
    try:
        page_lines: list[list[_Line]] = []
        page_tables: list[list[Table]] = []
        page_heights: list[float] = []
        for index, page in enumerate(doc):
            tables, rects = _extract_tables(page, index + 1)
            page_tables.append(tables)
            page_lines.append(_extract_lines(page, rects))
            page_heights.append(page.rect.height)
        metadata_title = (doc.metadata or {}).get("title", "") or ""
    finally:
        doc.close()

    n_pages = len(page_lines)

    # 1. Detect running headers/footers: lines in the margin zones repeated on >50% of pages.
    def in_margin(line: _Line, height: float) -> bool:
        return line.y1 <= height * HEADER_FOOTER_ZONE or line.y0 >= height * (1 - HEADER_FOOTER_ZONE)

    counts: Counter[str] = Counter()
    for lines, height in zip(page_lines, page_heights):
        counts.update({_repeat_key(l.text) for l in lines if in_margin(l, height)})
    repeated = {k for k, c in counts.items() if n_pages >= 2 and c > REPEAT_THRESHOLD * n_pages}

    for i, (lines, height) in enumerate(zip(page_lines, page_heights)):
        page_lines[i] = [l for l in lines if not (in_margin(l, height) and _repeat_key(l.text) in repeated)]

    # 2. Body font size = the most common size weighted by characters.
    size_counter: Counter[float] = Counter()
    for lines in page_lines:
        for l in lines:
            size_counter[round(l.size, 1)] += len(l.text)
    body_size = size_counter.most_common(1)[0][0] if size_counter else 10.0

    # 3. Group lines into heading / paragraph blocks.
    pages: list[ParsedPage] = []
    for index, lines in enumerate(page_lines):
        number = index + 1
        blocks: list[TextBlock] = []
        groups: list[list[_Line]] = []
        for line in lines:
            if groups and groups[-1][-1].block_no == line.block_no:
                groups[-1].append(line)
            else:
                groups.append([line])

        for group in groups:
            # Split a pymupdf block where heading-style lines meet body lines.
            run: list[_Line] = []
            run_heading: bool | None = None
            for line in group:
                heading_like = _is_heading([line], body_size)
                if run and heading_like != run_heading:
                    blocks.append(_make_block(run, body_size, number))
                    run = []
                run.append(line)
                run_heading = heading_like
            if run:
                blocks.append(_make_block(run, body_size, number))

        elements: list[TextBlock | Table] = _merge_list_markers([b for b in blocks if b.text])
        elements.extend(page_tables[index])
        elements.sort(key=lambda e: e.y0)
        pages.append(ParsedPage(number=number, elements=elements))

    doc_title = metadata_title.strip()
    if not doc_title:
        first_heading = next(
            (b.text for p in pages[:1] for b in p.blocks if b.is_heading), None
        )
        doc_title = first_heading or path.stem.replace("_", " ")

    return ParsedDocument(
        doc_id=slugify(path.stem), doc_title=doc_title, source_path=str(path), pages=pages
    )


LIST_MARKER_RE = re.compile(r"^(?:\d{1,2}[.)]?|[a-z][.)]|[•●▪*-])$")


def _merge_list_markers(blocks: list[TextBlock]) -> list[TextBlock]:
    """Attach stand-alone list markers ("1", "•") to the list item text that follows."""
    merged: list[TextBlock] = []
    marker: TextBlock | None = None
    for block in blocks:
        if LIST_MARKER_RE.match(block.text):
            marker = block
            continue
        if marker is not None and not block.is_heading:
            label = marker.text.rstrip(".)")
            prefix = f"{label}." if label.isalnum() else "-"
            block = TextBlock(f"{prefix} {block.text}", False, block.page, marker.y0)
        marker = None
        merged.append(block)
    return merged


def _make_block(lines: list[_Line], body_size: float, page: int) -> TextBlock:
    return TextBlock(
        text=_join_lines([l.text for l in lines]),
        is_heading=_is_heading(lines, body_size),
        page=page,
        y0=lines[0].y0,
    )
