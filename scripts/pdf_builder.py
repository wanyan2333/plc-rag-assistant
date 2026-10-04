"""Tiny helper for generating manual-like PDFs with reportlab.

Used by the test fixture and by scripts/generate_demo_manuals.py. A manual is a list of
elements:
    ("h1", "3 Fault Codes")           section heading
    ("h2", "3.1 Axis faults")         sub-heading
    ("p", "Paragraph text ...")       paragraph (reportlab markup allowed, e.g. <br/>)
    ("steps", ["step one", ...])      numbered list
    ("table", header, rows)           ruled table
    ("break",)                        page break
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def build_manual_pdf(
    path: str | Path,
    title: str,
    elements: list[tuple],
    header_text: str,
    footer_text: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=13)
    cell = ParagraphStyle("Cell", parent=body, fontSize=8.5, leading=10.5)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16)

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        width, height = A4
        canvas.drawString(20 * mm, height - 12 * mm, header_text)
        canvas.drawRightString(width - 20 * mm, 10 * mm, f"{footer_text} | Page {doc.page}")
        canvas.restoreState()

    story = []
    for element in elements:
        kind = element[0]
        if kind == "h1":
            story.append(Paragraph(element[1], h1))
        elif kind == "h2":
            story.append(Paragraph(element[1], h2))
        elif kind == "p":
            story.append(Paragraph(element[1], body))
            story.append(Spacer(1, 4))
        elif kind == "steps":
            items = [ListItem(Paragraph(s, body)) for s in element[1]]
            story.append(ListFlowable(items, bulletType="1", start="1"))
            story.append(Spacer(1, 4))
        elif kind == "table":
            header, rows = element[1], element[2]
            data = [[Paragraph(f"<b>{h}</b>", cell) for h in header]]
            data += [[Paragraph(str(v), cell) for v in row] for row in rows]
            col_width = (A4[0] - 40 * mm) / len(header)
            table = Table(data, colWidths=[col_width] * len(header), repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 8))
        elif kind == "break":
            story.append(PageBreak())
        else:
            raise ValueError(f"unknown element {kind!r}")

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=title,
        author="Synthetic manual generator",
        topMargin=22 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return path
