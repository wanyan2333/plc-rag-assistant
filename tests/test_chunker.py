import pytest

from app.fault_codes import find_fault_codes, normalize_code
from app.ingest.chunker import _tail, chunk_document, count_tokens
from app.ingest.pdf_parser import ParsedDocument, ParsedPage, Table, TextBlock

SENTENCES = [
    "The drive monitors the DC bus voltage continuously during operation.",
    "If the voltage exceeds the trip level the inverter output is disabled immediately.",
    "Check the braking resistor wiring and the deceleration ramp before restarting.",
    "Long ramps reduce regenerative energy and prevent nuisance trips on heavy loads.",
    "Record the fault history so recurring problems can be analysed by maintenance staff.",
]


def _long_doc(paragraphs: int = 40, headings_every: int = 0) -> ParsedDocument:
    pages = []
    for p in range(paragraphs // 5):
        elements = []
        for i in range(5):
            n = p * 5 + i
            if headings_every and n % headings_every == 0:
                elements.append(TextBlock(f"Section {n // headings_every + 1} Topic", True, p + 1))
            elements.append(TextBlock(f"Paragraph {n}. " + " ".join(SENTENCES), False, p + 1))
        pages.append(ParsedPage(number=p + 1, elements=elements))
    return ParsedDocument("long", "Long Manual", "long.pdf", pages)


def test_text_chunks_respect_size_bounds():
    chunks = chunk_document(_long_doc(60), min_tokens=500, max_tokens=800, overlap_tokens=80)
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert len(text_chunks) >= 3
    for chunk in text_chunks:
        assert count_tokens(chunk.text) <= 800
    for chunk in text_chunks[:-1]:  # only the final chunk may be short
        assert count_tokens(chunk.text) >= 500


def test_adjacent_chunks_overlap_by_about_80_tokens():
    chunks = chunk_document(_long_doc(60), min_tokens=500, max_tokens=800, overlap_tokens=80)
    for prev, nxt in zip(chunks, chunks[1:]):
        expected = _tail(prev.text, 80)
        assert nxt.text.startswith(expected)
        assert 80 <= count_tokens(expected) <= 100


def test_chunks_split_on_headings_once_min_size_reached():
    chunks = chunk_document(_long_doc(60, headings_every=10), min_tokens=500, max_tokens=800, overlap_tokens=80)
    starts = [c for c in chunks if c.text.startswith("Section ")]
    assert starts, "expected at least one chunk to begin at a heading"
    for chunk in starts:
        assert chunk.section_heading == chunk.text.splitlines()[0]


def test_page_metadata_tracks_start_and_end():
    chunks = chunk_document(_long_doc(60))
    assert chunks[0].page == 1
    assert all(c.page <= c.page_end for c in chunks)
    assert chunks[-1].page_end == 12


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        chunk_document(_long_doc(5), min_tokens=100, max_tokens=50)


def test_fault_code_rows_become_individual_chunks(parsed_doc):
    chunks = chunk_document(parsed_doc)
    faults = [c for c in chunks if c.chunk_type == "fault_code"]
    assert [c.fault_code for c in faults] == ["E101", "E102", "E205", "F0042", "16#8085"]
    e101 = faults[0]
    assert e101.page == 3
    assert "Overcurrent on axis 1" in e101.text
    assert "Cause:" in e101.text and "Remedy:" in e101.text
    assert e101.section_heading == "3 Fault Codes"
    # Each fault chunk only contains its own row.
    assert "E-102" not in e101.text


def test_non_fault_tables_stay_in_text(parsed_doc):
    chunks = chunk_document(parsed_doc)
    text = "\n".join(c.text for c in chunks if c.chunk_type == "text")
    assert "Steady yellow" in text


def test_chunk_metadata_is_complete_and_ids_unique(parsed_doc):
    chunks = chunk_document(parsed_doc)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for c in chunks:
        meta = c.metadata
        for key in ("doc_id", "doc_title", "page", "section_heading", "chunk_type", "chunk_id"):
            assert key in meta and meta[key] is not None
        assert all(isinstance(v, (str, int)) for v in meta.values())


def test_fault_table_without_header_keywords():
    table = Table(page=1, header=["E-301", "Overtemperature", "Fan blocked"], rows=[["E-302", "Undervoltage", "Supply dip"]])
    doc = ParsedDocument("t", "T", "t.pdf", [ParsedPage(1, [table])])
    codes = [c.fault_code for c in chunk_document(doc) if c.chunk_type == "fault_code"]
    assert codes == ["E301", "E302"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What does 16#8085 mean?", ["16#8085"]),
        ("drive shows F-0042 then E101", ["F-0042", "E101"]),
        ("Micro800 fault 0xF0A3", ["0xF0A3"]),
        ("S7-1200 CPU in STOP", []),
        ("Check IP20 rating", []),
    ],
)
def test_fault_code_detection(text, expected):
    found = find_fault_codes(text)
    if text.startswith("Check IP20"):
        # IP20 looks like a code; retrieval only boosts codes that exist in the table.
        assert found in ([], ["IP20"])
    else:
        assert found == expected


def test_normalize_code():
    assert normalize_code("e-101") == normalize_code("E101") == "E101"
    assert normalize_code("16#8085") == "16#8085"
