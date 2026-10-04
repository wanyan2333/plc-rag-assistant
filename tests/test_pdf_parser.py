from app.ingest.pdf_parser import _join_lines, slugify
from tests.fixtures.make_fixture_pdf import FIXTURE_TITLE, FOOTER, HEADER


def test_page_numbers_are_one_based_and_complete(parsed_doc):
    assert [p.number for p in parsed_doc.pages] == [1, 2, 3, 4]


def test_content_lands_on_the_right_page(parsed_doc):
    page_of = {}
    for page in parsed_doc.pages:
        for block in page.blocks:
            if block.is_heading:
                page_of[block.text] = page.number
    assert page_of["1 Introduction"] == 1
    assert page_of["2 LED Status Indicators"] == 2
    assert page_of["3 Fault Codes"] == 3
    assert page_of["4 Troubleshooting Procedures"] == 4


def test_running_header_and_footer_are_removed(parsed_doc):
    for page in parsed_doc.pages:
        text = page.text
        assert HEADER not in text
        assert FOOTER not in text
        assert f"Page {page.number}" not in text


def test_document_metadata(parsed_doc):
    assert parsed_doc.doc_title == FIXTURE_TITLE
    assert parsed_doc.doc_id == "fx100-manual"


def test_hyphenated_line_break_is_merged(parsed_doc):
    assert "monitors the internal bus" in parsed_doc.pages[0].text


def test_tables_are_extracted_separately(parsed_doc):
    tables = parsed_doc.pages[2].tables
    assert len(tables) == 1
    assert tables[0].header == ["Code", "Description", "Cause", "Remedy"]
    assert tables[0].rows[0][0] == "E-101"
    # Table cells must not leak into the running text.
    assert "Overcurrent on axis 1" not in parsed_doc.pages[2].text


def test_join_lines_handles_hyphenation_but_keeps_real_hyphens():
    assert _join_lines(["over-", "current fault"]) == "overcurrent fault"
    assert _join_lines(["see code E-", "101"]) == "see code E- 101"  # digits are not merged
    assert _join_lines(["lock-", "Out procedure"]) == "lock- Out procedure"


def test_slugify():
    assert slugify("S7-1200 System Manual") == "s7-1200-system-manual"
