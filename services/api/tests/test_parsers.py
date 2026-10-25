import pymupdf
import pytest

from app.ingest.parsers import UnsupportedFormat, mime_for, normalize, parse


def make_pdf(pages: list[str], title: str | None = None) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 540, 720), text)
    if title:
        doc.set_metadata({"title": title})
    return doc.tobytes()


def test_pdf_keeps_page_boundaries_and_title():
    data = make_pdf(["First page about indemnity.", "", "Third page on governing law."], "MSA")
    parsed = parse(data, "application/pdf")

    assert parsed.title == "MSA"
    # Blank pages are kept so page numbers match the PDF.
    assert parsed.pages == ["First page about indemnity.", "", "Third page on governing law."]


def test_html_splits_on_print_page_breaks_and_drops_hidden_content():
    html = b"""
    <html><head><title>10-K</title><style>p{color:red}</style></head><body>
      <div style="display:none"><ix:header>hidden xbrl facts</ix:header></div>
      <p>Item 1. <span>Business</span> overview.</p>
      <p>Revenue grew<br>in 2025.</p>
      <hr style="page-break-after: always">
      <table><tr><td>Net income</td><td>$10</td></tr></table>
      <div style="break-before: page"><p>Risk factors.</p></div>
      <script>alert(1)</script>
    </body></html>
    """
    parsed = parse(html, "text/html")

    assert parsed.title == "10-K"
    assert parsed.pages == [
        "Item 1. Business overview.\n\nRevenue grew in 2025.",
        "Net income $10",
        "Risk factors.",
    ]


def test_markdown_title_from_first_heading():
    parsed = parse(
        b"# Lease Agreement\n\nTerm is five\nyears.\n\n\nRent is due monthly.", "text/markdown"
    )

    assert parsed.title == "Lease Agreement"
    assert parsed.pages == ["# Lease Agreement\n\nTerm is five years.\n\nRent is due monthly."]


def test_normalize_joins_wrapped_lines_and_keeps_paragraphs():
    raw = "Line one\r\ncontinues  here.\n \n\nNext\u00a0para\t\tgraph."
    assert normalize(raw) == "Line one continues here.\n\nNext para graph."


def test_mime_for_rejects_unknown_types():
    assert mime_for("Contract.PDF") == "application/pdf"
    with pytest.raises(UnsupportedFormat):
        mime_for("archive.zip")


def test_normalize_rejoins_hyphenated_words_and_collapses_dot_leaders():
    raw = "Judg-\nment in cases of impeach-\nment. Anglo-\nSaxon law.\nPowers . . . . . . 12"
    assert normalize(raw) == "Judgment in cases of impeachment. Anglo- Saxon law. Powers ... 12"


def test_running_headers_and_footers_are_stripped():
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    pages = [
        f"CONSTITUTION OF THE UNITED STATES {n}\n"
        + "\n".join(f"Body {words[n - 1]} line {i}." for i in range(8))
        + f"\nPage {n}"
        for n in range(1, 7)
    ]
    data = make_pdf(pages)
    parsed = parse(data, "application/pdf")

    assert parsed.pages[2] == " ".join(f"Body charlie line {i}." for i in range(8))


def _two_column_pdf() -> bytes:
    """A CFR-like page: full-width header, two columns of two paragraphs, page number."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(72, 60, 540, 80), "TITLE 16 COMMERCIAL PRACTICES")
    para = "{} " + "This paragraph carries enough words to look like real regulatory text. " * 3
    for x0, x1, label in ((72, 296, "LEFT"), (316, 540, "RIGHT")):
        page.insert_textbox(pymupdf.Rect(x0, 100, x1, 300), para.format(f"{label}-ONE"))
        page.insert_textbox(pymupdf.Rect(x0, 320, x1, 520), para.format(f"{label}-TWO"))
    page.insert_textbox(pymupdf.Rect(286, 740, 326, 760), "500")
    return doc.tobytes()


def test_two_column_pages_read_one_column_then_the_other():
    text = parse(_two_column_pdf(), "application/pdf").pages[0]
    order = [text.index(m) for m in ("TITLE 16", "LEFT-ONE", "LEFT-TWO", "RIGHT-ONE", "RIGHT-TWO")]
    assert order == sorted(order)
    # Sorting by position alone would splice the columns line by line.
    assert "LEFT-ONE This paragraph carries enough words" in text


def test_side_by_side_short_blocks_keep_the_default_order():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(72, 100, 290, 160), "ACME CORP.\nBy: Alice Smith")
    page.insert_textbox(pymupdf.Rect(320, 100, 540, 160), "GLOBEX LLC\nBy: Bob Jones")
    flags = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_DEHYPHENATE
    default = normalize(page.get_text("text", flags=flags, sort=True))
    assert parse(doc.tobytes(), "application/pdf").pages[0] == default
