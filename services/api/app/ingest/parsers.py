"""Turn source files into a list of page texts. Page numbers drive citations, so parsers keep
real page boundaries where the format has them (PDF pages, HTML print page breaks)."""

import re
from dataclasses import dataclass
from pathlib import PurePath

import pymupdf
from bs4 import BeautifulSoup, NavigableString

MIME_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".html": "text/html",
    ".htm": "text/html",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}

PAGE_BREAK = "\f"


class UnsupportedFormat(ValueError):
    pass


@dataclass
class ParsedDocument:
    title: str | None
    pages: list[str]


def mime_for(filename: str) -> str:
    suffix = PurePath(filename).suffix.lower()
    try:
        return MIME_BY_SUFFIX[suffix]
    except KeyError:
        raise UnsupportedFormat(f"unsupported file type {suffix!r}") from None


def parse(data: bytes, mime_type: str) -> ParsedDocument:
    match mime_type:
        case "application/pdf":
            return parse_pdf(data)
        case "text/html":
            return parse_html(data)
        case "text/markdown" | "text/plain":
            return parse_text(data)
    raise UnsupportedFormat(f"unsupported mime type {mime_type!r}")


def parse_pdf(data: bytes) -> ParsedDocument:
    flags = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_DEHYPHENATE
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        raw = [_page_text(page, flags) for page in doc]
        title = (doc.metadata or {}).get("title") or None
    return ParsedDocument(title=title, pages=[normalize(p) for p in strip_running_lines(raw)])


# Blocks within this share of the page width of the middle still count as one column's.
_GUTTER_TOLERANCE = 0.02
# A band is two-column text only with this much on each side; less is a signature block, a
# key/value header or a table, which the default extraction handles no worse.
_COLUMN_MIN_BLOCKS = 2
_COLUMN_MIN_CHARS = 300


def _page_text(page: pymupdf.Page, flags: int) -> str:
    """Page text in reading order, column by column on two-column pages.

    Sorting by position (PyMuPDF's sort=True) interleaves the columns of a two-column page,
    the CFR's layout, splicing unrelated sentences together. Here, blocks that span the middle
    of the page (headers, page numbers, full-width paragraphs) cut the page into bands, and a
    band with substantial text on both sides of the middle is read left column first, then
    right. Pages without such a band keep the default extraction unchanged.
    """
    blocks = [b for b in page.get_text("blocks", flags=flags) if b[4].strip()]
    mid = page.rect.width / 2
    tol = page.rect.width * _GUTTER_TOLERANCE

    def side(b) -> str:
        if b[2] <= mid + tol:
            return "left"
        if b[0] >= mid - tol:
            return "right"
        return "full"

    bands: list[list] = [[]]
    for b in sorted(blocks, key=lambda b: (b[1], b[0])):
        if side(b) == "full":
            bands.append([b])
            bands.append([])
        else:
            bands[-1].append(b)

    def two_column(band: list) -> bool:
        return all(
            len(col) >= _COLUMN_MIN_BLOCKS and sum(len(b[4]) for b in col) >= _COLUMN_MIN_CHARS
            for col in ([b for b in band if side(b) == s] for s in ("left", "right"))
        )

    if not any(two_column(band) for band in bands):
        return page.get_text("text", flags=flags, sort=True)

    ordered = []
    for band in bands:
        key = (lambda b: (side(b) == "right", b[1], b[0])) if two_column(band) else None
        ordered.extend(sorted(band, key=key or (lambda b: (b[1], b[0]))))
    # Single newlines: a paragraph often runs on from the foot of the left column to the top
    # of the right, and normalize() rejoins those lines (and hyphenated words).
    return "\n".join(b[4].rstrip("\n") for b in ordered)


def strip_running_lines(pages: list[str], edge: int = 3, min_share: float = 0.3) -> list[str]:
    """Drop running headers/footers: lines near the top or bottom of a page that repeat on many
    pages once digits are masked (so "Page 4" and "Page 5" count as the same line)."""
    if len(pages) < 4:
        return pages

    def key(line: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.strip().lower()))

    lines = [[ln for ln in p.split("\n") if ln.strip()] for p in pages]
    counts: dict[str, int] = {}
    for page_lines in lines:
        for k in {key(ln) for ln in page_lines[:edge] + page_lines[-edge:]}:
            counts[k] = counts.get(k, 0) + 1
    running = {k for k, n in counts.items() if n >= min_share * len(pages)}
    if not running:
        return pages

    out = []
    for page_lines in lines:
        n = len(page_lines)
        kept = [
            ln
            for i, ln in enumerate(page_lines)
            if not ((i < edge or i >= n - edge) and key(ln) in running)
        ]
        out.append("\n".join(kept))
    return out


BLOCK_TAGS = [
    "p", "div", "section", "article", "li", "tr", "table", "blockquote", "pre",
    "h1", "h2", "h3", "h4", "h5", "h6",
]  # fmt: skip
_PAGE_BREAK_STYLE = re.compile(r"(page-)?break-(before|after)\s*:\s*(always|page)", re.I)
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.I)


def parse_html(data: bytes) -> ParsedDocument:
    soup = BeautifulSoup(data, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else None

    for el in soup(["script", "style", "head", "noscript", "template"]):
        el.decompose()
    # Inline XBRL (SEC filings) keeps machine-readable facts in hidden blocks.
    for el in soup.find_all(style=_HIDDEN_STYLE):
        el.decompose()

    # Print page breaks (common in SEC filings) become real page boundaries.
    for el in soup.find_all(style=_PAGE_BREAK_STYLE):
        m = _PAGE_BREAK_STYLE.search(el["style"])
        if m.group(2).lower() == "before":
            el.insert_before(NavigableString(PAGE_BREAK))
        else:
            el.insert_after(NavigableString(PAGE_BREAK))

    for br in soup.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for td in soup.find_all(["td", "th"]):
        td.append(NavigableString(" "))
    for el in soup.find_all(BLOCK_TAGS):
        el.append(NavigableString("\n\n"))

    text = soup.get_text("")
    pages = [normalize(p) for p in text.split(PAGE_BREAK)]
    return ParsedDocument(title=title, pages=_drop_empty_pages(pages))


def parse_text(data: bytes) -> ParsedDocument:
    text = data.decode("utf-8", errors="replace")
    heading = re.search(r"^#\s+(.+)$", text, re.M)
    pages = [normalize(p) for p in text.split(PAGE_BREAK)]
    return ParsedDocument(
        title=heading.group(1).strip() if heading else None, pages=_drop_empty_pages(pages)
    )


def normalize(text: str) -> str:
    """Collapse layout whitespace: single newlines are line wraps, blank lines are paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    # Dot leaders from tables of contents and indexes.
    text = re.sub(r"(?:\. ?){4,}", " ... ", text)
    text = re.sub(r"[ \t\u2000-\u200b]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    # Line-break hyphenation ("judg-\nment"). Only before a lowercase letter, so compounds
    # split at a line end before a capital ("Anglo-\nSaxon") keep their hyphen.
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
    paragraphs = re.split(r"\n{2,}", text)
    return "\n\n".join(p.replace("\n", " ").strip() for p in paragraphs if p.strip())


def _drop_empty_pages(pages: list[str]) -> list[str]:
    # Only for formats with synthetic breaks; PDF keeps blank pages so numbering matches.
    return [p for p in pages if p] or [""]
