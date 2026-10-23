"""Page-aware chunking by embedder tokens.

Pages are joined into one document text. That text is cut into units (paragraphs, then
sentences, then word windows for anything still too long), and units are packed greedily into
chunks of about `target` tokens, never more than `max_tokens`. Consecutive chunks share up to
`overlap` tokens of trailing units. Each chunk records its character span in the joined text
and the pages it covers.
"""

import bisect
import re
from collections.abc import Callable
from dataclasses import dataclass

PAGE_SEP = "\n\n"

TokenCounter = Callable[[list[str]], list[int]]

_PARAGRAPH = re.compile(r"\S(?:.*?\S)?(?=\n\n|\Z)", re.S)
# Sentence end: terminal punctuation, whitespace, then something that can start a sentence.
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[\"'(\[A-Z0-9§])")


@dataclass(frozen=True)
class TextChunk:
    ordinal: int
    content: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    token_count: int


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    tokens: int


def join_pages(pages: list[str]) -> tuple[str, list[int]]:
    """Join pages and return (text, start offset of each page)."""
    starts, pos = [], 0
    for page in pages:
        starts.append(pos)
        pos += len(page) + len(PAGE_SEP)
    return PAGE_SEP.join(pages), starts


def chunk_pages(
    pages: list[str],
    count_tokens: TokenCounter,
    target: int = 400,
    max_tokens: int = 480,
    overlap: int = 64,
) -> list[TextChunk]:
    if not 0 <= overlap < target <= max_tokens:
        raise ValueError("need 0 <= overlap < target <= max_tokens")

    text, page_starts = join_pages(pages)
    units = _units(text, count_tokens, target, max_tokens)

    def page_of(offset: int) -> int:
        return bisect.bisect_right(page_starts, offset)  # 1-based

    chunks: list[TextChunk] = []
    current: list[_Unit] = []
    tokens = 0

    def emit() -> None:
        start, end = current[0].start, current[-1].end
        chunks.append(
            TextChunk(
                ordinal=len(chunks),
                content=text[start:end],
                page_start=page_of(start),
                page_end=page_of(end - 1),
                char_start=start,
                char_end=end,
                token_count=tokens,
            )
        )

    for unit in units:
        if current and (tokens >= target or tokens + unit.tokens > max_tokens):
            emit()
            # Carry trailing units forward as overlap, as long as the next unit still fits.
            carry: list[_Unit] = []
            carried = 0
            for prev in reversed(current):
                if carried + prev.tokens > overlap:
                    break
                carry.insert(0, prev)
                carried += prev.tokens
            if carried + unit.tokens > max_tokens:
                carry, carried = [], 0
            current, tokens = carry, carried
        current.append(unit)
        tokens += unit.tokens

    # Skip a trailing chunk that is pure overlap (already fully inside the previous chunk).
    if current and not (chunks and current[-1].end <= chunks[-1].char_end):
        emit()
    return chunks


def _units(text: str, count_tokens: TokenCounter, target: int, max_tokens: int) -> list[_Unit]:
    spans = [(m.start(), m.end()) for m in _PARAGRAPH.finditer(text)]
    counts = count_tokens([text[s:e] for s, e in spans])

    out: list[_Unit] = []
    for (start, end), n in zip(spans, counts, strict=True):
        if n <= target:
            out.append(_Unit(start, end, n))
        else:
            out.extend(_split_sentences(text, start, end, count_tokens, max_tokens))
    return out


def _split_sentences(
    text: str, start: int, end: int, count_tokens: TokenCounter, max_tokens: int
) -> list[_Unit]:
    spans, pos = [], start
    for m in _SENTENCE_END.finditer(text, start, end):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, end))

    out: list[_Unit] = []
    for (s, e), n in zip(spans, count_tokens([text[s:e] for s, e in spans]), strict=True):
        if n <= max_tokens:
            out.append(_Unit(s, e, n))
        else:
            out.extend(_split_words(text, s, e, count_tokens, max_tokens))
    return out


def _split_words(
    text: str, start: int, end: int, count_tokens: TokenCounter, max_tokens: int
) -> list[_Unit]:
    """Halve at the whitespace nearest the middle until every piece fits."""
    (n,) = count_tokens([text[start:end]])
    if n <= max_tokens:
        return [_Unit(start, end, n)]
    mid = (start + end) // 2
    cut = text.rfind(" ", start, mid)
    if cut <= start:
        cut = text.find(" ", mid, end)
    if cut <= start or cut >= end:
        cut = mid  # no whitespace at all: hard split
        left_end, right_start = cut, cut
    else:
        left_end, right_start = cut, cut + 1
    return _split_words(text, start, left_end, count_tokens, max_tokens) + _split_words(
        text, right_start, end, count_tokens, max_tokens
    )
