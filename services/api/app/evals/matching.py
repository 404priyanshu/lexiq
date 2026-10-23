"""Decide whether a retrieved chunk contains a golden evidence span.

Gold spans come from CUAD's plain-text contracts while chunks come from our PDF parser, so
the two differ in whitespace, hyphenation, punctuation and line breaks. Both sides are reduced
to lowercase alphanumeric words. A chunk matches when it contains the span, when it holds at
least half of the span as one run of words (a span cut at a chunk boundary), or when nearly all
of it lies inside the span (a span longer than a chunk).
"""

import math
import re
from functools import lru_cache

MIN_SHARED_WORDS = 7
MIN_SHARED_RATIO = 0.5
CHUNK_INSIDE_RATIO = 0.9

_MOD = (1 << 61) - 1
_BASE = 1_000_003


@lru_cache(maxsize=65536)
def words(text: str) -> tuple[str, ...]:
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # rejoin line-break hyphenation
    return tuple(re.findall(r"[a-z0-9]+", text.lower()))


def _ngram_hashes(ws: tuple[str, ...], n: int) -> set[int]:
    """Rolling hashes of every run of n consecutive words, in O(len) time."""
    if n > len(ws):
        return set()
    ids = [hash(w) % _MOD for w in ws]
    top = pow(_BASE, n - 1, _MOD)
    h = 0
    for x in ids[:n]:
        h = (h * _BASE + x) % _MOD
    out = {h}
    for i in range(n, len(ids)):
        h = ((h - ids[i - n] * top) * _BASE + ids[i]) % _MOD
        out.add(h)
    return out


def shares_run(a: tuple[str, ...], b: tuple[str, ...], n: int) -> bool:
    """True if a and b share a run of at least n consecutive words."""
    if n <= 0:
        return True
    if n > min(len(a), len(b)):
        return False
    return not _ngram_hashes(a, n).isdisjoint(_ngram_hashes(b, n))


@lru_cache(maxsize=262144)
def span_in_chunk(span: str, chunk: str) -> bool:
    s, c = words(span), words(chunk)
    if not s or not c:
        return False
    if " ".join(s) in " ".join(c):
        return True
    # Either at least half the span sits in this chunk (span cut at a chunk boundary), or the
    # chunk is almost entirely inside the span (span longer than a chunk). A short chunk that
    # merely shares boilerplate ("the laws of the State of ...") satisfies neither.
    half_span = max(MIN_SHARED_WORDS, math.ceil(MIN_SHARED_RATIO * len(s)))
    most_of_chunk = max(MIN_SHARED_WORDS, math.ceil(CHUNK_INSIDE_RATIO * len(c)))
    return shares_run(s, c, half_span) or shares_run(s, c, most_of_chunk)


def first_hit(evidence: list[str], chunks: list[str]) -> int | None:
    """1-based rank of the first chunk containing any evidence span, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if any(span_in_chunk(span, chunk) for span in evidence):
            return rank
    return None
