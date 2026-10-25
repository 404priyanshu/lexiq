"""Query focusing for two-stage retrieval.

A question that names its document ("In the Supply Agreement between Acme Corp. and Globex
LLC, who owns the IP?") pulls every retriever toward the document's preamble, the one chunk
that repeats the names. After stage 1 has picked the document, stage 2 searches inside it with
the identifying words removed. The identifying words are found without NER: capitalized query
words (not the first word) that also appear in the document's opening chunks.

The number of words removed doubles as the routing gate: a question that names nothing in the
document ("Which contracts have a non-compete?") is never routed to a single document.
"""

import re

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'.-]*")
# Words left dangling once a "In the <name> between <party> and <party>," prefix is removed.
_FILLER = {
    "in", "under", "for", "of", "the", "a", "an", "between", "with", "and", "by", "to", "from",
    "number", "no", "dated", "&",
}  # fmt: skip


def _norm(word: str) -> str:
    return word.lower().strip(".,;:()'\"")


def focus_query(query: str, preamble: str) -> tuple[str, int]:
    """Return (query without the document-identifying words, number of words removed)."""
    vocabulary = {_norm(w) for w in _WORD.findall(preamble)}
    tokens = query.split()
    kept, removed = [], 0
    for i, token in enumerate(tokens):
        core = token.strip(",.;:()\"'?")
        if i > 0 and core[:1].isupper() and _norm(core) in vocabulary:
            removed += 1
            continue
        kept.append(token)

    # Drop the leading filler/numbers the removed name leaves behind ("In the between and ,").
    while kept and (_norm(kept[0]) in _FILLER or _norm(kept[0]).isdigit() or not _norm(kept[0])):
        kept.pop(0)
    focused = " ".join(kept).lstrip(" ,;:")
    focused = re.sub(r"\s+([,?.;:])", r"\1", focused)
    if focused:
        focused = focused[0].upper() + focused[1:]
    return (focused or query), removed
