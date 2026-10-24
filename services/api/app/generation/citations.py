"""Post-check of [n] citations in a generated answer."""

import re
from dataclasses import dataclass

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# A sentence runs to terminal punctuation plus any citations right after it ("... [2]."
# and "... . [2]" both count), followed by whitespace or the end of the text.
_SENTENCE = re.compile(r".+?(?:[.!?]+(?:\s*\[\d+(?:\s*,\s*\d+)*\])*(?=\s|$)|$)", re.S)


@dataclass(frozen=True)
class CitationCheck:
    cited: list[int]  # valid source numbers used, sorted
    invalid: list[int]  # numbers that don't match any source
    sentences: int
    cited_sentences: int

    @property
    def coverage(self) -> float:
        """Share of sentences carrying at least one citation."""
        return self.cited_sentences / self.sentences if self.sentences else 0.0


def citation_numbers(text: str) -> list[int]:
    return [int(n) for m in _CITATION.finditer(text) for n in m.group(1).split(",")]


def check_citations(answer: str, num_sources: int) -> CitationCheck:
    numbers = set(citation_numbers(answer))
    sentences = [s for s in (m.group(0).strip() for m in _SENTENCE.finditer(answer)) if s]
    return CitationCheck(
        cited=sorted(n for n in numbers if 1 <= n <= num_sources),
        invalid=sorted(n for n in numbers if not 1 <= n <= num_sources),
        sentences=len(sentences),
        cited_sentences=sum(1 for s in sentences if _CITATION.search(s)),
    )
