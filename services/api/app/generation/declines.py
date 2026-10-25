"""Declines the model writes as prose instead of the INSUFFICIENT_EVIDENCE sentinel.

Models often open with "The sources do not mention X" and go on to describe the nearest
related provision, rather than replying with the sentinel. Only a first sentence about what the
sources say counts, and only one without a citation: "The agreement does not permit assignment
[2]" is an answer about what the agreement does, not a decline. Verbs about legal effect
(require, permit, impose) are left out for the same reason unless hedged with "explicitly".
"""

import re

_SUBJECT = (
    r"(?:the |these |this )?(?:provided |cited |given |available )?"
    r"(?:sources?|excerpts?|passages?|provisions?|documents?|text|agreement|contract)"
    r"(?: excerpts?| provided)?"
)
_INFO_VERB = (
    r"(?:mention|specify|state|describe|detail|contain|include|address|discuss|cover|identify"
    r"|define|set out|indicate|reference|say|disclose|provide (?:any|specific|information|details"
    r"|a provision))"
)
_DECLINE = re.compile(
    rf"^{_SUBJECT} (?:do|does) not (?:specifically |clearly |directly )?{_INFO_VERB}\b"
    rf"|^{_SUBJECT} (?:do|does) not explicitly \w+"
    rf"|^{_SUBJECT} (?:contains?|provides?|includes?|makes?) no\b",
    re.IGNORECASE,
)
_CITATION = re.compile(r"\[\d+(?:\]\[\d+)*\]")


def first_sentence(text: str) -> str:
    text = text.strip().lstrip("-*# ")
    return re.split(r"(?<=[.!?;])\s+|\n", text, maxsplit=1)[0]


def declines_in_prose(answer: str) -> bool:
    """True when the answer opens by saying the sources don't contain what was asked."""
    first = first_sentence(answer)
    return not _CITATION.search(first) and bool(_DECLINE.match(first))
