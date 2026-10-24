"""Question -> retrieve -> stream a cited answer, as a sequence of named events.

Events, in order:
  sources  the passages given to the model, numbered as the model sees them
  delta    answer text as it streams (zero or more)
  notice   a model fallback happened mid-answer (rare; informational)
  done     final answer, citation check, abstain/refusal flags, usage, timings
  error    replaces `done` if generation fails
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic

from app.generation.citations import check_citations
from app.generation.generator import Completion, FallbackNotice, Generator, TextDelta
from app.generation.prompt import INSUFFICIENT
from app.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

Event = tuple[str, dict[str, Any]]


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


async def answer_events(
    retriever: Retriever,
    generator: Generator,
    question: str,
    top_k: int | None = None,
    document_ids: Sequence[uuid.UUID] | None = None,
) -> AsyncIterator[Event]:
    started = time.perf_counter()
    result = await retriever.retrieve(question, top_k, document_ids)
    sources = result.chunks

    yield (
        "sources",
        {
            "sources": [
                {
                    "n": n,
                    "chunk_id": c.chunk_id,
                    "document_id": str(c.document_id),
                    "document_title": c.document_title,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "relevance": round(c.relevance, 4),
                    "content": c.content,
                }
                for n, c in enumerate(sources, start=1)
            ]
        },
    )

    done: dict[str, Any] = {
        "answer": "",
        "abstained": False,
        "abstain_reason": None,
        "refused": False,
        "truncated": False,
        "citations": None,
        "model": None,
        "usage": None,
        "timings_ms": {"retrieval": result.timings_ms},
    }

    if result.abstained:
        # Not enough evidence to bother the model; the closest passages are still shown.
        done |= {"abstained": True, "abstain_reason": f"retrieval: {result.abstain_reason}"}
        done["timings_ms"]["total"] = _ms(started)
        yield "done", done
        return

    gen_started = time.perf_counter()
    text: list[str] = []
    # Hold back the start of the answer until we know it isn't the INSUFFICIENT sentinel,
    # so the sentinel never reaches the client.
    pending, decided = "", False
    completion: Completion | None = None
    try:
        async for event in generator.stream(question, sources):
            if isinstance(event, TextDelta):
                if "ttft" not in done["timings_ms"]:
                    done["timings_ms"]["ttft"] = _ms(started)
                text.append(event.text)
                if decided:
                    yield "delta", {"text": event.text}
                    continue
                pending += event.text
                stripped = pending.lstrip()
                if len(stripped) >= len(INSUFFICIENT) or not INSUFFICIENT.startswith(stripped):
                    decided = True
                    if not stripped.startswith(INSUFFICIENT):
                        yield "delta", {"text": pending}
            elif isinstance(event, FallbackNotice):
                yield "notice", {"fallback": {"from": event.from_model, "to": event.to_model}}
            elif isinstance(event, Completion):
                completion = event
    except anthropic.APIError as exc:
        log.warning("generation failed: %s", exc)
        retryable = isinstance(exc, anthropic.RateLimitError | anthropic.APIConnectionError) or (
            isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500
        )
        yield (
            "error",
            {"message": "The answer model is unavailable right now.", "retryable": retryable},
        )
        return

    # A whole answer shorter than the sentinel (and a prefix of it) is still held back.
    if not decided and pending.strip() and not pending.lstrip().startswith(INSUFFICIENT):
        yield "delta", {"text": pending}

    answer = "".join(text).strip()
    done["timings_ms"]["generation"] = _ms(gen_started)
    done["timings_ms"]["total"] = _ms(started)
    if completion:
        done["model"] = completion.model
        done["usage"] = {
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
        }

    if completion and completion.stop_reason == "refusal":
        # Every model in the fallback chain declined; any partial text is not an answer.
        done["refused"] = True
    elif answer.startswith(INSUFFICIENT):
        done |= {"abstained": True, "abstain_reason": "model: sources don't answer the question"}
    else:
        check = check_citations(answer, len(sources))
        done |= {
            "answer": answer,
            "truncated": bool(completion and completion.stop_reason == "max_tokens"),
            "citations": {
                "cited": check.cited,
                "invalid": check.invalid,
                "coverage": round(check.coverage, 3),
            },
        }
    yield "done", done
