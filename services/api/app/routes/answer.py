import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.generation.service import answer_events
from app.limits import rate_limit
from app.routes.search import SearchRequest

router = APIRouter(tags=["answer"])
log = logging.getLogger(__name__)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/answer", dependencies=[Depends(rate_limit("answer"))])
async def answer(request: Request, body: SearchRequest) -> StreamingResponse:
    """Server-sent events; see app.generation.service for the event sequence."""
    retriever = getattr(request.app.state, "retriever", None)
    generator = getattr(request.app.state, "generator", None)
    if retriever is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "retrieval models not loaded")
    if generator is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "No LLM API key is set (ANTHROPIC_API_KEY or OPENAI_API_KEY)",
        )

    top_k = body.top_k
    settings = request.app.state.settings
    if settings.read_only:
        # More passages means a longer, costlier prompt; the public demo sticks to the default.
        top_k = min(top_k or settings.top_k, settings.top_k)

    async def stream():
        try:
            async for event, data in answer_events(
                retriever,
                generator,
                body.query,
                top_k,
                body.document_ids,
                cache=getattr(request.app.state, "cache", None),
            ):
                yield _sse(event, data)
        except Exception:
            # Headers are already sent, so report failures in-band.
            log.exception("answer stream failed")
            yield _sse("error", {"message": "Something went wrong.", "retryable": True})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
