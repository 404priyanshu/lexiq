"""Per-client rate limits and the global daily answer budget for the public demo, in Redis.

Rate limits are fixed windows keyed by client IP (behind the demo's reverse proxy, uvicorn's
--proxy-headers makes request.client the real client). The budget counts model calls, not
requests: cache hits and retrieval-stage abstains never reach the generator, so they're free.
"""

import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import HTTPException, Request, status
from redis.asyncio import Redis

from app.generation.generator import GenerationError, Generator, GenEvent
from app.retrieval.retriever import RetrievedChunk

_WINDOWS = {
    "minute": (60, "Too many requests in a minute; wait a moment and try again."),
    "day": (86400, "This demo's daily limit for your address is reached; try again tomorrow."),
}
BUDGET_MESSAGE = (
    "Today's answer budget for this demo is used up, so only the retrieved passages are shown. "
    "Answers resume at midnight UTC."
)


async def _count(redis: Redis, key: str, window_s: int) -> int:
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, window_s)
    return count


def rate_limit(scope: str):
    """FastAPI dependency enforcing `<scope>_limit_per_minute` / `_per_day` from settings."""

    async def check(request: Request) -> None:
        s = request.app.state.settings
        client = request.client.host if request.client else "unknown"
        now = int(time.time())
        for window, (size, message) in _WINDOWS.items():
            limit = getattr(s, f"{scope}_limit_per_{window}", None)
            if not limit:
                continue
            key = f"lexiq:rl:{scope}:{window}:{now // size}:{client}"
            if await _count(request.app.state.redis, key, size) > limit:
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    message,
                    headers={"Retry-After": str(size - now % size)},
                )

    return check


class BudgetedGenerator:
    """Wraps a generator so at most `daily_limit` answers are generated per UTC day."""

    def __init__(self, inner: Generator, redis: Redis, daily_limit: int):
        self.inner = inner
        self.redis = redis
        self.daily_limit = daily_limit

    async def stream(self, question: str, sources: list[RetrievedChunk]) -> AsyncIterator[GenEvent]:
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        if await _count(self.redis, f"lexiq:budget:{day}", 2 * 86400) > self.daily_limit:
            raise GenerationError("daily answer budget reached", False, BUDGET_MESSAGE)
        async for event in self.inner.stream(question, sources):
            yield event
