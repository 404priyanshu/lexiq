"""Example questions for the public demo, and a command that warms their answers in the cache.

The examples are chosen to retrieve their evidence first on the 52,650-page index (checked
with the retriever, not assumed), across both corpora. Warming runs each through the normal
answer path with the cache on, so a visitor clicking one gets the cached answer: instant, and
no model call. The cache keys on corpus version and config, so warm entries stay valid until
the next ingest or settings change; rerun this after either.

Usage:
  uv run python -m app.demo warm
"""

import argparse
import asyncio

from redis.asyncio import Redis

from app.cache import QueryCache
from app.config import get_settings
from app.db import engine
from app.generation.generator import make_generator
from app.generation.service import answer_events
from app.retrieval.retriever import Retriever

EXAMPLES = [
    "How long must employers keep payroll records under the Fair Labor Standards Act?",
    "Who qualifies as an accredited investor under Regulation D?",
    "When is a gift to a spouse excluded from the federal gift tax?",
    "What is the de minimis fringe benefit exclusion?",
    "In the Supply Agreement between Bellicum Pharmaceuticals, Inc. and Miltenyi Biotec GmbH, "
    "which jurisdiction's law governs the agreement?",
    "In the Intellectual Property Agreement between AHF Holding, Inc. and AFI Licensing LLC, "
    "what license does the agreement grant, and to whom?",
]


async def warm() -> None:
    s = get_settings()
    generator = make_generator(s)
    if generator is None:
        raise SystemExit("warming needs ANTHROPIC_API_KEY or OPENAI_API_KEY")
    redis = Redis.from_url(s.redis_url)
    cache = QueryCache(redis, s)
    retriever = Retriever.from_settings(engine, s, cache)
    try:
        for question in EXAMPLES:
            done = {}
            async for name, data in answer_events(retriever, generator, question, cache=cache):
                if name in ("done", "error"):
                    done = data | {"event": name}
            state = "cached" if done.get("cached") else done["event"]
            if done.get("abstained"):
                state += ", abstained"
            print(f"[{state}] {question}\n  {done.get('answer') or done.get('message', '')}\n")
    finally:
        await redis.aclose()
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["warm"])
    parser.parse_args()
    asyncio.run(warm())


if __name__ == "__main__":
    main()
