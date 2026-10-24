"""Latency and cost benchmark: questions run one at a time, as a user's request would.

Retrieval stages are always measured. With --answers (needs ANTHROPIC_API_KEY) each question
also goes through generation, adding time to first token, total answer time and $/query from
the API's reported token usage. --repeat 2 runs the same questions again to measure warm
caches. Results go to bench/results/<timestamp>.json.

Usage:
  uv run python -m app.bench.run --questions 100
  uv run python -m app.bench.run --questions 50 --answers --repeat 2
  uv run python -m app.bench.run --set rerank_candidates=15
"""

import argparse
import asyncio
import json
import statistics
from datetime import UTC, datetime

from anthropic import AsyncAnthropic
from redis.asyncio import Redis

from app.cache import CORPUS_VERSION_KEY, QueryCache
from app.config import REPO_ROOT, settings_with
from app.db import engine
from app.evals.corpus import document_ids_by_sha, load_corpus
from app.evals.golden import GOLDEN_PATH
from app.generation.generator import ClaudeGenerator
from app.generation.service import answer_events
from app.retrieval.retriever import Retriever

RESULTS_DIR = REPO_ROOT / "bench" / "results"
WARMUP = 3

# USD per million tokens (input, output), from Anthropic's published pricing.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    for prefix, (inp, out) in PRICES.items():
        if model.startswith(prefix):
            return (input_tokens * inp + output_tokens * out) / 1_000_000
    return None


def pcts(values: list[float]) -> dict | None:
    if not values:
        return None
    if len(values) == 1:
        return {"p50": round(values[0], 1), "p95": round(values[0], 1), "n": 1}
    q = statistics.quantiles(values, n=100)
    return {"p50": round(q[49], 1), "p95": round(q[94], 1), "n": len(values)}


async def bench(args, settings) -> dict:
    golden = [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines()]
    questions = golden[: args.questions]
    docs = await document_ids_by_sha()
    doc_for = {d["contract"]: docs[d["sha256"]][0] for d in load_corpus() if d["sha256"] in docs}

    cache = None
    if args.cache:
        redis = Redis.from_url(settings.redis_url)
        await redis.incr(CORPUS_VERSION_KEY)  # start cold: no entries from earlier runs
        cache = QueryCache(redis, settings)
    retriever = Retriever.from_settings(engine, settings, cache)
    generator = None
    if args.answers:
        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
        if not key:
            raise SystemExit("--answers needs ANTHROPIC_API_KEY")
        generator = ClaudeGenerator(AsyncAnthropic(api_key=key), settings)

    def scope(g: dict):
        return [doc_for[g["contract"]]] if args.scope == "document" else None

    # Load models and warm connections. Distinct text, so warm-ups never seed the cache entries
    # of the questions being measured.
    for g in questions[:WARMUP]:
        await retriever.retrieve(f"{g['question']} (warm-up)", document_ids=scope(g))

    passes = []
    for rep in range(args.repeat):
        stages: dict[str, list[float]] = {}
        ttft, totals, costs = [], [], []
        for g in questions:
            if generator:
                events = [
                    e
                    async for e in answer_events(
                        retriever, generator, g["question"], document_ids=scope(g), cache=cache
                    )
                ]
                done = next(d for name, d in events if name == "done")
                timings = done["timings_ms"]
                for stage, ms in timings["retrieval"].items():
                    stages.setdefault(stage, []).append(ms)
                if "ttft" in timings:
                    ttft.append(timings["ttft"])
                totals.append(timings["total"])
                if done["usage"] and done["model"]:
                    u = done["usage"]
                    c = cost_usd(done["model"], u["input_tokens"], u["output_tokens"])
                    if c is not None:
                        costs.append(c)
            else:
                res = await retriever.retrieve(g["question"], document_ids=scope(g))
                for stage, ms in res.timings_ms.items():
                    stages.setdefault(stage, []).append(ms)
                totals.append(res.timings_ms["total"])
        passes.append(
            {
                "pass": rep + 1,
                "stages_ms": {k: pcts(v) for k, v in stages.items()},
                "ttft_ms": pcts(ttft),
                "total_ms": pcts(totals),
                "cost_per_query_usd": round(statistics.mean(costs), 5) if costs else None,
            }
        )
    return {
        "questions": len(questions),
        "scope": args.scope,
        "answers": bool(generator),
        "overrides": args.overrides,
        "cache": args.cache,
        "passes": passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--scope", choices=["document", "corpus"], default="document")
    parser.add_argument("--answers", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cache", action="store_true", help="use the Redis caches (cold first)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    settings = settings_with(args.overrides)

    async def run():
        try:
            return await bench(args, settings)
        finally:
            await engine.dispose()

    result = asyncio.run(run())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    for p in result["passes"]:
        stages = "  ".join(
            f"{k} {v['p50']:.0f}/{v['p95']:.0f}" for k, v in p["stages_ms"].items() if v
        )
        print(f"pass {p['pass']}: {stages}  (p50/p95 ms)")
        line = f"        total p50 {p['total_ms']['p50']:.0f} ms, p95 {p['total_ms']['p95']:.0f} ms"
        if p["ttft_ms"]:
            line += f"; first token p50 {p['ttft_ms']['p50']:.0f} ms"
        if p["cost_per_query_usd"] is not None:
            line += f"; ${p['cost_per_query_usd']:.4f}/query"
        print(line)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
