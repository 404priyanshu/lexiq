# LexIQ

Eval-driven RAG platform for document QA: hybrid retrieval (BM25 + pgvector HNSW),
cross-encoder rerank, citation-grounded streaming answers, and an abstain path when
evidence is insufficient. See [docs/PLAN.md](docs/PLAN.md) for the build plan.

**Status:** Phase 6 (latency + cost) — rerank pruning and an int8 reranker cut retrieval
p95 from 516 ms to 169 ms at unchanged recall; Redis caches serve repeat questions in under
1 ms. See [Performance](#performance).

## Stack

- `apps/web` — Next.js 16 (App Router, Tailwind)
- `services/api` — FastAPI, SQLAlchemy (async), Alembic, Python 3.12 via uv
- `infra` — ParadeDB Postgres 17 (pgvector + pg_search BM25), Redis

## Run locally

Requires Docker, [uv](https://docs.astral.sh/uv/), pnpm, and an `ANTHROPIC_API_KEY` in `.env`.

```bash
make setup   # .env, deps, ONNX models, containers, migrations
make api     # http://localhost:8000/health
make worker  # ingestion jobs
make web     # http://localhost:3000
make test
make eval-corpus   # once: fetch CUAD, ingest 98 eval contracts
make eval          # golden-set retrieval metrics + regression gate
```

Upload a document and watch it move `pending → processing → ready`:

```bash
curl -F file=@contract.pdf localhost:8000/documents
curl localhost:8000/documents
```

Search (returns reranked passages with pages, per-stage timings, and an abstain flag):

```bash
curl localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "How old must a senator be?", "top_k": 3}'
```

Ask (server-sent events: `sources`, `delta`…, then `done` with citations, abstain/refusal
flags, token usage and timings):

```bash
curl -N localhost:8000/answer -H 'content-type: application/json' \
  -d '{"query": "How old must a senator be?"}'
```

The answer path abstains twice over: before calling the model when no passage is relevant
enough, and when the model reports the passages don't answer the question. Requests use
Claude's server-side refusal fallback (`fallbacks: "default"`).

Tests use a separate `lexiq_test` database and Redis DB 15. Tests that call the real API
are skipped unless `LEXIQ_LIVE_TESTS=1`.

Postgres listens on `5433` and Redis on `6380` so they don't clash with local installs.

## Performance

Retrieval latency per request, 100 golden questions run one at a time on an Apple M5 Pro
(`make bench`); recall from the 500-question golden set (`make eval`).

| Configuration | p50 | p95 | recall@5 | recall@1 |
|---|---|---|---|---|
| Phase 5: rerank 30 candidates, fp32 | 387 ms | 516 ms | 0.765 | 0.452 |
| Rerank 10 candidates | 186 ms | 198 ms | 0.762 | 0.457 |
| **+ int8 reranker (default)** | **159 ms** | **169 ms** | **0.762** | **0.459** |
| Repeat question (Redis cache hit) | <1 ms | 1 ms | — | — |

What the sweeps showed:

- Reranking 10 candidates instead of 30 costs 0.003 recall@5. The cross-encoder rarely lifts
  a candidate from below RRF's top 10 into the top 8.
- Truncating reranker input from 512 to 256 tokens would have cut latency further but dropped
  recall@5 to ~0.70: many clauses sit past a chunk's first 256 tokens. Kept at 512.
- Dynamic int8 quantization of the cross-encoder (87 MB → 22 MB) leaves recall unchanged.
- BM25 (~40 ms, pg_search's per-query cost) is now the slowest first-stage step; it already
  runs in parallel with query embedding and the HNSW search.

Caches (`app/cache.py`): reranked retrievals (1 h) and finished answers (24 h) live in Redis
under keys that include a corpus version bumped by every ingest and a fingerprint of the
settings and prompt, so stale results are never served and nothing needs invalidating.
Refusals and truncated answers aren't cached. Anthropic prompt caching isn't used: the system
prompt is below the minimum cacheable length and the sources differ per question.

End-to-end answer latency and $/query (`make bench` with `--answers`) need `ANTHROPIC_API_KEY`
and haven't been measured yet.
