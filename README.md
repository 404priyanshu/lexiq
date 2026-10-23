# LexIQ

Eval-driven RAG platform for document QA: hybrid retrieval (BM25 + pgvector HNSW),
cross-encoder rerank, citation-grounded streaming answers, and an abstain path when
evidence is insufficient. See [docs/PLAN.md](docs/PLAN.md) for the build plan.

**Status:** Phase 1 (foundation) — stack, schema, API and web skeletons.

## Stack

- `apps/web` — Next.js 16 (App Router, Tailwind)
- `services/api` — FastAPI, SQLAlchemy (async), Alembic, Python 3.12 via uv
- `infra` — ParadeDB Postgres 17 (pgvector + pg_search BM25), Redis

## Run locally

Requires Docker, [uv](https://docs.astral.sh/uv/) and pnpm.

```bash
make setup   # .env, deps, containers, migrations
make api     # http://localhost:8000/health
make web     # http://localhost:3000
make test
```

Postgres listens on `5433` and Redis on `6380` so they don't clash with local installs.
