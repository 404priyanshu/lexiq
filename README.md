# LexIQ

Eval-driven RAG platform for document QA: hybrid retrieval (BM25 + pgvector HNSW),
cross-encoder rerank, citation-grounded streaming answers, and an abstain path when
evidence is insufficient. See [docs/PLAN.md](docs/PLAN.md) for the build plan.

**Status:** Phase 2 (ingestion) — upload PDF/HTML/Markdown, page-aware chunking, local ONNX
embeddings, background ingestion with per-document status.

## Stack

- `apps/web` — Next.js 16 (App Router, Tailwind)
- `services/api` — FastAPI, SQLAlchemy (async), Alembic, Python 3.12 via uv
- `infra` — ParadeDB Postgres 17 (pgvector + pg_search BM25), Redis

## Run locally

Requires Docker, [uv](https://docs.astral.sh/uv/) and pnpm.

```bash
make setup   # .env, deps, ONNX models, containers, migrations
make api     # http://localhost:8000/health
make worker  # ingestion jobs
make web     # http://localhost:3000
make test
```

Upload a document and watch it move `pending → processing → ready`:

```bash
curl -F file=@contract.pdf localhost:8000/documents
curl localhost:8000/documents
```

Tests use a separate `lexiq_test` database and Redis DB 15.

Postgres listens on `5433` and Redis on `6380` so they don't clash with local installs.
