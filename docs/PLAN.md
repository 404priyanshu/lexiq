# LexIQ build plan

Eval-driven RAG platform for document QA: hybrid retrieval (BM25 + pgvector HNSW),
cross-encoder rerank, citation-grounded streaming answers, and an abstain path.

## Layout

```
apps/web/            Next.js (App Router, TS, Tailwind): upload, ask, streamed answers with citations
services/api/        FastAPI (Python, uv): ingest, retrieve, rerank, generate, abstain
  models/            ONNX embedder + cross-encoder (downloaded by script, not committed)
evals/               golden set (JSONL), eval runner, baseline scores
bench/               latency + cost benchmarks
infra/               docker-compose: ParadeDB Postgres (pgvector + pg_search), Redis
.github/workflows/   CI: lint, tests, eval gate
```

## Phases

1. **Foundation** — compose stack, FastAPI skeleton, Alembic migrations, Next.js scaffold.
   Schema: `documents`, `chunks` (text, page, offsets, `vector(384)`), HNSW + BM25 indexes.
2. **Ingestion** — PyMuPDF parsing (PDF/HTML/MD), page-aware chunking with overlap,
   ONNX `bge-small-en-v1.5` embeddings, arq background jobs with per-document status.
3. **Retrieval** — BM25 and HNSW in parallel, Reciprocal Rank Fusion, ONNX
   `ms-marco-MiniLM-L-6-v2` cross-encoder rerank, abstain on low evidence.
4. **Generation** — Claude with numbered context, required `[n]` citations, SSE streaming,
   citation validation post-check.
5. **Eval harness + CI gate** — 500-question golden set (incl. unanswerable), recall@k, MRR,
   LLM-judge faithfulness, abstain precision; GitHub Actions blocks regressions vs `evals/baseline.json`.
6. **Latency + cost** — naive baseline first, then parallel retrieval/rerank, Redis caching,
   Anthropic prompt caching, rerank pruning; `bench/` reports p50/p95 and $/query.
7. **Scale + polish** — 50k+ page corpus (CUAD + SEC EDGAR), re-run evals and bench, README.

## Decisions

- Corpus: CUAD contracts (CC BY 4.0) + Code of Federal Regulations from govinfo.gov (public
  domain). CFR replaced SEC EDGAR, whose fair-access policy requires a contact email in every
  request; the corpus reached 52,650 pages.
- Golden set: built from CUAD's lawyer annotations rather than LLM-drafted questions, including
  75 unanswerable questions where CUAD records a clause is absent.
- BM25: ParadeDB `pg_search` (real BM25), not Postgres `ts_rank`.
- LLM: Claude via the Anthropic SDK, model set in config.
- CI: retrieval metrics on all 500 questions every PR; LLM-judge faithfulness on a fixed
  100-question subset per PR, full 500 on `main`.
- Reported numbers come from `bench/` and `evals/` runs, not targets.
