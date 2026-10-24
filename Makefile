COMPOSE := docker compose -f infra/docker-compose.yml --env-file .env
API := cd services/api &&
WEB := cd apps/web &&

.PHONY: setup up down migrate models api worker web test lint eval-corpus eval eval-answers bench

setup: ## First-time setup: env file, deps, stack, schema
	test -f .env || cp .env.example .env
	$(API) uv sync
	$(MAKE) models
	$(WEB) pnpm install
	$(MAKE) up migrate

up:
	$(COMPOSE) up -d --wait

down:
	$(COMPOSE) down

migrate:
	$(API) uv run alembic upgrade head

models: ## Download pinned ONNX models
	$(API) uv run python -m app.ml.download

api:
	$(API) uv run uvicorn app.main:app --reload --port 8000

worker: ## Background ingestion
	$(API) uv run arq app.worker.WorkerSettings

web:
	$(WEB) pnpm dev

test:
	$(API) uv run pytest -q

lint:
	$(API) uv run ruff check . && uv run ruff format --check .
	$(WEB) pnpm lint

CUAD := $(CURDIR)/data/cuad/extracted/CUAD_v1

eval-corpus: ## Download CUAD (checksum-verified) and ingest the 98 eval contracts
	test -f data/cuad/CUAD_v1.zip || (mkdir -p data/cuad && curl -sSfL -o data/cuad/CUAD_v1.zip $$(jq -r .url evals/corpus.json))
	echo "$$(jq -r .zip_md5 evals/corpus.json)  data/cuad/CUAD_v1.zip" | md5sum -c - 2>/dev/null || [ "$$(md5 -q data/cuad/CUAD_v1.zip)" = "$$(jq -r .zip_md5 evals/corpus.json)" ]
	test -d $(CUAD) || unzip -q data/cuad/CUAD_v1.zip -d data/cuad/extracted
	$(API) uv run python -m app.evals.corpus --cuad $(CUAD)

eval: ## Retrieval metrics on all 500 golden questions, then the regression gate
	$(API) uv run python -m app.evals.run --out ../../evals/runs/latest
	$(API) uv run python -m app.evals.gate ../../evals/runs/latest/summary.json

eval-answers: ## Also generate + judge answers for 100 questions (uses the Anthropic API)
	$(API) uv run python -m app.evals.run --answers 100 --out ../../evals/runs/latest
	$(API) uv run python -m app.evals.gate ../../evals/runs/latest/summary.json

bench: ## Per-request retrieval latency over 100 golden questions (cold, then warm cache)
	$(API) uv run python -m app.bench.run --questions 100 --cache --repeat 2
