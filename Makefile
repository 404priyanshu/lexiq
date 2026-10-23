COMPOSE := docker compose -f infra/docker-compose.yml --env-file .env
API := cd services/api &&
WEB := cd apps/web &&

.PHONY: setup up down migrate models api worker web test lint

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
