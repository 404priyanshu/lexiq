#!/usr/bin/env bash
# Build and (re)start the demo from the current checkout. Run on the server, in this directory,
# after filling in .env. Safe to rerun: migrations are idempotent and data volumes persist.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "missing .env (copy .env.example)"; exit 1; }

git pull --ff-only
docker compose build
docker compose up -d postgres redis
docker compose run --rm api alembic upgrade head
docker compose up -d
docker compose ps
