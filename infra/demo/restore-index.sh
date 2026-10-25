#!/usr/bin/env bash
# Replace the demo's documents and chunks with a data-only dump made by `make demo-dump`, then
# drop stale cached results and warm the example answers. Run on the server after deploy.sh.
set -euo pipefail
cd "$(dirname "$0")"
dump=${1:?usage: ./restore-index.sh path/to/demo-index.dump}

docker compose exec -T postgres psql -U lexiq -d lexiq -v ON_ERROR_STOP=1 \
  -c "truncate documents cascade"
# BM25 and HNSW indexes are maintained as rows load, so this takes several minutes at 124k chunks.
docker compose exec -T postgres pg_restore -U lexiq -d lexiq --data-only --disable-triggers < "$dump"
# Cache keys carry the corpus version; bumping it retires anything cached before the restore.
docker compose exec -T redis redis-cli incr lexiq:corpus_version
docker compose exec -T api python -m app.demo warm
