# API image for the public demo (infra/demo). The repo layout is mirrored under /repo because
# app.config finds the repo root three levels above app/config.py.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /repo/services/api

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
RUN uv sync --locked --no-dev

# Pinned ONNX models (and the int8 reranker) baked into the image: ~240 MB.
RUN uv run --no-sync python -m app.ml.download

ENV PATH="/repo/services/api/.venv/bin:$PATH"
EXPOSE 8000
# Behind the demo's Caddy proxy, which is the only thing that can reach this port, so
# X-Forwarded-For from it is trusted and request.client is the real visitor (for rate limits).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
