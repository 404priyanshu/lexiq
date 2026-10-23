"""arq worker for background ingestion. Run: uv run arq app.worker.WorkerSettings"""

import logging
import uuid

from arq.connections import RedisSettings

from app.config import get_settings
from app.db import SessionLocal, engine
from app.ingest.pipeline import Models, ingest_document
from app.ml.embedder import Embedder, load_tokenizer

INGEST_JOB = "ingest"


def ingest_job_id(document_id: uuid.UUID | str) -> str:
    # Stable id: arq won't enqueue a second job while one with this id is queued or running.
    return f"ingest:{document_id}"


async def startup(ctx: dict) -> None:
    logging.basicConfig(level=logging.INFO)
    s = get_settings()
    ctx["models"] = Models(
        embedder=Embedder(batch_size=s.embed_batch_size), tokenizer=load_tokenizer()
    )


async def shutdown(ctx: dict) -> None:
    await engine.dispose()


async def ingest(ctx: dict, document_id: str) -> None:
    async with SessionLocal() as session:
        await ingest_document(session, uuid.UUID(document_id), ctx["models"])


class WorkerSettings:
    functions = [ingest]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    # Embedding is CPU-bound and already multi-threaded inside ONNX Runtime.
    max_jobs = 2
    job_timeout = 1800
    # Status lives on the documents row. Not keeping arq results also frees the stable job id
    # as soon as a job finishes, so a document can be re-ingested right away.
    keep_result = 0
