"""Ingest the eval corpus (evals/corpus.json) into the configured database.

Runs the normal ingestion pipeline inline (no worker). Documents already ingested with the
same content hash are skipped, so re-running only does new work.

Usage: uv run python -m app.evals.corpus --cuad ../../data/cuad/extracted/CUAD_v1
"""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

from sqlalchemy import select

from app.config import REPO_ROOT
from app.db import SessionLocal, engine
from app.evals.golden import CORPUS_PATH
from app.ingest.parsers import mime_for
from app.ingest.pipeline import Models, ingest_document
from app.ingest.store import register_document
from app.ml.embedder import Embedder
from app.ml.registry import EMBEDDER
from app.ml.runtime import load_tokenizer
from app.models import Document, DocumentStatus


def load_corpus() -> list[dict]:
    return json.loads(CORPUS_PATH.read_text())["documents"]


async def document_ids_by_sha() -> dict[str, tuple[uuid.UUID, DocumentStatus]]:
    async with SessionLocal() as session:
        rows = await session.execute(select(Document.sha256, Document.id, Document.status))
        return {sha: (doc_id, status) for sha, doc_id, status in rows}


async def ingest_corpus(cuad: Path) -> None:
    corpus = load_corpus()
    existing = await document_ids_by_sha()
    todo = [d for d in corpus if existing.get(d["sha256"], (None, None))[1] != DocumentStatus.ready]
    print(f"{len(corpus) - len(todo)}/{len(corpus)} already ingested")
    if not todo:
        return

    models = Models(embedder=Embedder(), tokenizer=load_tokenizer(EMBEDDER))
    started = time.perf_counter()
    for i, d in enumerate(todo, start=1):
        path = cuad / d["pdf"]
        data = path.read_bytes()
        async with SessionLocal() as session:
            doc, _ = await register_document(
                session, data, title=d["contract"], mime_type=mime_for(path.name)
            )
            await ingest_document(session, doc.id, models)
            await session.refresh(doc)
        pages = doc.page_count or 0
        print(f"[{i}/{len(todo)}] {doc.status.value:6s} {pages:3d}p  {d['contract'][:70]}")
    print(f"ingested {len(todo)} documents in {time.perf_counter() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cuad", type=Path, default=REPO_ROOT / "data/cuad/extracted/CUAD_v1")
    args = parser.parse_args()

    async def run():
        try:
            await ingest_corpus(args.cuad)
        finally:
            await engine.dispose()

    asyncio.run(run())


if __name__ == "__main__":
    main()
