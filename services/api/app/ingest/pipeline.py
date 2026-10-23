"""Parse -> chunk -> embed -> store for one document. Safe to re-run: chunks are replaced."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

from sqlalchemy import delete, insert, update
from sqlalchemy.ext.asyncio import AsyncSession
from tokenizers import Tokenizer

from app.config import get_settings
from app.ingest.chunker import chunk_pages
from app.ingest.parsers import parse
from app.ingest.store import storage_path
from app.ml.embedder import Embedder
from app.models import Chunk, Document, DocumentStatus

log = logging.getLogger(__name__)

INSERT_BATCH = 500


@dataclass
class Models:
    embedder: Embedder
    # Untruncated tokenizer for counting; the embedder's own one truncates at 512.
    tokenizer: Tokenizer

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(e.ids) for e in self.tokenizer.encode_batch(texts, add_special_tokens=False)]


def _prepare(doc: Document, models: Models) -> tuple[int, list[dict]]:
    """CPU-bound part; runs in a worker thread (ONNX Runtime releases the GIL)."""
    s = get_settings()
    parsed = parse(storage_path(doc.sha256).read_bytes(), doc.mime_type)
    chunks = chunk_pages(
        parsed.pages,
        models.count_tokens,
        target=s.chunk_target_tokens,
        max_tokens=s.chunk_max_tokens,
        overlap=s.chunk_overlap_tokens,
    )
    vectors = models.embedder.embed_documents([c.content for c in chunks])
    rows = [
        {
            "document_id": doc.id,
            "ordinal": c.ordinal,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "content": c.content,
            "token_count": c.token_count,
            "embedding": vec,
        }
        for c, vec in zip(chunks, vectors, strict=True)
    ]
    return len(parsed.pages), rows


async def ingest_document(session: AsyncSession, document_id: uuid.UUID, models: Models) -> None:
    doc = await session.get(Document, document_id)
    if doc is None:
        log.warning("ingest: document %s not found", document_id)
        return

    doc.status = DocumentStatus.processing
    doc.error = None
    await session.commit()

    started = time.perf_counter()
    try:
        page_count, rows = await asyncio.to_thread(_prepare, doc, models)
        await session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
        for i in range(0, len(rows), INSERT_BATCH):
            await session.execute(insert(Chunk), rows[i : i + INSERT_BATCH])
        doc.page_count = page_count
        doc.status = DocumentStatus.ready
        await session.commit()
    except Exception as exc:
        log.exception("ingest failed for %s", document_id)
        await session.rollback()
        # A direct UPDATE, not the (now stale) ORM object: if the document was deleted
        # meanwhile, nothing matches and there's nothing left to mark failed.
        marked = await session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(status=DocumentStatus.failed, error=f"{type(exc).__name__}: {exc}"[:2000])
        )
        await session.commit()
        if marked.rowcount == 0:
            log.warning("ingest: document %s was deleted while ingesting", document_id)
        return

    log.info(
        "ingested %s: %d pages, %d chunks in %.1fs",
        document_id, page_count, len(rows), time.perf_counter() - started,
    )  # fmt: skip
