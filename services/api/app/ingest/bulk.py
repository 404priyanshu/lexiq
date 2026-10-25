"""Bulk-load a corpus through the normal upload path: register each file (deduplicated by
content hash) and enqueue it on the arq worker, then report progress until every document is
ready or failed. Needs the worker running (`make worker`).

Usage:
  uv run python -m app.ingest.bulk cuad              # all 510 CUAD contracts
  uv run python -m app.ingest.bulk cfr               # volumes in corpora/cfr-2025.json
"""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy import func, select

from app.config import REPO_ROOT, get_settings
from app.db import SessionLocal, engine
from app.ingest.parsers import MIME_BY_SUFFIX, mime_for
from app.ingest.store import register_document
from app.models import Document, DocumentStatus
from app.worker import INGEST_JOB, ingest_job_id

CUAD_PDFS = REPO_ROOT / "data/cuad/extracted/CUAD_v1/full_contract_pdf"
CFR_DIR = REPO_ROOT / "data/cfr"


def cuad_files() -> list[tuple[Path, str, str | None]]:
    pdfs = sorted(p for p in CUAD_PDFS.rglob("*") if p.suffix.lower() in MIME_BY_SUFFIX)
    return [(p, p.stem.strip(), None) for p in pdfs]


def cfr_files(manifest: Path) -> list[tuple[Path, str, str | None]]:
    docs = json.loads(manifest.read_text())["documents"]
    return [(CFR_DIR / d["file"], d["name"], d["url"]) for d in docs]


async def enqueue(files: list[tuple[Path, str, str | None]]) -> list[uuid.UUID]:
    redis = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    ids, queued = [], 0
    try:
        for path, title, source_uri in files:
            async with SessionLocal() as session:
                doc, created = await register_document(
                    session,
                    path.read_bytes(),
                    title=title,
                    mime_type=mime_for(path.name),
                    source_uri=source_uri,
                )
            ids.append(doc.id)
            if created or doc.status == DocumentStatus.failed:
                await redis.enqueue_job(INGEST_JOB, str(doc.id), _job_id=ingest_job_id(doc.id))
                queued += 1
    finally:
        await redis.aclose()
    # Files with identical bytes map to one document; waiting on duplicate ids would never
    # finish, since only one document exists for them.
    unique = list(dict.fromkeys(ids))
    print(
        f"{len(files)} files -> {len(unique)} documents, {queued} queued for ingestion",
        flush=True,
    )
    return unique


async def wait(ids: list[uuid.UUID], interval: float = 30, max_wait_s: float = 4 * 3600) -> None:
    """Report progress until every document is ready or failed. Raises TimeoutError if that
    hasn't happened within `max_wait_s` seconds (e.g. the worker died mid-job and left a document
    in `processing`), naming the documents still unfinished."""
    started = time.perf_counter()
    while True:
        async with SessionLocal() as session:
            rows = await session.execute(
                select(
                    Document.status, func.count(), func.coalesce(func.sum(Document.page_count), 0)
                )
                .where(Document.id.in_(ids))
                .group_by(Document.status)
            )
            by_status = {status: (n, pages) for status, n, pages in rows}
        done = sum(
            by_status.get(s, (0, 0))[0] for s in (DocumentStatus.ready, DocumentStatus.failed)
        )
        pages = by_status.get(DocumentStatus.ready, (0, 0))[1]
        failed = by_status.get(DocumentStatus.failed, (0, 0))[0]
        elapsed = time.perf_counter() - started
        print(
            f"[{elapsed:6.0f}s] {done}/{len(ids)} done, {pages} pages ready, {failed} failed",
            flush=True,
        )
        if done == len(ids):
            return
        if elapsed >= max_wait_s:
            async with SessionLocal() as session:
                stuck = (
                    await session.execute(
                        select(Document.title, Document.status).where(
                            Document.id.in_(ids),
                            Document.status.not_in([DocumentStatus.ready, DocumentStatus.failed]),
                        )
                    )
                ).all()
            names = ", ".join(f"{t} ({s.value})" for t, s in stuck[:5])
            raise TimeoutError(
                f"{len(stuck)} documents unfinished after {max_wait_s:.0f}s: {names}"
            )
        await asyncio.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("corpus", choices=["cuad", "cfr"])
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "corpora/cfr-2025.json")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument(
        "--timeout", type=float, default=4 * 3600, help="give up waiting after this many seconds"
    )
    args = parser.parse_args()

    files = cuad_files() if args.corpus == "cuad" else cfr_files(args.manifest)

    async def run():
        try:
            ids = await enqueue(files)
            if not args.no_wait:
                await wait(ids, max_wait_s=args.timeout)
        finally:
            await engine.dispose()

    asyncio.run(run())


if __name__ == "__main__":
    main()
