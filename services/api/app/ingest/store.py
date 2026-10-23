"""Register uploaded files: content-addressed storage plus a `documents` row."""

import hashlib
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Document


def storage_path(sha256: str) -> Path:
    return get_settings().uploads_dir / sha256


async def register_document(
    session: AsyncSession,
    data: bytes,
    *,
    title: str,
    mime_type: str,
    source_uri: str | None = None,
) -> tuple[Document, bool]:
    """Store the file and insert a pending document. Returns (document, created).

    Identical bytes map to the existing document instead of a duplicate.
    """
    sha = hashlib.sha256(data).hexdigest()
    existing = await session.scalar(select(Document).where(Document.sha256 == sha))
    if existing:
        return existing, False

    path = storage_path(sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    doc = Document(title=title, mime_type=mime_type, sha256=sha, source_uri=source_uri)
    session.add(doc)
    try:
        await session.commit()
    except IntegrityError:
        # Lost a race with a concurrent upload of the same bytes.
        await session.rollback()
        return await session.scalar(select(Document).where(Document.sha256 == sha)), False
    await session.refresh(doc)
    return doc, True
