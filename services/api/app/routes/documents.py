import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.ingest.parsers import UnsupportedFormat, mime_for
from app.ingest.store import register_document
from app.models import Chunk, Document, DocumentStatus
from app.worker import INGEST_JOB, ingest_job_id

router = APIRouter(prefix="/documents", tags=["documents"])

Session = Annotated[AsyncSession, Depends(get_session)]


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    mime_type: str
    source_uri: str | None
    page_count: int | None
    status: DocumentStatus
    error: str | None
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int


def _chunk_count():
    return (
        select(func.count(Chunk.id))
        .where(Chunk.document_id == Document.id)
        .correlate(Document)
        .scalar_subquery()
    )


async def _get_out(session: AsyncSession, document_id: uuid.UUID) -> DocumentOut:
    row = (
        await session.execute(select(Document, _chunk_count()).where(Document.id == document_id))
    ).first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    doc, count = row
    return DocumentOut.model_validate(doc).model_copy(update={"chunk_count": count})


async def _enqueue(request: Request, document_id: uuid.UUID) -> None:
    await request.app.state.redis.enqueue_job(
        INGEST_JOB, str(document_id), _job_id=ingest_job_id(document_id)
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    response: Response,
    session: Session,
    file: UploadFile,
    title: Annotated[str | None, Form()] = None,
    source_uri: Annotated[str | None, Form()] = None,
) -> DocumentOut:
    try:
        mime_type = mime_for(file.filename or "")
    except UnsupportedFormat as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from None

    limit = get_settings().max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "file too large")
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "file is empty")

    doc, created = await register_document(
        session,
        data,
        title=title or file.filename or "untitled",
        mime_type=mime_type,
        source_uri=source_uri,
    )
    if created:
        await _enqueue(request, doc.id)
    else:
        response.status_code = status.HTTP_200_OK
    return await _get_out(session, doc.id)


@router.get("")
async def list_documents(session: Session, limit: int = 50, offset: int = 0) -> DocumentList:
    limit = max(1, min(limit, 200))
    rows = await session.execute(
        select(Document, _chunk_count())
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    total = await session.scalar(select(func.count(Document.id)))
    items = [DocumentOut.model_validate(d).model_copy(update={"chunk_count": n}) for d, n in rows]
    return DocumentList(items=items, total=total or 0)


@router.get("/{document_id}")
async def get_document(session: Session, document_id: uuid.UUID) -> DocumentOut:
    return await _get_out(session, document_id)


@router.post("/{document_id}/reingest", status_code=status.HTTP_202_ACCEPTED)
async def reingest_document(
    request: Request, session: Session, document_id: uuid.UUID
) -> DocumentOut:
    doc = await session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    doc.status = DocumentStatus.pending
    doc.error = None
    await session.commit()
    await _enqueue(request, document_id)
    return await _get_out(session, document_id)
