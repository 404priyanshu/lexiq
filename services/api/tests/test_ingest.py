import uuid

import pytest
from sqlalchemy import select

from app.db import SessionLocal
from app.ingest.pipeline import ingest_document
from app.main import app
from app.models import Chunk
from app.worker import ingest_job_id
from tests.test_parsers import make_pdf

pytestmark = pytest.mark.usefixtures("clean_db")

PAGES = [
    "MASTER SERVICES AGREEMENT. This agreement is between Acme Corp and Globex Inc.",
    "Section 7. Indemnification. Each party shall indemnify the other against third-party claims.",
    "Section 12. Governing Law. This Agreement is governed by the laws of the State of Delaware.",
]


async def upload(client, data: bytes, name: str = "msa.pdf"):
    return await client.post("/documents", files={"file": (name, data)}, data={"title": "MSA"})


async def test_upload_enqueues_then_ingest_stores_cited_chunks(client, models):
    resp = await upload(client, make_pdf(PAGES))
    assert resp.status_code == 201, resp.text
    doc = resp.json()
    assert doc["status"] == "pending" and doc["title"] == "MSA"
    assert await app.state.redis.exists(f"arq:job:{ingest_job_id(doc['id'])}")

    async with SessionLocal() as session:
        await ingest_document(session, uuid.UUID(doc["id"]), models)

    doc = (await client.get(f"/documents/{doc['id']}")).json()
    assert doc["status"] == "ready", doc["error"]
    assert doc["page_count"] == 3
    assert doc["chunk_count"] >= 1

    async with SessionLocal() as session:
        chunks = (await session.scalars(select(Chunk).order_by(Chunk.ordinal))).all()
    assert all(c.embedding is not None and len(c.embedding) == 384 for c in chunks)
    delaware = next(c for c in chunks if "Delaware" in c.content)
    assert delaware.page_end == 3


async def test_reingest_replaces_chunks(client, models):
    doc = (await upload(client, make_pdf(PAGES))).json()
    doc_id = uuid.UUID(doc["id"])
    async with SessionLocal() as session:
        await ingest_document(session, doc_id, models)
        await ingest_document(session, doc_id, models)
        ordinals = (await session.scalars(select(Chunk.ordinal))).all()
    assert sorted(ordinals) == list(range(len(ordinals)))


async def test_duplicate_upload_returns_existing_document(client):
    data = make_pdf(PAGES)  # generated PDFs embed timestamps, so reuse the same bytes
    first = await upload(client, data)
    second = await upload(client, data, name="copy.pdf")

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert (await client.get("/documents")).json()["total"] == 1


async def test_corrupt_file_marks_document_failed(client, models):
    doc = (await upload(client, b"%PDF-1.7 not really a pdf")).json()
    async with SessionLocal() as session:
        await ingest_document(session, uuid.UUID(doc["id"]), models)

    doc = (await client.get(f"/documents/{doc['id']}")).json()
    assert doc["status"] == "failed"
    assert doc["error"]


@pytest.mark.parametrize(
    ("name", "data", "code"),
    [("notes.zip", b"PK..", 415), ("empty.pdf", b"", 422)],
)
async def test_rejects_bad_uploads(client, name, data, code):
    assert (await upload(client, data, name)).status_code == code


async def test_unknown_document_is_404(client):
    assert (await client.get(f"/documents/{uuid.uuid4()}")).status_code == 404
