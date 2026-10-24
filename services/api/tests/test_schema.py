"""Schema smoke tests against the migrated database. Each test rolls back its writes."""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.models import EMBEDDING_DIM


@pytest.fixture
async def conn():
    engine = create_async_engine(get_settings().database_url)
    async with engine.connect() as c:
        tx = await c.begin()
        yield c
        await tx.rollback()
    await engine.dispose()


async def test_chunk_indexes_exist(conn):
    rows = await conn.execute(
        text("select indexname, indexdef from pg_indexes where tablename = 'chunks'")
    )
    defs = {name: d for name, d in rows}
    assert "USING hnsw" in defs["ix_chunks_embedding_hnsw"]
    assert "vector_cosine_ops" in defs["ix_chunks_embedding_hnsw"]
    assert "USING bm25" in defs["ix_chunks_bm25"]


def _unit(i: int) -> str:
    v = [0.0] * EMBEDDING_DIM
    v[i] = 1.0
    return str(v)


async def test_bm25_and_vector_search_round_trip(conn):
    doc_id = (
        await conn.execute(
            text(
                "insert into documents (title, mime_type, sha256) "
                "values ('t', 'application/pdf', 'test-sha') returning id"
            )
        )
    ).scalar_one()
    chunks = [
        "The indemnification obligations survive termination of this Agreement.",
        "This Agreement is governed by the laws of the State of Delaware.",
    ]
    for i, content in enumerate(chunks):
        await conn.execute(
            text(
                "insert into chunks (document_id, ordinal, page_start, page_end, char_start, "
                "char_end, content, token_count, embedding) "
                "values (:d, :o, 1, 1, 0, :n, :c, 10, cast(:e as vector))"
            ),
            {"d": doc_id, "o": i, "n": len(content), "c": content, "e": _unit(i)},
        )

    lexical = await conn.execute(
        text(
            "select content from chunks where content ||| 'delaware law' "
            "order by pdb.score(id) desc limit 1"
        )
    )
    assert "Delaware" in lexical.scalar_one()

    dense = await conn.execute(
        text("select content from chunks order by embedding <=> cast(:q as vector) limit 1"),
        {"q": _unit(0)},
    )
    assert "indemnification" in dense.scalar_one()
