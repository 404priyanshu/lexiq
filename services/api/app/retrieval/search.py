"""First-stage candidate generation: BM25 (pg_search) and dense HNSW (pgvector), plus
Reciprocal Rank Fusion to merge them."""

import uuid
from collections.abc import Sequence

import numpy as np
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.ext.asyncio import AsyncConnection

_DOC_FILTER = "and document_id = any(:document_ids)"
_DOC_IDS = bindparam("document_ids", type_=ARRAY(UUID(as_uuid=True)))


async def bm25_search(
    conn: AsyncConnection,
    query: str,
    k: int,
    document_ids: Sequence[uuid.UUID] | None = None,
) -> list[int]:
    """Chunk ids by BM25 score. `|||` matches any query term (tokenized, no query syntax),
    so user input needs no escaping."""
    sql = f"""
        select id from chunks
        where content ||| :query {_DOC_FILTER if document_ids else ""}
        order by pdb.score(id) desc
        limit :k
    """
    stmt = text(sql)
    params: dict = {"query": query, "k": k}
    if document_ids:
        stmt = stmt.bindparams(_DOC_IDS)
        params["document_ids"] = list(document_ids)
    return list((await conn.execute(stmt, params)).scalars())


async def dense_search(
    conn: AsyncConnection,
    query_vec: np.ndarray,
    k: int,
    ef_search: int,
    document_ids: Sequence[uuid.UUID] | None = None,
) -> list[int]:
    """Chunk ids by cosine distance via the HNSW index."""
    async with conn.begin():
        # SET can't take bind params; both values are ints we control.
        await conn.execute(text(f"set local hnsw.ef_search = {int(ef_search)}"))
        if document_ids:
            # Keep scanning the graph until enough rows survive the filter (pgvector >= 0.8).
            await conn.execute(text("set local hnsw.iterative_scan = strict_order"))
        sql = f"""
            select id from chunks
            where embedding is not null {_DOC_FILTER if document_ids else ""}
            order by embedding <=> cast(:qvec as vector)
            limit :k
        """
        stmt = text(sql)
        params: dict = {"qvec": str(query_vec.tolist()), "k": k}
        if document_ids:
            stmt = stmt.bindparams(_DOC_IDS)
            params["document_ids"] = list(document_ids)
        return list((await conn.execute(stmt, params)).scalars())


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank), rank from 1.

    Rank-based, so BM25 and cosine scores never need to be put on the same scale.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
