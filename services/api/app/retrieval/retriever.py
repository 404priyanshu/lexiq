"""Hybrid retrieval: BM25 + dense in parallel -> RRF -> cross-encoder rerank -> abstain check."""

import asyncio
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import Settings
from app.ml.embedder import Embedder
from app.ml.reranker import Reranker, sigmoid
from app.retrieval.routing import focus_query
from app.retrieval.search import bm25_search, dense_search, rrf

if TYPE_CHECKING:
    from app.cache import QueryCache


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: uuid.UUID
    document_title: str
    ordinal: int
    page_start: int
    page_end: int
    content: str
    bm25_rank: int | None
    dense_rank: int | None
    rrf_score: float
    rerank_score: float
    relevance: float  # sigmoid(rerank_score)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    abstained: bool
    abstain_reason: str | None
    candidates: int
    timings_ms: dict[str, float] = field(default_factory=dict)
    cached: bool = False
    # Set when two-stage retrieval narrowed a corpus-wide question to the document it names.
    routed_document_id: uuid.UUID | None = None
    focused_query: str | None = None


class _Timer:
    def __init__(self, timings: dict[str, float], name: str):
        self.timings, self.name = timings, name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.timings[self.name] = round((time.perf_counter() - self.start) * 1000, 1)


_PREAMBLE = text(
    """
    select string_agg(content, ' ' order by ordinal) from chunks
    where document_id = :document_id and ordinal <= 1
    """
)

_LOAD_CHUNKS = text(
    """
    select c.id, c.document_id, d.title, c.ordinal, c.page_start, c.page_end, c.content
    from chunks c join documents d on d.id = c.document_id
    where c.id = any(:ids)
    """
)


class Retriever:
    def __init__(
        self,
        engine: AsyncEngine,
        embedder: Embedder,
        reranker: Reranker,
        settings: Settings,
        cache: "QueryCache | None" = None,
    ):
        self.engine = engine
        self.embedder = embedder
        self.reranker = reranker
        self.s = settings
        self.cache = cache

    @classmethod
    def from_settings(
        cls, engine: AsyncEngine, settings: Settings, cache: "QueryCache | None" = None
    ) -> "Retriever":
        reranker = Reranker(max_length=settings.rerank_max_tokens, int8=settings.reranker_int8)
        return cls(engine, Embedder(), reranker, settings, cache)

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.s.top_k
        if self.cache is None:
            return await self._search(query, top_k, document_ids)

        started = time.perf_counter()
        if hit := await self.cache.get_retrieval(query, top_k, document_ids):
            hit.timings_ms = {"cache": _ms_since(started), "total": _ms_since(started)}
            return hit
        result = await self._search(query, top_k, document_ids)
        await self.cache.set_retrieval(query, top_k, document_ids, result)
        return result

    async def _search(
        self, query: str, top_k: int, document_ids: Sequence[uuid.UUID] | None
    ) -> RetrievalResult:
        """Hybrid retrieval, with two-stage routing for corpus-wide questions that name a
        document: stage 1 finds the document, stage 2 searches inside it for the clause."""
        if document_ids or not self.s.route_to_document:
            return await self._retrieve(query, top_k, document_ids)

        started = time.perf_counter()
        first = await self._retrieve(query, max(top_k, self.s.rerank_candidates), None)
        if first.abstained or not first.chunks:
            return first
        target = first.chunks[0].document_id
        focused, removed = focus_query(query, await self._preamble(target))
        if removed < self.s.route_min_identity_words:
            # The question doesn't name this document; keep the corpus-wide results.
            first.chunks = first.chunks[:top_k]
            return first

        second = await self._retrieve(focused, top_k, [target])
        second.query = query
        second.routed_document_id = target
        second.focused_query = focused
        stage2 = {k: v for k, v in second.timings_ms.items() if k != "total"}
        second.timings_ms = {"route": first.timings_ms["total"], **stage2}
        second.timings_ms["total"] = _ms_since(started)
        return second

    async def _preamble(self, document_id: uuid.UUID) -> str:
        """The document's opening chunks, where its title and parties are named."""
        async with self.engine.connect() as conn:
            text_ = await conn.scalar(_PREAMBLE, {"document_id": document_id})
        return text_ or ""

    async def _retrieve(
        self, query: str, top_k: int, document_ids: Sequence[uuid.UUID] | None
    ) -> RetrievalResult:
        t: dict[str, float] = {}
        started = time.perf_counter()

        # BM25 needs no embedding, so it runs while the query is embedded and the HNSW
        # search executes; each branch uses its own pooled connection.
        lexical, dense = await asyncio.gather(
            self._lexical(query, document_ids, t), self._dense(query, document_ids, t)
        )

        fused = rrf([lexical, dense], k=self.s.rrf_k)[: self.s.rerank_candidates]
        if not fused:
            t["total"] = _ms_since(started)
            return RetrievalResult(query, [], True, "no candidates", 0, t)

        with _Timer(t, "load"):
            async with self.engine.connect() as conn:
                rows = (await conn.execute(_LOAD_CHUNKS, {"ids": [cid for cid, _ in fused]})).all()
        by_id = {r.id: r for r in rows}
        fused = [(cid, score) for cid, score in fused if cid in by_id]  # deleted mid-query
        if not fused:
            t["total"] = _ms_since(started)
            return RetrievalResult(query, [], True, "no candidates", 0, t)

        with _Timer(t, "rerank"):
            logits = await asyncio.to_thread(
                self.reranker.score, query, [by_id[cid].content for cid, _ in fused]
            )

        lex_rank = {cid: i for i, cid in enumerate(lexical, start=1)}
        dense_rank = {cid: i for i, cid in enumerate(dense, start=1)}
        chunks = [
            RetrievedChunk(
                chunk_id=cid,
                document_id=by_id[cid].document_id,
                document_title=by_id[cid].title,
                ordinal=by_id[cid].ordinal,
                page_start=by_id[cid].page_start,
                page_end=by_id[cid].page_end,
                content=by_id[cid].content,
                bm25_rank=lex_rank.get(cid),
                dense_rank=dense_rank.get(cid),
                rrf_score=score,
                rerank_score=float(logit),
                relevance=float(sigmoid(logit)),
            )
            for (cid, score), logit in zip(fused, logits, strict=True)
        ]
        chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        chunks = chunks[:top_k]

        abstained, reason = False, None
        if chunks[0].relevance < self.s.abstain_min_relevance:
            abstained = True
            reason = (
                f"best passage relevance {chunks[0].relevance:.3f} < {self.s.abstain_min_relevance}"
            )

        t["total"] = _ms_since(started)
        return RetrievalResult(query, chunks, abstained, reason, len(fused), t)

    async def _dense(
        self, query: str, document_ids: Sequence[uuid.UUID] | None, t: dict[str, float]
    ) -> list[int]:
        with _Timer(t, "embed"):
            qvec = await asyncio.to_thread(self.embedder.embed_query, query)
        with _Timer(t, "dense"):
            async with self.engine.connect() as conn:
                return await dense_search(
                    conn, qvec, self.s.dense_k, self.s.hnsw_ef_search, document_ids
                )

    async def _lexical(
        self, query: str, document_ids: Sequence[uuid.UUID] | None, t: dict[str, float]
    ) -> list[int]:
        with _Timer(t, "lexical"):
            async with self.engine.connect() as conn:
                return await bm25_search(conn, query, self.s.lexical_k, document_ids)


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
