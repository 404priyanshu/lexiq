"""Redis caches for retrieval results and finished answers.

Every key includes the corpus version, which each successful ingest bumps, so a cached result
can't outlive the documents it was computed from and nothing needs explicit invalidation.
Keys also include a fingerprint of the settings and prompt that produced the value, so a
config change (or a deploy with a new prompt) never serves results made under the old one.
"""

import hashlib
import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from redis.asyncio import Redis

from app.config import Settings
from app.generation.prompt import SYSTEM_PROMPT
from app.ml.registry import EMBEDDER, RERANKER
from app.retrieval.retriever import RetrievalResult, RetrievedChunk

CORPUS_VERSION_KEY = "lexiq:corpus_version"


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split()).rstrip("?.! ")


async def bump_corpus_version(redis: Redis) -> None:
    await redis.incr(CORPUS_VERSION_KEY)


def _fingerprint(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


class QueryCache:
    def __init__(self, redis: Redis, settings: Settings):
        self.redis = redis
        self.s = settings
        self.retrieval_config = _fingerprint(
            settings.lexical_k, settings.dense_k, settings.hnsw_ef_search, settings.rrf_k,
            settings.rerank_candidates, settings.rerank_max_tokens, settings.reranker_int8,
            settings.abstain_min_relevance, settings.route_to_document,
            settings.route_min_identity_words, EMBEDDER.revision, RERANKER.revision,
        )  # fmt: skip
        self.answer_config = _fingerprint(
            self.retrieval_config, settings.llm_provider, settings.llm_model,
            settings.llm_effort, settings.openai_reasoning_effort, SYSTEM_PROMPT,
        )  # fmt: skip

    async def _key(
        self, kind: str, config: str, query: str, top_k: int | None, document_ids
    ) -> str:
        version = int(await self.redis.get(CORPUS_VERSION_KEY) or 0)
        scope = sorted(str(d) for d in document_ids) if document_ids else None
        return f"lexiq:{kind}:{version}:{config}:" + _fingerprint(
            normalize_query(query), top_k, scope
        )

    async def get_retrieval(
        self, query: str, top_k: int | None, document_ids: Sequence[uuid.UUID] | None
    ) -> RetrievalResult | None:
        key = await self._key("ret", self.retrieval_config, query, top_k, document_ids)
        raw = await self.redis.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        chunks = [
            RetrievedChunk(**(c | {"document_id": uuid.UUID(c["document_id"])}))
            for c in data.pop("chunks")
        ]
        if data.get("routed_document_id"):
            data["routed_document_id"] = uuid.UUID(data["routed_document_id"])
        return RetrievalResult(**data, chunks=chunks)

    async def set_retrieval(
        self,
        query: str,
        top_k: int | None,
        document_ids: Sequence[uuid.UUID] | None,
        result: RetrievalResult,
    ) -> None:
        key = await self._key("ret", self.retrieval_config, query, top_k, document_ids)
        payload = json.dumps(asdict(result) | {"cached": True}, default=str)
        await self.redis.set(key, payload, ex=self.s.cache_retrieval_ttl_s)

    async def get_answer(self, query: str, top_k: int | None, document_ids) -> list | None:
        key = await self._key("ans", self.answer_config, query, top_k, document_ids)
        raw = await self.redis.get(key)
        return json.loads(raw) if raw is not None else None

    async def set_answer(self, query: str, top_k: int | None, document_ids, events: list) -> None:
        key = await self._key("ans", self.answer_config, query, top_k, document_ids)
        await self.redis.set(key, json.dumps(events), ex=self.s.cache_answer_ttl_s)
