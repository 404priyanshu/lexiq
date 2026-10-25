import uuid
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[uuid.UUID] | None = None

    @field_validator("query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query is blank")
        return v.strip()


class SearchHit(BaseModel):
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
    relevance: float


class SearchResponse(BaseModel):
    query: str
    abstained: bool
    abstain_reason: str | None
    candidates: int
    timings_ms: dict[str, float]
    chunks: list[SearchHit]
    cached: bool = False
    routed_document_id: uuid.UUID | None = None
    focused_query: str | None = None


@router.post("/search")
async def search(request: Request, body: SearchRequest) -> SearchResponse:
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "retrieval models not loaded; run make models"
        )
    result = await retriever.retrieve(body.query, body.top_k, body.document_ids)
    return SearchResponse.model_validate(asdict(result))
