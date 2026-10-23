import hashlib

import pytest
from sqlalchemy import insert

from app.config import get_settings
from app.db import SessionLocal, engine
from app.models import Chunk, Document
from app.retrieval.retriever import Retriever
from app.retrieval.search import bm25_search, dense_search, rrf

CORPUS = {
    "Master Services Agreement": [
        "Governing Law. This Agreement shall be governed by and construed in accordance with the "
        "laws of the State of Delaware, without regard to its conflict of laws principles.",
        "Indemnification. Supplier shall defend, indemnify and hold harmless Customer from any "
        "third-party claims arising out of Supplier's gross negligence or willful misconduct.",
        "Termination for Convenience. Either party may terminate this Agreement upon ninety (90) "
        "days' prior written notice to the other party.",
        "Confidentiality. Each party shall keep the other party's Confidential Information "
        "secret for a period of five years after disclosure.",
    ],
    "Office Lease": [
        "Rent. Tenant shall pay monthly base rent of $12,500 on the first day of each month.",
        "Security Deposit. Tenant shall deposit an amount equal to two months' rent, refundable "
        "within 30 days after the lease ends.",
        "Governing Law. This Lease is governed by the laws of the State of New York.",
    ],
}


@pytest.fixture
async def corpus(clean_db, models) -> dict[str, Document]:
    """Insert CORPUS with real embeddings, one chunk per passage on its own page."""
    docs = {}
    async with SessionLocal() as session:
        for title, passages in CORPUS.items():
            doc = Document(
                title=title,
                mime_type="text/plain",
                sha256=hashlib.sha256(title.encode()).hexdigest(),
                page_count=len(passages),
                status="ready",
            )
            session.add(doc)
            await session.flush()
            vectors = models.embedder.embed_documents(passages)
            await session.execute(
                insert(Chunk),
                [
                    {
                        "document_id": doc.id,
                        "ordinal": i,
                        "page_start": i + 1,
                        "page_end": i + 1,
                        "char_start": 0,
                        "char_end": len(p),
                        "content": p,
                        "token_count": len(p.split()),
                        "embedding": v,
                    }
                    for i, (p, v) in enumerate(zip(passages, vectors, strict=True))
                ],
            )
            docs[title] = doc
        await session.commit()
    return docs


@pytest.fixture
def retriever(models, reranker) -> Retriever:
    return Retriever(engine, models.embedder, reranker, get_settings())


def test_rrf_rewards_agreement_between_lists():
    fused = rrf([[1, 2, 3], [3, 4]], k=60)
    ids = [cid for cid, _ in fused]

    assert ids[0] == 3  # in both lists
    assert set(ids) == {1, 2, 3, 4}
    assert dict(fused)[1] == pytest.approx(1 / 61)


def test_reranker_prefers_relevant_passage_and_keeps_input_order(reranker):
    passages = ["The cafeteria serves soup on Fridays.", CORPUS["Office Lease"][0]]
    scores = reranker.score("How much is the monthly rent?", passages)

    assert scores[1] > scores[0]
    assert reranker.score("How much is the monthly rent?", passages[::-1])[0] == pytest.approx(
        scores[1], abs=1e-4
    )


async def test_bm25_and_dense_respect_document_filter(corpus, models):
    lease = corpus["Office Lease"]
    async with engine.connect() as conn:
        lexical = await bm25_search(conn, "governing law", 10, [lease.id])
    async with engine.connect() as conn:
        dense = await dense_search(
            conn, models.embedder.embed_query("governing law"), 10, 40, [lease.id]
        )
    async with SessionLocal() as session:
        for ids in (lexical, dense):
            assert ids
            docs = {(await session.get(Chunk, cid)).document_id for cid in ids}
            assert docs == {lease.id}


async def test_paraphrased_question_finds_clause_without_shared_keywords(corpus, retriever):
    result = await retriever.retrieve("How can a party end the contract early?", top_k=3)

    assert not result.abstained, result.abstain_reason
    top = result.chunks[0]
    assert top.content.startswith("Termination for Convenience")
    assert (top.document_title, top.page_start) == ("Master Services Agreement", 3)
    assert set(result.timings_ms) >= {"embed", "dense", "lexical", "rerank", "total"}


async def test_reranker_resolves_ambiguous_keyword_match(corpus, retriever):
    # Both documents have a "Governing Law" clause; only one is about the lease.
    result = await retriever.retrieve("Which state's law applies to the office lease?", top_k=2)

    assert "New York" in result.chunks[0].content


async def test_off_topic_question_abstains(corpus, retriever):
    result = await retriever.retrieve("What is the boiling point of mercury?")

    assert result.abstained
    assert result.chunks, "abstain still returns the closest passages"


async def test_search_endpoint(client, corpus):
    resp = await client.post("/search", json={"query": "security deposit refund", "top_k": 2})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["chunks"]) == 2
    assert body["chunks"][0]["content"].startswith("Security Deposit")
    assert body["abstained"] is False


@pytest.mark.parametrize("payload", [{"query": "   "}, {"query": "x", "top_k": 0}, {}])
async def test_search_rejects_bad_requests(client, payload):
    assert (await client.post("/search", json=payload)).status_code == 422


async def test_candidates_deleted_mid_query_abstain_instead_of_crashing(
    corpus, retriever, monkeypatch
):
    # Both first-stage searches return a chunk id that no longer exists by load time.
    async def gone(*_args, **_kwargs):
        return [987_654_321]

    monkeypatch.setattr("app.retrieval.retriever.bm25_search", gone)
    monkeypatch.setattr("app.retrieval.retriever.dense_search", gone)

    result = await retriever.retrieve("anything")

    assert result.abstained and result.chunks == []
    assert result.abstain_reason == "no candidates"
