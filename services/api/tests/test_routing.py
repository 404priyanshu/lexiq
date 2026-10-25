import hashlib

import pytest
from sqlalchemy import insert

from app.config import get_settings, settings_with
from app.db import SessionLocal, engine
from app.models import Chunk, Document
from app.retrieval.retriever import Retriever
from app.retrieval.routing import focus_query

ACME = [
    "SUPPLY AGREEMENT. This Supply Agreement is entered into between Acme Widgets Corp. and "
    "Globex Industries LLC.",
    "Pricing. Prices are fixed for the first contract year and reviewed annually.",
    "Governing Law. This Agreement is governed by the laws of the State of Texas.",
]
INITECH = [
    "SUPPLY AGREEMENT. This Supply Agreement is entered into between Initech Systems Inc. and "
    "Umbrella Holdings Ltd.",
    "Pricing. Prices may change on thirty days' notice.",
    "Governing Law. This Agreement is governed by the laws of the State of Ohio.",
]


def test_focus_query_strips_names_found_in_the_preamble():
    focused, removed = focus_query(
        "In the Supply Agreement between Acme Widgets Corp. and Globex Industries LLC, "
        "which state's law governs the agreement?",
        ACME[0],
    )
    assert focused == "Which state's law governs the agreement?"
    assert removed >= 6


@pytest.mark.parametrize(
    "query",
    ["Which state's law governs this agreement?", "Which contracts include a non-compete?"],
)
def test_focus_query_leaves_questions_that_name_nothing(query):
    assert focus_query(query, ACME[0]) == (query, 0)


def test_focus_query_drops_dangling_agreement_numbers():
    focused, _ = focus_query(
        "In the License Agreement Number 12965 between Fox Mobile Inc., who owns the content?",
        "LICENSE AGREEMENT NUMBER 12965 between Fox Mobile Inc. and Glu Mobile Inc.",
    )
    assert focused == "Who owns the content?"


@pytest.fixture
async def two_contracts(clean_db, models) -> dict[str, Document]:
    docs = {}
    async with SessionLocal() as session:
        for name, passages in (("acme", ACME), ("initech", INITECH)):
            doc = Document(
                title=name, mime_type="text/plain",
                sha256=hashlib.sha256(name.encode()).hexdigest(), status="ready",
            )  # fmt: skip
            session.add(doc)
            await session.flush()
            vectors = models.embedder.embed_documents(passages)
            await session.execute(
                insert(Chunk),
                [
                    {
                        "document_id": doc.id,
                        "ordinal": i,
                        "page_start": 1,
                        "page_end": 1,
                        "char_start": 0,
                        "char_end": len(p),
                        "content": p,
                        "token_count": len(p.split()),
                        "embedding": v,
                    }  # fmt: skip
                    for i, (p, v) in enumerate(zip(passages, vectors, strict=True))
                ],
            )
            docs[name] = doc
        await session.commit()
    return docs


async def test_named_question_routes_to_its_contract_and_finds_the_clause(
    two_contracts, models, reranker
):
    retriever = Retriever(engine, models.embedder, reranker, get_settings())
    result = await retriever.retrieve(
        "In the Supply Agreement between Initech Systems Inc. and Umbrella Holdings Ltd., "
        "which state's law governs the agreement?",
        top_k=3,
    )

    assert result.routed_document_id == two_contracts["initech"].id
    assert result.focused_query == "Which state's law governs the agreement?"
    assert {c.document_id for c in result.chunks} == {two_contracts["initech"].id}
    assert "Ohio" in result.chunks[0].content
    assert {"route", "rerank", "total"} <= set(result.timings_ms)


async def test_generic_question_is_not_routed(two_contracts, models, reranker):
    retriever = Retriever(engine, models.embedder, reranker, get_settings())
    result = await retriever.retrieve("Which state's law governs this agreement?", top_k=4)

    assert result.routed_document_id is None
    assert {c.document_id for c in result.chunks} == {d.id for d in two_contracts.values()}


async def test_routing_can_be_turned_off(two_contracts, models, reranker):
    s = settings_with(["route_to_document=false"])
    retriever = Retriever(engine, models.embedder, reranker, s)
    result = await retriever.retrieve(
        "In the Supply Agreement between Initech Systems Inc. and Umbrella Holdings Ltd., "
        "which state's law governs the agreement?"
    )
    assert result.routed_document_id is None
