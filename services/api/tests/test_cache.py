import pytest

from app.cache import QueryCache, bump_corpus_version, normalize_query
from app.config import get_settings, settings_with
from app.db import engine
from app.generation.service import answer_events
from app.main import app
from app.retrieval.retriever import Retriever
from tests.test_generation import FakeGenerator, FakeRetriever, parse_sse


@pytest.fixture
def cache(client) -> QueryCache:
    return QueryCache(app.state.redis, get_settings())


def test_normalize_query_ignores_case_spacing_and_trailing_punctuation():
    assert normalize_query("  How can a party END the contract early?? ") == normalize_query(
        "how can a party end the contract early"
    )


async def test_retrieval_cache_hits_repeat_and_reformatted_question(
    corpus, models, reranker, cache
):
    retriever = Retriever(engine, models.embedder, reranker, get_settings(), cache)
    await bump_corpus_version(app.state.redis)  # isolate from earlier tests

    first = await retriever.retrieve("How can a party end the contract early?", top_k=3)
    again = await retriever.retrieve("  how can a party END the contract early ", top_k=3)

    assert not first.cached and again.cached
    assert [c.chunk_id for c in again.chunks] == [c.chunk_id for c in first.chunks]
    assert again.chunks[0].document_id == first.chunks[0].document_id
    assert set(again.timings_ms) == {"cache", "total"}


async def test_corpus_version_bump_and_config_change_miss(corpus, models, reranker, cache):
    retriever = Retriever(engine, models.embedder, reranker, get_settings(), cache)
    q = "What is the monthly rent?"
    await retriever.retrieve(q)
    assert (await retriever.retrieve(q)).cached

    await bump_corpus_version(app.state.redis)
    assert not (await retriever.retrieve(q)).cached

    other = settings_with([f"rerank_candidates={get_settings().rerank_candidates + 5}"])
    other_retriever = Retriever(
        engine, models.embedder, reranker, other, QueryCache(app.state.redis, other)
    )
    assert not (await other_retriever.retrieve(q)).cached


async def collect(gen, cache, question="How can the contract end?"):
    return [e async for e in answer_events(FakeRetriever(), gen, question, cache=cache)]


async def test_answer_cache_replays_without_calling_the_model(cache):
    await bump_corpus_version(app.state.redis)
    gen = FakeGenerator(["Either party may terminate", " on ninety days' notice [1]."])

    first = await collect(gen, cache)
    second = await collect(gen, cache)

    assert gen.calls == 1
    assert [n for n, _ in second] == ["sources", "delta", "done"]
    assert second[1][1]["text"] == first[-1][1]["answer"]
    assert second[-1][1]["cached"] and not first[-1][1]["cached"]
    assert second[0] == first[0]  # same numbered sources


async def test_refusals_are_not_cached(cache):
    await bump_corpus_version(app.state.redis)
    gen = FakeGenerator(["Partial"], stop_reason="refusal")

    await collect(gen, cache)
    await collect(gen, cache)

    assert gen.calls == 2


async def test_answer_endpoint_serves_repeat_from_cache(client, monkeypatch):
    await bump_corpus_version(app.state.redis)
    gen = FakeGenerator(["Delaware law governs [2]."])
    monkeypatch.setattr(app.state, "retriever", FakeRetriever())
    monkeypatch.setattr(app.state, "generator", gen)
    monkeypatch.setattr(app.state, "cache", QueryCache(app.state.redis, get_settings()))

    body = {"query": "Which law governs this agreement?"}
    first = parse_sse((await client.post("/answer", json=body)).text)
    second = parse_sse((await client.post("/answer", json=body)).text)

    assert gen.calls == 1
    assert second[-1][1]["cached"] and second[-1][1]["answer"] == first[-1][1]["answer"]


async def test_abstained_answers_are_cached_until_the_corpus_changes(cache):
    await bump_corpus_version(app.state.redis)
    gen = FakeGenerator(["unused"])
    weak = FakeRetriever(abstained=True)

    async def ask():
        return [e async for e in answer_events(weak, gen, "Best pizza topping?", cache=cache)]

    first, second = await ask(), await ask()
    assert first[-1][1]["abstained"] and second[-1][1]["cached"]

    await bump_corpus_version(app.state.redis)  # e.g. a new document was ingested
    assert not (await ask())[-1][1]["cached"]
