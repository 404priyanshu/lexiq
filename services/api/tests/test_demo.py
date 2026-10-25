import uuid

import pytest

from app.config import get_settings
from app.demo import EXAMPLES
from app.limits import BUDGET_MESSAGE, BudgetedGenerator
from app.main import app
from tests.test_generation import FakeGenerator, FakeRetriever, collect, parse_sse


@pytest.fixture
def demo(client, monkeypatch):
    """The public-demo settings, on top of whatever the environment configures."""

    def apply(**overrides):
        s = get_settings().model_copy(update={"read_only": True, **overrides})
        monkeypatch.setattr(app.state, "settings", s)
        return s

    return apply


@pytest.fixture
async def fresh_limits(client):
    for key in await app.state.redis.keys("lexiq:rl:*"):
        await app.state.redis.delete(key)
    for key in await app.state.redis.keys("lexiq:budget:*"):
        await app.state.redis.delete(key)


async def test_read_only_rejects_uploads_and_reingests(client, demo):
    demo()
    upload = await client.post("/documents", files={"file": ("a.txt", b"hello", "text/plain")})
    reingest = await client.post(f"/documents/{uuid.uuid4()}/reingest")
    assert upload.status_code == reingest.status_code == 403
    assert (await client.get("/documents")).status_code == 200


async def test_config_reports_read_only_and_examples(client, demo):
    assert (await client.get("/config")).json() == {"read_only": False, "examples": []}
    demo()
    assert (await client.get("/config")).json() == {"read_only": True, "examples": EXAMPLES}


async def test_answers_are_rate_limited_per_client(client, demo, fresh_limits, monkeypatch):
    demo(answer_limit_per_minute=2)
    monkeypatch.setattr(app.state, "cache", None)
    monkeypatch.setattr(app.state, "retriever", FakeRetriever())
    monkeypatch.setattr(app.state, "generator", FakeGenerator(["Ninety days [1]."]))

    codes = [(await client.post("/answer", json={"query": "q"})).status_code for _ in range(3)]

    assert codes == [200, 200, 429]
    limited = await client.post("/answer", json={"query": "q"})
    assert "Retry-After" in limited.headers and "minute" in limited.json()["detail"]


async def test_search_limit_is_separate_from_answers(client, demo, fresh_limits, monkeypatch):
    demo(search_limit_per_minute=1, answer_limit_per_minute=5)
    monkeypatch.setattr(app.state, "retriever", FakeRetriever())
    assert (await client.post("/search", json={"query": "q"})).status_code == 200
    assert (await client.post("/search", json={"query": "q"})).status_code == 429


async def test_read_only_caps_answer_passages(client, demo, fresh_limits, monkeypatch):
    s = demo()
    seen = {}

    class Recorder(FakeRetriever):
        async def retrieve(self, query, top_k=None, document_ids=None):
            seen["top_k"] = top_k
            return await super().retrieve(query, top_k, document_ids)

    monkeypatch.setattr(app.state, "cache", None)
    monkeypatch.setattr(app.state, "retriever", Recorder())
    monkeypatch.setattr(app.state, "generator", FakeGenerator(["Ninety days [1]."]))
    await client.post("/answer", json={"query": "q", "top_k": 50})
    assert seen["top_k"] == s.top_k
    await client.post("/answer", json={"query": "q2"})
    assert seen["top_k"] == s.top_k


async def test_every_limit_key_gets_its_window_as_ttl(client, demo, fresh_limits, monkeypatch):
    demo(search_limit_per_minute=5)
    monkeypatch.setattr(app.state, "retriever", FakeRetriever())
    for _ in range(2):
        await client.post("/search", json={"query": "q"})
    (key,) = await app.state.redis.keys("lexiq:rl:search:minute:*")
    assert int(await app.state.redis.get(key)) == 2
    assert 0 < await app.state.redis.ttl(key) <= 60


async def test_budget_stops_model_calls_but_still_shows_sources(client, fresh_limits):
    inner = FakeGenerator(["Ninety days [1]."])
    gen = BudgetedGenerator(inner, app.state.redis, daily_limit=1)

    first = await collect(FakeRetriever(), gen)
    second = await collect(FakeRetriever(), gen)

    assert first[-1][0] == "done" and inner.calls == 1
    assert [n for n, _ in second] == ["sources", "error"]
    assert second[-1][1] == {"message": BUDGET_MESSAGE, "retryable": False}
    assert inner.calls == 1


async def test_budget_error_streams_through_the_endpoint(client, fresh_limits, monkeypatch):
    gen = BudgetedGenerator(FakeGenerator(["x [1]."]), app.state.redis, daily_limit=0)
    monkeypatch.setattr(app.state, "cache", None)
    monkeypatch.setattr(app.state, "retriever", FakeRetriever())
    monkeypatch.setattr(app.state, "generator", gen)
    events = parse_sse((await client.post("/answer", json={"query": "q"})).text)
    assert events[-1] == ("error", {"message": BUDGET_MESSAGE, "retryable": False})
