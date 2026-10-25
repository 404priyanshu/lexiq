import httpx
import openai
import pytest
from openai.types.chat import ChatCompletionChunk, ParsedChatCompletion

from app.config import OPENAI_DEFAULT_MODEL, Settings
from app.evals.judge import ClaudeJudge, OpenAIJudge, Verdict, make_judge
from app.generation.generator import (
    ClaudeGenerator,
    Completion,
    GenerationError,
    OpenAIGenerator,
    TextDelta,
    make_generator,
)
from tests.test_generation import SOURCES, FakeRetriever, collect


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


# --- provider selection -----------------------------------------------------------------


def test_provider_follows_the_key_that_is_set():
    s = settings(openai_api_key="sk-test")
    assert (s.llm_provider, s.llm_model, s.judge_model) == (
        "openai", OPENAI_DEFAULT_MODEL, OPENAI_DEFAULT_MODEL,
    )  # fmt: skip
    assert isinstance(make_generator(s), OpenAIGenerator)
    assert isinstance(make_judge(s), OpenAIJudge)

    both = settings(openai_api_key="sk-test", anthropic_api_key="sk-ant-test")
    assert both.llm_provider == "anthropic" and both.llm_model.startswith("claude")
    assert isinstance(make_generator(both), ClaudeGenerator)
    assert isinstance(make_judge(both), ClaudeJudge)


def test_explicit_model_wins_and_missing_key_disables_generation():
    s = settings(llm_provider="openai", llm_model="gpt-6-luna")
    assert (s.llm_model, s.judge_model) == ("gpt-6-luna", OPENAI_DEFAULT_MODEL)
    assert make_generator(s) is None and make_judge(s) is None
    assert make_generator(settings()) is None


# --- OpenAI generator -------------------------------------------------------------------


def chunk(content=None, finish=None, refusal=None, usage=None) -> ChatCompletionChunk:
    choices = []
    if content is not None or finish or refusal:
        delta = {"content": content, "refusal": refusal}
        choices = [{"index": 0, "delta": delta, "finish_reason": finish}]
    return ChatCompletionChunk.model_validate(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4.1-mini-2025-04-14",
            "choices": choices,
            "usage": usage,
        }  # fmt: skip
    )


class FakeCompletions:
    def __init__(self, chunks=(), error: Exception | None = None, parsed=None):
        self.chunks, self.error, self.parsed = chunks, error, parsed
        self.kwargs: dict = {}

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error

        async def gen():
            for c in self.chunks:
                yield c

        return gen()

    async def parse(self, **kwargs):
        self.kwargs = kwargs
        return self.parsed


class FakeOpenAI:
    def __init__(self, completions: FakeCompletions):
        self.chat = type("Chat", (), {"completions": completions})()


USAGE = {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940}


async def test_openai_stream_maps_text_usage_and_stop_reason():
    completions = FakeCompletions(
        [
            chunk("Either party may "),
            chunk("terminate [1]."),
            chunk(finish="stop"),
            chunk(usage=USAGE),
        ]
    )
    gen = OpenAIGenerator(FakeOpenAI(completions), settings(llm_provider="openai"))

    events = [e async for e in gen.stream("How can it end?", SOURCES)]

    assert [e.text for e in events if isinstance(e, TextDelta)] == [
        "Either party may ",
        "terminate [1].",
    ]
    assert events[-1] == Completion("gpt-4.1-mini-2025-04-14", "end_turn", 900, 40)
    assert completions.kwargs["messages"][0]["role"] == "system"
    assert completions.kwargs["stream_options"] == {"include_usage": True}
    assert "reasoning_effort" not in completions.kwargs


@pytest.mark.parametrize(
    "final, expected",
    [
        (chunk(finish="length"), "max_tokens"),
        (chunk(finish="content_filter"), "refusal"),
        (chunk(refusal="I can't help with that.", finish="stop"), "refusal"),
    ],
)
async def test_openai_truncation_and_refusals_use_the_service_stop_reasons(final, expected):
    gen = OpenAIGenerator(FakeOpenAI(FakeCompletions([chunk("Partial"), final])), settings())
    events = [e async for e in gen.stream("q", SOURCES)]
    assert events[-1].stop_reason == expected


async def test_openai_reasoning_effort_is_passed_only_when_set():
    completions = FakeCompletions([chunk(finish="stop")])
    s = settings(llm_provider="openai", llm_model="gpt-6-luna", openai_reasoning_effort="low")
    _ = [e async for e in OpenAIGenerator(FakeOpenAI(completions), s).stream("q", SOURCES)]
    assert completions.kwargs["reasoning_effort"] == "low"


def api_error(status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    cls = openai.RateLimitError if status == 429 else openai.APIStatusError
    return cls("boom", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize("status, retryable", [(429, True), (503, True), (401, False)])
async def test_openai_errors_become_generation_errors(status, retryable):
    gen = OpenAIGenerator(FakeOpenAI(FakeCompletions(error=api_error(status))), settings())
    with pytest.raises(GenerationError) as exc:
        _ = [e async for e in gen.stream("q", SOURCES)]
    assert exc.value.retryable is retryable


async def test_answer_service_reports_generation_errors_as_an_error_event():
    gen = OpenAIGenerator(FakeOpenAI(FakeCompletions(error=api_error(429))), settings())
    events = await collect(FakeRetriever(), gen)
    assert [n for n, _ in events] == ["sources", "error"]
    assert events[-1][1]["retryable"] is True


# --- OpenAI judge -----------------------------------------------------------------------


def parsed_completion(verdict: Verdict | None, refusal: str | None = None):
    message = {
        "role": "assistant", "content": verdict.model_dump_json() if verdict else None,
        "refusal": refusal, "parsed": verdict,
    }  # fmt: skip
    return ParsedChatCompletion[Verdict].model_validate(
        {
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4.1-mini",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": USAGE,
        }  # fmt: skip
    )


async def test_openai_judge_returns_the_parsed_verdict_and_usage():
    verdict = Verdict(faithful=True, answers_question=True, declined=False)
    completions = FakeCompletions(parsed=parsed_completion(verdict))
    judge = OpenAIJudge(FakeOpenAI(completions), "gpt-4.1-mini")

    got, usage = await judge.grade("q", SOURCES, "Ninety days [1].", ["ninety days"])

    assert got == verdict
    assert usage == {"input_tokens": 900, "output_tokens": 40}
    assert completions.kwargs["response_format"] is Verdict


async def test_openai_judge_refusal_is_no_verdict():
    completions = FakeCompletions(parsed=parsed_completion(None, refusal="No."))
    got, _ = await OpenAIJudge(FakeOpenAI(completions), "gpt-4.1-mini").grade("q", SOURCES, "a", [])
    assert got is None


async def test_openai_judge_with_no_choices_is_no_verdict():
    empty = parsed_completion(None).model_copy(update={"choices": []})
    got, usage = await OpenAIJudge(FakeOpenAI(FakeCompletions(parsed=empty)), "m").grade(
        "q", SOURCES, "a", []
    )
    assert got is None and usage["input_tokens"] == 900
