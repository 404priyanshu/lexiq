"""Streams a grounded answer from Claude or an OpenAI model."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol

import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import Settings
from app.generation.prompt import SYSTEM_PROMPT, build_user_message
from app.retrieval.retriever import RetrievedChunk

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class FallbackNotice:
    """A safety classifier declined mid-answer and another model continued the stream.
    Text already streamed stays valid; it's passed to the fallback model as context."""

    from_model: str | None
    to_model: str | None


@dataclass(frozen=True)
class Completion:
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int


GenEvent = TextDelta | FallbackNotice | Completion


class GenerationError(Exception):
    """The answer model failed; `retryable` when trying again later may succeed."""

    def __init__(self, message: str, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


def _api_error(exc: Exception, sdk: ModuleType) -> GenerationError:
    # Both SDKs name their error classes the same way.
    retryable = isinstance(exc, sdk.RateLimitError | sdk.APIConnectionError) or (
        isinstance(exc, sdk.APIStatusError) and exc.status_code >= 500
    )
    return GenerationError(str(exc), retryable)


class Generator(Protocol):
    def stream(self, question: str, sources: list[RetrievedChunk]) -> AsyncIterator[GenEvent]: ...


def make_generator(settings: Settings) -> "Generator | None":
    """The configured provider's generator, or None when its API key isn't set."""
    key = settings.llm_api_key()
    if not key:
        return None
    if settings.llm_provider == "openai":
        return OpenAIGenerator(AsyncOpenAI(api_key=key), settings)
    return ClaudeGenerator(AsyncAnthropic(api_key=key), settings)


class ClaudeGenerator:
    def __init__(self, client: AsyncAnthropic, settings: Settings):
        self.client = client
        self.s = settings

    async def stream(self, question: str, sources: list[RetrievedChunk]) -> AsyncIterator[GenEvent]:
        extra: dict = {}
        if self.s.llm_fallbacks:
            # On a safety-classifier decline, the API retries on Anthropic's recommended
            # model for that refusal category within the same stream.
            extra = {"betas": [FALLBACK_BETA], "fallbacks": "default"}

        try:
            async with self.client.beta.messages.stream(
                model=self.s.llm_model,
                max_tokens=self.s.llm_max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_message(question, sources)}],
                output_config={"effort": self.s.llm_effort},
                **extra,
            ) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield TextDelta(event.text)
                    elif (
                        event.type == "content_block_start"
                        and event.content_block.type == "fallback"
                    ):
                        block = event.content_block
                        yield FallbackNotice(
                            getattr(getattr(block, "from_", None), "model", None),
                            getattr(getattr(block, "to", None), "model", None),
                        )
                final = await stream.get_final_message()
        except anthropic.APIError as exc:
            raise _api_error(exc, anthropic) from exc

        yield Completion(
            model=final.model,
            stop_reason=final.stop_reason,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )


# OpenAI finish reasons, in the Anthropic stop_reason terms the answer service checks.
_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}


class OpenAIGenerator:
    def __init__(self, client: AsyncOpenAI, settings: Settings):
        self.client = client
        self.s = settings

    async def stream(self, question: str, sources: list[RetrievedChunk]) -> AsyncIterator[GenEvent]:
        extra: dict = {}
        if self.s.openai_reasoning_effort:
            extra["reasoning_effort"] = self.s.openai_reasoning_effort
        model, finish, usage, refused = self.s.llm_model, None, None, False
        try:
            stream = await self.client.chat.completions.create(
                model=self.s.llm_model,
                max_completion_tokens=self.s.llm_max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(question, sources)},
                ],
                stream=True,
                # Usage arrives in one last chunk with no choices.
                stream_options={"include_usage": True},
                **extra,
            )
            async for chunk in stream:
                model = chunk.model or model
                if chunk.usage:
                    usage = chunk.usage
                for choice in chunk.choices:
                    if choice.delta.content:
                        yield TextDelta(choice.delta.content)
                    if choice.delta.refusal:
                        refused = True
                    if choice.finish_reason:
                        finish = choice.finish_reason
        except openai.APIError as exc:
            raise _api_error(exc, openai) from exc

        yield Completion(
            model=model,
            stop_reason="refusal" if refused else _STOP_REASONS.get(finish, finish),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
