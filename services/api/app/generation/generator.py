"""Streams a grounded answer from Claude."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from anthropic import AsyncAnthropic

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


class Generator(Protocol):
    def stream(self, question: str, sources: list[RetrievedChunk]) -> AsyncIterator[GenEvent]: ...


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
                elif event.type == "content_block_start" and event.content_block.type == "fallback":
                    block = event.content_block
                    yield FallbackNotice(
                        getattr(getattr(block, "from_", None), "model", None),
                        getattr(getattr(block, "to", None), "model", None),
                    )
            final = await stream.get_final_message()

        yield Completion(
            model=final.model,
            stop_reason=final.stop_reason,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )
