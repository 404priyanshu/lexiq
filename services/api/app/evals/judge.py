"""LLM judge for generated answers: faithfulness to the given sources, and whether the answer
addresses the question in line with CUAD's reference evidence."""

from typing import Protocol

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import Settings
from app.generation.generator import FALLBACK_BETA
from app.generation.prompt import build_user_message
from app.retrieval.retriever import RetrievedChunk

JUDGE_SYSTEM = """You grade answers produced by a document question-answering system.

You get the question, the numbered sources the system was given, its answer, and reference \
evidence: clause text a lawyer marked as relevant (empty when the contract has no such clause).

Judge each field independently:
- faithful: true if every factual claim in the answer is supported by the numbered sources. \
Ignore whether citations are formatted correctly; judge support only. An answer that only \
says the sources don't contain the information is faithful.
- unsupported_claims: the claims that are not supported by the sources (empty if none).
- answers_question: true if the answer addresses what was asked and agrees with the reference \
evidence. If the reference evidence is empty, true only if the answer says the information \
isn't in the contract.
- declined: true if the answer says the sources don't contain what was asked, instead of \
answering it."""


class Verdict(BaseModel):
    faithful: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    answers_question: bool
    declined: bool


def build_judge_message(
    question: str, sources: list[RetrievedChunk], answer: str, evidence: list[str]
) -> str:
    reference = "\n\n".join(f"<evidence>\n{e}\n</evidence>" for e in evidence) or "(none)"
    return (
        f"{build_user_message(question, sources)}\n\n"
        f"<answer>\n{answer}\n</answer>\n\n"
        f"<reference_evidence>\n{reference}\n</reference_evidence>"
    )


class Judge(Protocol):
    async def grade(
        self, question: str, sources: list[RetrievedChunk], answer: str, evidence: list[str]
    ) -> tuple[Verdict | None, dict]:
        """Returns (verdict or None if the judge refused, token usage)."""
        ...


def make_judge(settings: Settings) -> Judge | None:
    """The configured provider's judge, or None when its API key isn't set."""
    key = settings.llm_api_key()
    if not key:
        return None
    if settings.llm_provider == "openai":
        return OpenAIJudge(AsyncOpenAI(api_key=key), settings.judge_model)
    return ClaudeJudge(AsyncAnthropic(api_key=key), settings.judge_model)


def _messages(question, sources, answer, evidence) -> list[dict]:
    return [{"role": "user", "content": build_judge_message(question, sources, answer, evidence)}]


class ClaudeJudge:
    def __init__(self, client: AsyncAnthropic, model: str):
        self.client = client
        self.model = model

    async def grade(
        self, question: str, sources: list[RetrievedChunk], answer: str, evidence: list[str]
    ) -> tuple[Verdict | None, dict]:
        response = await self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=JUDGE_SYSTEM,
            messages=_messages(question, sources, answer, evidence),
            output_format=Verdict,
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        if response.stop_reason == "refusal":
            return None, usage
        return response.parsed_output, usage


class OpenAIJudge:
    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def grade(
        self, question: str, sources: list[RetrievedChunk], answer: str, evidence: list[str]
    ) -> tuple[Verdict | None, dict]:
        # Structured outputs: the reply is constrained to Verdict's JSON schema.
        completion = await self.client.chat.completions.parse(
            model=self.model,
            max_completion_tokens=4000,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                *_messages(question, sources, answer, evidence),
            ],
            response_format=Verdict,
        )
        usage = {
            "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
            "output_tokens": completion.usage.completion_tokens if completion.usage else 0,
        }
        message = completion.choices[0].message if completion.choices else None
        if message is None or message.refusal or message.parsed is None:
            return None, usage
        return message.parsed, usage
