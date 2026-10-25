import pytest

from app.generation.declines import declines_in_prose
from tests.test_generation import FakeGenerator, FakeRetriever, collect


@pytest.mark.parametrize(
    "answer",
    [
        "The sources do not mention any liquidated damages provision.",
        "The provided excerpts do not specify a minimum purchase. They do set prices [2].",
        "The agreement does not explicitly impose any non-compete restrictions.",
        "The sources provided do not contain any information about insurance.",
        "The cited provisions provide no liquidated-damages amount or formula.",
        "The excerpts do not state a general non-compete restriction. It does limit use [3].",
    ],
)
def test_detects_declines_written_as_prose(answer):
    assert declines_in_prose(answer)


@pytest.mark.parametrize(
    "answer",
    [
        # A claim about what the agreement does, not about what the sources say.
        "The agreement does not permit assignment without consent [2].",
        "The agreement does not require any minimum purchase commitment.",
        "Neither party may assign the agreement without written consent [1].",
        # Cited: the model is answering from a source.
        "The sources do not state a cap, but liability excludes indirect damages [4].",
        "The sources do not state a cap, but liability excludes indirect damages [1, 2].",
        # The decline comes after an answer.
        "Licensee may audit once a year [1]. The sources do not specify the notice period.",
        "Either party may terminate on ninety days' notice [1].",
    ],
)
def test_leaves_answers_alone(answer):
    assert not declines_in_prose(answer)


async def test_prose_decline_is_an_abstain_that_keeps_its_text():
    text = "The sources do not mention a non-compete. They restrict use to internal purposes [1]."
    events = await collect(FakeRetriever(), FakeGenerator([text]))
    done = events[-1][1]
    assert done["abstained"] and done["abstain_reason"].startswith("model:")
    assert done["answer"] == text and done["citations"]["cited"] == [1]
