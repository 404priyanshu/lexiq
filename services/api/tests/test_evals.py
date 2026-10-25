import json

import pytest

from app.evals.corpus import load_corpus
from app.evals.gate import check
from app.evals.golden import GOLDEN_PATH, QUESTIONS
from app.evals.matching import first_hit, shares_run, span_in_chunk, words
from app.evals.run import summarize, summarize_answers

CLAUSE = (
    "This Agreement shall be governed by and construed in accordance with the laws of the "
    "State of New York, without regard to its conflict of laws principles."
)

# --- matching ----------------------------------------------------------------------------


def test_contained_span_matches_despite_formatting_differences():
    chunk = "12.  GOVERNING LAW.\nThis Agree-\nment shall be governed by and construed in accordance with the laws of the State of New York, without regard to its conflict of laws principles. 13. Notices."  # noqa: E501
    assert span_in_chunk(CLAUSE, chunk)


def test_span_split_across_chunk_boundary_matches_the_chunk_holding_most_of_it():
    first, second = CLAUSE[:100], CLAUSE[95:]  # ~2/3 and ~1/3 of the words
    assert span_in_chunk(CLAUSE, "Section 11 ... " + first)
    assert not span_in_chunk(CLAUSE, second + " Section 13 Notices shall be in writing.")


def test_shared_boilerplate_alone_does_not_match():
    other = "Each party shall comply with the laws of the State of New York in performing this."
    assert not span_in_chunk(CLAUSE, other)


def test_shares_run_checks_consecutive_words():
    a, b = words("a b c d e"), words("x b c d y")
    assert shares_run(a, b, 3)
    assert not shares_run(a, b, 4)
    assert not shares_run(words("a b"), words("c d"), 1)


def test_first_hit_returns_one_based_rank():
    assert first_hit([CLAUSE], ["unrelated text here", CLAUSE]) == 2
    assert first_hit([CLAUSE], ["unrelated text here"]) is None


# --- golden set integrity (no CUAD download needed) -----------------------------------


def test_golden_set_is_well_formed():
    golden = [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines()]
    contracts = {d["contract"] for d in load_corpus()}

    assert len(golden) == 500
    assert len({g["id"] for g in golden}) == 500
    assert sum(not g["answerable"] for g in golden) == 75
    for g in golden:
        assert g["question"] == QUESTIONS[g["category"]]
        assert g["contract"] in contracts
        assert bool(g["evidence"]) == g["answerable"]


# --- metrics -----------------------------------------------------------------------------


def row(rank, answerable=True, rel=0.9, cat="Governing Law"):
    r = {"answerable": answerable, "category": cat, "top_relevance": rel, "retrieval_ms": 100.0}
    if answerable:
        r["rank"] = {"rerank": rank, "rrf": rank, "bm25": None, "dense": rank}
    return r


def test_retrieval_summary_recall_mrr_and_abstain_rates():
    rows = [row(1), row(3), row(None), row(6), row(None, answerable=False, rel=0.05)]
    s = summarize(rows, abstain_threshold=0.1)

    assert s["retrieval"]["recall@1"] == 0.25
    assert s["retrieval"]["recall@5"] == 0.5
    assert s["retrieval"]["recall@8"] == 0.75
    assert s["retrieval"]["mrr@10"] == pytest.approx((1 + 1 / 3 + 1 / 6) / 4, abs=1e-4)
    assert s["retrieval"]["ablation_recall@5"]["bm25"] == 0.0
    assert s["retrieval_abstain"]["false_abstain_rate"] == 0.0
    assert s["retrieval_abstain"]["unanswerable_abstain_rate"] == 1.0


def gen_row(
    answerable, *, abstained=False, faithful=True, correct=True, declined=False, evidence=None
):
    judge = None
    if not abstained:
        judge = {"faithful": faithful, "unsupported_claims": [], "answers_question": correct,
                 "declined": declined}  # fmt: skip
    return {
        "answerable": answerable,
        "generation": {
            "answer": "" if abstained else "x [1].",
            "abstained": abstained,
            "refused": False,
            "citations": None if abstained else {"cited": [1], "invalid": [], "coverage": 1.0},
            "usage": {"input_tokens": 1000, "output_tokens": 50},
            "total_ms": 1500.0,
            "judge": judge,
        }
        | ({"evidence_in_sources": evidence} if evidence is not None else {}),
    }


def test_answer_summary_counts_judge_declines_as_abstains():
    rows = [
        gen_row(True),
        gen_row(True, faithful=False, correct=False),
        gen_row(True, abstained=True),  # false abstain
        gen_row(False, declined=True, correct=True),  # model said "not in the contract"
        gen_row(False, abstained=True),
    ]
    s = summarize_answers(rows)

    assert s["faithfulness"] == pytest.approx(2 / 3, abs=1e-4)  # 3 judged answers, 1 unfaithful
    assert s["answer_correctness"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["abstain_recall"] == 1.0
    assert s["abstain_precision"] == pytest.approx(2 / 3, abs=1e-4)


def test_answer_summary_leaves_failed_generations_out_of_the_metrics():
    rows = [
        gen_row(True),
        gen_row(False, abstained=True),
        {"answerable": True, "generation": {"error": "rate limited"}},
    ]
    s = summarize_answers(rows)
    assert (s["n"], s["errors"]) == (2, 1)
    assert s["answer_correctness"] == 1.0 and s["abstain_recall"] == 1.0


def test_answer_summary_separates_declines_the_sources_justified():
    rows = [
        gen_row(True, evidence=True),
        gen_row(True, abstained=True, evidence=True),  # model error: clause was in the sources
        gen_row(True, abstained=True, evidence=False),  # retrieval miss: declining was right
        gen_row(False, abstained=True),
    ]
    s = summarize_answers(rows)["given_evidence"]
    assert s == {"n": 2, "false_abstain_rate": 0.5, "answer_correctness": 0.5}


# --- gate --------------------------------------------------------------------------------


def test_gate_passes_within_tolerance_fails_beyond_and_skips_missing():
    baseline = {
        "metrics": {
            "retrieval.recall@5": {"value": 0.80, "tolerance": 0.02},
            "retrieval.mrr@10": {"value": 0.60, "tolerance": 0.02},
            "answers.faithfulness": {"value": 0.95, "tolerance": 0.03},
        }
    }
    ok, rows = check({"retrieval": {"recall@5": 0.79, "mrr@10": 0.55}}, baseline)

    assert not ok
    verdicts = {m: r for m, _, _, r in rows}
    assert verdicts["retrieval.recall@5"] == "ok"
    assert verdicts["retrieval.mrr@10"].startswith("FAIL")
    assert verdicts["answers.faithfulness"] == "skipped"


def test_gate_lower_is_better_metrics_fail_on_rise():
    baseline = {"metrics": {"retrieval_abstain.false_abstain_rate": {
        "value": 0.01, "tolerance": 0.01, "direction": "lower"}}}  # fmt: skip
    assert check({"retrieval_abstain": {"false_abstain_rate": 0.015}}, baseline)[0]
    assert not check({"retrieval_abstain": {"false_abstain_rate": 0.05}}, baseline)[0]


@pytest.mark.parametrize("answerable", [True, False])
def test_one_sided_subsets_report_none_instead_of_crashing(answerable, capsys):
    from app.evals.run import print_summary

    rows = [row(2 if answerable else None, answerable=answerable) for _ in range(3)]
    s = summarize(rows, abstain_threshold=0.1)
    print_summary(s)

    rates = s["retrieval_abstain"]
    if answerable:
        assert s["retrieval"]["recall@5"] == 1.0
        assert rates["unanswerable_abstain_rate"] is None
    else:
        assert s["retrieval"]["recall@5"] is None and s["retrieval"]["mrr@10"] is None
        assert rates["false_abstain_rate"] is None
    assert "n/a" in capsys.readouterr().out
