# Evals

A fixed golden set scores every retrieval and prompt change. CI runs it on each pull request
and fails the build when a gated metric drops below `baseline.json` by more than its tolerance.

## Golden set

`golden.jsonl` holds 500 questions about 98 contracts from
[CUAD v1](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus Dataset), whose
answer spans were labelled by lawyers. Each question asks for one clause type in one contract,
for example *"Which jurisdiction's law governs this agreement?"*

- **425 answerable:** the evidence is CUAD's annotated clause text.
- **75 unanswerable:** CUAD records that the contract has no such clause. The right response
  is to abstain.
- **Balanced across 24 clause types**, about 20 answerable questions each, fewer where CUAD has
  fewer examples (Source Code Escrow, No-Solicit Of Customers).
- **Document-scoped:** retrieval is restricted to the contract the question is about, the way
  a user would ask about one agreement.

`corpus.json` lists the contracts and their PDF hashes. The set is regenerated
deterministically (fixed seed) with:

```bash
cd services/api
uv run python -m app.evals.golden --cuad ../../data/cuad/extracted/CUAD_v1
```

A retrieved chunk counts as gold when it contains an evidence span after normalizing case,
punctuation, whitespace and hyphenation. If a span is cut at a chunk boundary, the chunk must
hold at least half of it; if a span is longer than a chunk, at least 90% of the chunk must lie
inside it ([`matching.py`](../services/api/app/evals/matching.py)). Every answerable question's
evidence is locatable in the parsed PDFs, and a median of one chunk per question matches.

## Metrics

| Metric | What it measures | Needs API key |
|---|---|---|
| `recall@k` (k = 1, 3, 5, 8) | Share of answerable questions with a gold chunk in the top k after rerank | no |
| `mrr@10` | Mean reciprocal rank of the first gold chunk | no |
| `candidate_recall` | Gold chunk anywhere in the reranked candidates | no |
| `ablation_recall@5` | recall@5 for BM25 only, dense only, RRF, and RRF + rerank | no |
| `retrieval_abstain` | False-abstain rate (answerable) and catch rate (unanswerable) at the threshold, plus a sweep | no |
| `faithfulness` | LLM judge: every claim in the answer is supported by the given sources | yes |
| `answer_correctness` | LLM judge: the answer addresses the question and agrees with the evidence | yes |
| `abstain_precision` / `abstain_recall` | Abstains (or "not in the contract" answers) vs unanswerable questions | yes |
| `citation_valid_rate`, `citation_coverage` | No out-of-range `[n]`; share of sentences cited | yes |

## Running

```bash
cd services/api
uv run python -m app.evals.corpus                 # ingest the 98 contracts (once)
uv run python -m app.evals.run                    # retrieval metrics, all 500
uv run python -m app.evals.run --answers 100      # + generation and LLM judge on 100
uv run python -m app.evals.gate ../../evals/runs/<run>/summary.json
```

When a change improves the scores on purpose, accept the new numbers by updating the
baseline in the same pull request:

```bash
uv run python -m app.evals.gate ../../evals/runs/<run>/summary.json --update
```

## Attribution

CUAD is © The Atticus Project, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The golden set reuses its contract
names and annotated answer spans; the question wording is ours.
