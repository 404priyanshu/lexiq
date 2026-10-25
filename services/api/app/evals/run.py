"""Run the golden-set eval and write evals/runs/<timestamp>/{results.jsonl,summary.json}.

Retrieval metrics run on every question and need no API key. With --answers N, the first N
questions (the golden file is pre-shuffled, so this is a stable mixed subset) also go through
generation and the LLM judge.

Usage:
  uv run python -m app.evals.run                    # retrieval, all 500 questions
  uv run python -m app.evals.run --answers 100      # + generation and judge on 100
  uv run python -m app.evals.run --answers 100 --answers-from 100   # tuning slice
"""

import argparse
import asyncio
import json
import statistics
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.config import REPO_ROOT, Settings, get_settings, settings_with
from app.db import engine
from app.evals.corpus import document_ids_by_sha, load_corpus
from app.evals.golden import GOLDEN_PATH
from app.evals.golden_corpus import CORPUS_GOLDEN_PATH
from app.evals.judge import make_judge
from app.evals.matching import first_hit
from app.generation.generator import make_generator
from app.generation.service import answer_events
from app.models import DocumentStatus
from app.retrieval.retriever import RetrievalResult, Retriever

RUNS_DIR = REPO_ROOT / "evals" / "runs"
RECALL_KS = (1, 3, 5, 8)
ABSTAIN_SWEEP = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3)
_T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - _T0:6.1f}s] {msg}", flush=True)


class _Replay:
    """Retriever stand-in that hands answer_events the retrieval we already ran."""

    def __init__(self, result: RetrievalResult, top_k: int):
        self.result, self.top_k = result, top_k

    async def retrieve(self, *_args, **_kwargs) -> RetrievalResult:
        r = self.result
        return RetrievalResult(
            r.query, r.chunks[: self.top_k], r.abstained, r.abstain_reason, r.candidates,
            r.timings_ms,
        )  # fmt: skip


def _rank_by(result: RetrievalResult, key, target: uuid.UUID) -> list[str]:
    """Chunk texts in the order `key` ranks them. Chunks from any other document are blanked,
    so evidence only counts when it comes from the contract the question is about."""
    ranked = [c for c in result.chunks if key(c) is not None]
    return [c.content if c.document_id == target else "" for c in sorted(ranked, key=key)]


def _recall(ranks: list[int | None], k: int) -> float | None:
    return _rate(ranks, lambda r: r is not None and r <= k)


def _pct(values: list[float], q: float) -> float:
    return round(statistics.quantiles(values, n=100)[int(q) - 1], 1) if len(values) > 1 else 0.0


async def evaluate(
    answer_limit: int,
    concurrency: int,
    limit: int | None = None,
    settings: Settings | None = None,
    scope: str = "document",
    answer_offset: int = 0,
) -> tuple[list[dict], dict]:
    """scope="document": golden questions, retrieval restricted to their contract.
    scope="corpus": contract-named questions, retrieval over every document.
    scope="unnamed": golden questions over every document; they name no contract, so this
    measures how often two-stage retrieval wrongly narrows a question to one document."""
    s = settings or get_settings()
    path = CORPUS_GOLDEN_PATH if scope == "corpus" else GOLDEN_PATH
    golden = [json.loads(line) for line in path.read_text().splitlines()][:limit]
    docs = await document_ids_by_sha()
    doc_for = {}
    for d in load_corpus():
        doc_id, status = docs.get(d["sha256"], (None, None))
        if status != DocumentStatus.ready:
            raise SystemExit(f"corpus not ingested: {d['contract']} (run app.evals.corpus)")
        doc_for[d["contract"]] = doc_id

    log(f"loading models for {len(golden)} questions")
    retriever = Retriever.from_settings(engine, s)
    generator = judge = None
    if answer_limit:
        generator, judge = make_generator(s), make_judge(s)
        if generator is None or judge is None:
            raise SystemExit("--answers needs ANTHROPIC_API_KEY or OPENAI_API_KEY")

    sem = asyncio.Semaphore(concurrency)
    done_count = 0
    started = time.perf_counter()

    async def one(i: int, g: dict) -> dict:
        nonlocal done_count
        async with sem:
            # Ask for every reranked candidate so recall@k can be read at any k.
            target = doc_for[g["contract"]]
            res = await retriever.retrieve(
                g["question"],
                top_k=s.rerank_candidates,
                document_ids=[target] if scope == "document" else None,
            )
        done_count += 1
        if res.timings_ms.get("total", 0) > 3000:
            log(f"slow retrieval {g['id']}: {res.timings_ms}")
        if done_count % 25 == 0 or done_count == len(golden):
            rate = done_count / (time.perf_counter() - started)
            log(f"retrieved {done_count}/{len(golden)} ({rate:.1f} q/s)")
        row = {
            "id": g["id"],
            "category": g["category"],
            "answerable": g["answerable"],
            "top_relevance": res.chunks[0].relevance if res.chunks else 0.0,
            "retrieval_abstained": res.abstained,
            "retrieval_ms": res.timings_ms.get("total"),
        }
        if g["answerable"]:
            reranked = [c.content if c.document_id == target else "" for c in res.chunks]
            row["rank"] = {
                "rerank": first_hit(g["evidence"], reranked),
                "rrf": first_hit(g["evidence"], _rank_by(res, lambda c: -c.rrf_score, target)),
                "bm25": first_hit(g["evidence"], _rank_by(res, lambda c: c.bm25_rank, target)),
                "dense": first_hit(g["evidence"], _rank_by(res, lambda c: c.dense_rank, target)),
            }
            if scope != "document":
                row["routed"] = res.routed_document_id is not None
                row["routed_to_target"] = res.routed_document_id == target
                # Did retrieval find the right contract at all, whatever the clause?
                row["doc_rank"] = next(
                    (i for i, c in enumerate(res.chunks, start=1) if c.document_id == target), None
                )
        if generator and answer_offset <= i < answer_offset + answer_limit:
            row["generation"] = await generate_and_judge(g, res)
            if g["answerable"]:
                # Was the gold clause among the passages the model saw? Declining is the right
                # call when it wasn't, so this separates model errors from retrieval misses.
                rank = row["rank"]["rerank"]
                row["generation"]["evidence_in_sources"] = rank is not None and rank <= s.top_k
        return row

    async def generate_and_judge(g: dict, res: RetrievalResult) -> dict:
        async with sem:
            events = [
                e
                async for e in answer_events(
                    _Replay(res, s.top_k), generator, g["question"], s.top_k
                )
            ]
        sources = res.chunks[: s.top_k]
        done = next((data for name, data in events if name == "done"), None)
        if done is None:
            # Generation failed after the SDK's own retries; keep the run going and leave this
            # question out of the answer metrics rather than scoring it as wrong.
            error = next((data for name, data in events if name == "error"), {})
            log(f"generation failed for {g['id']}: {error.get('message')}")
            return {"error": error.get("message", "no done event")}
        out = {
            "answer": done["answer"],
            "abstained": done["abstained"],
            "refused": done["refused"],
            "citations": done["citations"],
            "usage": done["usage"],
            "ttft_ms": done["timings_ms"].get("ttft"),
            "total_ms": done["timings_ms"].get("total"),
        }
        if done["answer"]:
            async with sem:
                verdict, usage = await judge.grade(
                    g["question"], sources, done["answer"], g["evidence"]
                )
            out["judge"] = verdict.model_dump() if verdict else None
            out["judge_usage"] = usage
        return out

    rows = await asyncio.gather(*(one(i, g) for i, g in enumerate(golden)))
    log("all questions done")
    summary = summarize(rows, s.abstain_min_relevance)
    if "answers" in summary:
        summary["answers"]["models"] = {"answer": s.llm_model, "judge": s.judge_model}
        summary["answers"]["questions"] = [answer_offset, answer_offset + answer_limit]
    summary["elapsed_s"] = round(time.perf_counter() - started, 1)
    return rows, summary


def summarize(rows: list[dict], abstain_threshold: float) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    ranks = {m: [r["rank"][m] for r in answerable] for m in ("rerank", "rrf", "bm25", "dense")}
    final = ranks["rerank"]

    retrieval = {f"recall@{k}": _recall(final, k) for k in RECALL_KS} | {
        "mrr@10": round(sum(1 / r for r in final if r and r <= 10) / len(final), 4)
        if final
        else None,
        "candidate_recall": _recall(final, 10**6),
        "ablation_recall@5": {m: _recall(v, 5) for m, v in ranks.items()},
        "latency_ms": {
            "p50": _pct([r["retrieval_ms"] for r in rows], 50),
            "p95": _pct([r["retrieval_ms"] for r in rows], 95),
        },
    }

    # Rates are None when a subset (e.g. --limit) has no questions of that kind.
    def abstain_rates(threshold: float) -> dict:
        return {
            "false_abstain_rate": _rate(answerable, lambda r: r["top_relevance"] < threshold),
            "unanswerable_abstain_rate": _rate(
                unanswerable, lambda r: r["top_relevance"] < threshold
            ),
        }

    if any("doc_rank" in r for r in answerable):
        doc_ranks = [r.get("doc_rank") for r in answerable]
        retrieval["document_recall"] = {f"@{k}": _recall(doc_ranks, k) for k in RECALL_KS}
        retrieval["routing"] = {
            "routed_rate": _rate(answerable, lambda r: r["routed"]),
            "routed_to_target_rate": _rate(answerable, lambda r: r["routed_to_target"]),
        }

    summary = {
        "questions": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "retrieval": retrieval,
        "retrieval_abstain": abstain_rates(abstain_threshold)
        | {
            "threshold": abstain_threshold,
            "sweep": {str(t): abstain_rates(t) for t in ABSTAIN_SWEEP},
        },
        "by_category_recall@5": {
            cat: _recall([r["rank"]["rerank"] for r in answerable if r["category"] == cat], 5)
            for cat in sorted({r["category"] for r in answerable})
        },
    }

    gen = [r for r in rows if "generation" in r]
    if gen:
        summary["answers"] = summarize_answers(gen)
    return summary


def _rate(items: list, pred) -> float | None:
    return round(sum(1 for x in items if pred(x)) / len(items), 4) if items else None


def summarize_answers(rows: list[dict]) -> dict:
    errors = sum(1 for r in rows if "error" in r["generation"])
    rows = [r for r in rows if "error" not in r["generation"]]
    gens = [r["generation"] for r in rows]
    judged = [g["judge"] for g in gens if g.get("judge")]
    cited = [g["citations"] for g in gens if g["citations"]]
    usage = [g["usage"] for g in gens if g["usage"]]
    totals = [g["total_ms"] for g in gens if g["total_ms"]]

    def abstained(r: dict) -> bool:
        g = r["generation"]
        return g["abstained"] or bool(g.get("judge") and g["judge"]["declined"])

    def correct(r: dict) -> bool:
        j = r["generation"].get("judge")
        return bool(j and j["answers_question"] and not j["declined"])

    given_evidence = [r for r in rows if r["generation"].get("evidence_in_sources")]

    return {
        "n": len(rows),
        "errors": errors,
        "faithfulness": _rate(judged, lambda j: j["faithful"]),
        "answer_correctness": _rate([r for r in rows if r["answerable"]], correct),
        "abstain_precision": _rate(
            [r for r in rows if abstained(r)], lambda r: not r["answerable"]
        ),
        "abstain_recall": _rate([r for r in rows if not r["answerable"]], abstained),
        # Answerable questions whose gold clause was in the sources the model saw.
        "given_evidence": {
            "n": len(given_evidence),
            "false_abstain_rate": _rate(given_evidence, abstained),
            "answer_correctness": _rate(given_evidence, correct),
        },
        "citation_valid_rate": _rate(cited, lambda c: not c["invalid"]),
        "citation_coverage": round(statistics.mean(c["coverage"] for c in cited), 4)
        if cited
        else None,
        "refusals": sum(g["refused"] for g in gens),
        "latency_ms": {"p50": _pct(totals, 50), "p95": _pct(totals, 95)} if totals else None,
        "mean_tokens": {
            "input": round(statistics.mean(u["input_tokens"] for u in usage)),
            "output": round(statistics.mean(u["output_tokens"] for u in usage)),
        }
        if usage
        else None,
    }


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def print_summary(summary: dict) -> None:
    r = summary["retrieval"]
    print(f"\n{summary['questions']} questions ({summary['answerable']} answerable)")
    print("retrieval   " + "  ".join(f"recall@{k} {_f(r[f'recall@{k}'])}" for k in RECALL_KS))
    print(f"            mrr@10 {_f(r['mrr@10'])}  candidate recall {_f(r['candidate_recall'])}")
    print("ablation    " + "  ".join(f"{m} {_f(v)}" for m, v in r["ablation_recall@5"].items()))
    print(f"latency     p50 {r['latency_ms']['p50']} ms  p95 {r['latency_ms']['p95']} ms")
    a = summary["retrieval_abstain"]
    print(
        f"abstain     @{a['threshold']}: false-abstain {_f(a['false_abstain_rate'])}, "
        f"unanswerable caught {_f(a['unanswerable_abstain_rate'])}"
    )
    if "document_recall" in r:
        dr = r["document_recall"]
        print("right doc   " + "  ".join(f"recall{k} {_f(v)}" for k, v in dr.items()))
        rt = r["routing"]
        print(f"routing     routed {_f(rt['routed_rate'])}  to the named contract "
              f"{_f(rt['routed_to_target_rate'])}")  # fmt: skip
    if "answers" in summary:
        print("answers     " + json.dumps(summary["answers"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--answers", type=int, default=0, metavar="N")
    parser.add_argument(
        "--answers-from", type=int, default=0, metavar="I",
        help="answer questions I..I+N-1 instead of the first N (e.g. a held-out tuning slice)",
    )  # fmt: skip
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument(
        "--scope", choices=["document", "corpus", "unnamed"], default="document",
        help="document: golden set within each contract; corpus: contract-named questions "
        "over every document (evals/golden_corpus.jsonl); unnamed: golden set over every "
        "document, to measure false routing",
    )  # fmt: skip
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a Settings field for this run, e.g. --set rerank_candidates=15",
    )  # fmt: skip
    args = parser.parse_args()
    settings = settings_with(args.overrides)

    async def run():
        try:
            return await evaluate(
                args.answers, args.concurrency, args.limit, settings, args.scope,
                args.answers_from,
            )  # fmt: skip
        finally:
            await engine.dispose()

    rows, summary = asyncio.run(run())
    summary["scope"] = args.scope
    summary["overrides"] = args.overrides
    out = args.out or RUNS_DIR / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print_summary(summary)
    print(f"\nwrote {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}")


if __name__ == "__main__":
    main()
