"""Fail when an eval run regresses against evals/baseline.json.

Each baseline metric is a dotted path into summary.json with the baseline value and an allowed
drop. Metrics are higher-is-better unless the spec says "direction": "lower", in which case
the tolerance is the allowed rise. Metrics missing from the run (e.g. answer metrics
when no API key was available) are reported as skipped, not failed. Latency is not gated: CI
hardware is too noisy for it.

Usage:
  uv run python -m app.evals.gate evals/runs/<run>/summary.json
  uv run python -m app.evals.gate evals/runs/<run>/summary.json --update   # accept new values
"""

import argparse
import json
import os
import sys
from pathlib import Path

from app.config import REPO_ROOT

BASELINE_PATH = REPO_ROOT / "evals" / "baseline.json"


def lookup(summary: dict, path: str):
    node = summary
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def check(summary: dict, baseline: dict) -> tuple[bool, list[tuple[str, str, str, str]]]:
    ok, rows = True, []
    for path, spec in baseline["metrics"].items():
        current = lookup(summary, path)
        base, tol = spec.get("value"), spec["tolerance"]
        if current is None or base is None:
            rows.append((path, "-" if base is None else f"{base:.3f}", "-", "skipped"))
            continue
        lower_is_better = spec.get("direction") == "lower"
        passed = current <= base + tol if lower_is_better else current >= base - tol
        ok &= passed
        verdict = (
            "ok" if passed else f"FAIL (allowed {'rise' if lower_is_better else 'drop'} {tol})"
        )
        rows.append((path, f"{base:.3f}", f"{current:.3f}", verdict))
    return ok, rows


def render(rows: list[tuple[str, str, str, str]]) -> str:
    lines = ["| metric | baseline | current | result |", "|---|---|---|---|"]
    lines += [f"| {m} | {b} | {c} | {r} |" for m, b, c, r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("summary", type=Path)
    parser.add_argument("--update", action="store_true", help="write current values as baseline")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    baseline = json.loads(BASELINE_PATH.read_text())

    if args.update:
        for path, spec in baseline["metrics"].items():
            value = lookup(summary, path)
            if value is not None:
                spec["value"] = round(value, 4)
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"updated {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return

    ok, rows = check(summary, baseline)
    table = render(rows)
    print(table)
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(step_summary, "a") as f:
            f.write(f"### Eval gate: {'passed' if ok else 'FAILED'}\n\n{table}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
