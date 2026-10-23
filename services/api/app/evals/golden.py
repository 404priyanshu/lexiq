"""Build the golden set from CUAD's lawyer annotations.

CUAD (Atticus Project, CC BY 4.0) labels 41 clause types in 510 contracts with answer spans,
and records when a clause type is absent. Each golden question asks for one clause's details
in one contract. Answerable questions carry CUAD's spans as evidence; questions about a clause
the contract doesn't have are unanswerable, and the system should abstain.

Questions are document-scoped: the evaluator restricts retrieval to that contract, as a user
asking about a specific agreement would.

Usage: uv run python -m app.evals.golden --cuad ../../data/cuad/extracted/CUAD_v1
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from app.config import REPO_ROOT

EVALS_DIR = REPO_ROOT / "evals"
GOLDEN_PATH = EVALS_DIR / "golden.jsonl"
CORPUS_PATH = EVALS_DIR / "corpus.json"

CUAD_ZIP_MD5 = "c38f490a984420b8a62600db401fafd5"
CUAD_URL = "https://zenodo.org/api/records/4595826/files/CUAD_v1.zip/content"

# CUAD category -> a question a user would ask. Phrased to ask for the clause's details, so
# "the contract doesn't say" is the right response when the clause is absent.
QUESTIONS = {
    "Governing Law": "Which jurisdiction's law governs this agreement?",
    "Expiration Date": "When does the initial term of this agreement end?",
    "Renewal Term": "How does this agreement renew after its initial term, and for how long?",
    "Notice Period To Terminate Renewal": (
        "How much notice must a party give to stop this agreement from renewing?"
    ),
    "Termination For Convenience": (
        "Can a party terminate this agreement for convenience, without cause, and on what notice?"
    ),
    "Anti-Assignment": "What restrictions apply to assigning this agreement?",
    "Change Of Control": (
        "What happens under this agreement if a party undergoes a change of control?"
    ),
    "Cap On Liability": "How is each party's liability capped under this agreement?",
    "Liquidated Damages": "What liquidated damages does this agreement provide for?",
    "Audit Rights": "What rights does a party have to audit the other party's books or records?",
    "Insurance": "What insurance must a party maintain under this agreement?",
    "Warranty Duration": "How long do the warranties in this agreement last?",
    "Exclusivity": "What exclusivity obligations does this agreement impose?",
    "Non-Compete": "What non-compete restrictions does this agreement impose?",
    "No-Solicit Of Employees": (
        "What restrictions apply to soliciting or hiring the other party's employees?"
    ),
    "No-Solicit Of Customers": "What restrictions apply to soliciting the other party's customers?",
    "Minimum Commitment": "What minimum purchase or order commitment does this agreement require?",
    "Revenue/Profit Sharing": "How are revenues or profits shared between the parties?",
    "Rofr/Rofo/Rofn": (
        "What right of first refusal, first offer or first negotiation does a party have?"
    ),
    "License Grant": "What license does this agreement grant, and to whom?",
    "Ip Ownership Assignment": "Who owns intellectual property created under this agreement?",
    "Post-Termination Services": "What obligations continue after this agreement ends?",
    "Volume Restriction": "What limits does this agreement place on volumes or quantities?",
    "Source Code Escrow": "What source code escrow arrangements does this agreement set up?",
}

SEED = 20261023
NUM_CONTRACTS = 100
NUM_ANSWERABLE = 425
NUM_UNANSWERABLE = 75
# Very short spans ("Delaware") match too many chunks to say which passage was meant.
MIN_EVIDENCE_CHARS = 25


def _pdf_index(cuad: Path) -> dict[str, Path]:
    return {
        p.stem.strip(): p
        for p in (cuad / "full_contract_pdf").rglob("*")
        if p.suffix.lower() == ".pdf"
    }


def _round_robin(by_category: dict[str, list], n: int, rng: random.Random) -> list:
    """Take one item per category in turn so common clause types don't dominate."""
    pools = {cat: rng.sample(items, len(items)) for cat, items in sorted(by_category.items())}
    out: list = []
    while len(out) < n and any(pools.values()):
        for cat in sorted(pools):
            if pools[cat] and len(out) < n:
                out.append(pools[cat].pop())
    return out


def build(cuad: Path) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    data = json.loads((cuad / "CUAD_v1.json").read_text())["data"]
    pdfs = _pdf_index(cuad)
    contracts = sorted(d["title"] for d in data if d["title"].strip() in pdfs)
    chosen = set(rng.sample(contracts, NUM_CONTRACTS))

    answerable: dict[str, list] = defaultdict(list)
    unanswerable: dict[str, list] = defaultdict(list)
    for doc in data:
        title = doc["title"]
        if title not in chosen:
            continue
        for qa in doc["paragraphs"][0]["qas"]:
            category = qa["id"].split("__")[-1]
            if category not in QUESTIONS:
                continue
            if qa["is_impossible"]:
                unanswerable[category].append((title, category, []))
                continue
            spans = sorted({a["text"].strip() for a in qa["answers"]})
            if max(len(s) for s in spans) >= MIN_EVIDENCE_CHARS:
                answerable[category].append((title, category, spans))

    picked = _round_robin(answerable, NUM_ANSWERABLE, rng) + _round_robin(
        unanswerable, NUM_UNANSWERABLE, rng
    )
    rng.shuffle(picked)

    golden = [
        {
            "id": f"q{i:04d}",
            "question": QUESTIONS[category],
            "category": category,
            "answerable": bool(spans),
            "contract": title,
            "evidence": spans,
        }
        for i, (title, category, spans) in enumerate(picked, start=1)
    ]
    used = sorted({g["contract"] for g in golden})
    corpus = [
        {
            "contract": title,
            "pdf": str(pdfs[title.strip()].relative_to(cuad)),
            "sha256": hashlib.sha256(pdfs[title.strip()].read_bytes()).hexdigest(),
        }
        for title in used
    ]
    return golden, corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cuad", type=Path, required=True, help="extracted CUAD_v1 directory")
    args = parser.parse_args()

    golden, corpus = build(args.cuad)
    EVALS_DIR.mkdir(exist_ok=True)
    with GOLDEN_PATH.open("w") as f:
        for g in golden:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    CORPUS_PATH.write_text(
        json.dumps(
            {"source": "CUAD v1", "url": CUAD_URL, "zip_md5": CUAD_ZIP_MD5, "documents": corpus},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    n_ans = sum(g["answerable"] for g in golden)
    print(
        f"{len(golden)} questions ({n_ans} answerable, {len(golden) - n_ans} unanswerable) "
        f"over {len(corpus)} contracts -> {GOLDEN_PATH.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
