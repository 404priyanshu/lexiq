"""Whole-corpus slice of the golden set: the answerable questions, rewritten to name their
contract so they can be asked without a document filter.

The golden questions say "this agreement" because the evaluator restricts retrieval to one
contract. Here each question names its contract from CUAD's own annotations (the document
name and up to two parties that are real legal entities), e.g. "In the Supply Agreement
between Reynolds Consumer Products LLC and Pactiv LLC, ...". A retrieved chunk only counts
as gold when it comes from that contract: governing-law and assignment clauses are nearly
identical across contracts, so text alone can't tell them apart.

Questions whose contract has no identifiable party are skipped (and counted) rather than
asked ambiguously.

Usage: uv run python -m app.evals.golden_corpus --cuad ../../data/cuad/extracted/CUAD_v1
"""

import argparse
import json
import re
from pathlib import Path

from app.config import REPO_ROOT
from app.evals.golden import EVALS_DIR, GOLDEN_PATH

CORPUS_GOLDEN_PATH = EVALS_DIR / "golden_corpus.jsonl"

_ENTITY_SUFFIX = re.compile(
    r"\b(inc|incorporated|corp|corporation|llc|l\.l\.c|ltd|limited|company|co|l\.?p|plc|gmbh"
    r"|s\.?a|n\.?v|b\.?v|a\.?g|ag|holdings|trust|bank|partners)\b\.?,?$",
    re.I,
)


def legal_entities(parties: list[str]) -> list[str]:
    """Party annotations that name a company, in order, deduplicated.

    CUAD's "Parties" spans mix company names with role labels ("Seller", "Party A"),
    defined-term sentences and signature-page placeholders; only names ending in a corporate
    suffix are kept.
    """
    out, seen = [], set()
    for raw in parties:
        name = " ".join(raw.split()).strip(" ,;:")
        if not name or len(name) > 80 or '"' in name or "_" in name:
            continue
        suffix = _ENTITY_SUFFIX.search(name)
        # The suffix must follow an actual name: "Company" or "The Company" alone is a defined
        # term for one of the parties, not a name.
        if not suffix or name[: suffix.start()].strip(" ,").lower() in ("", "the"):
            continue
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


def contract_reference(document_name: str, parties: list[str]) -> str | None:
    entities = legal_entities(parties)
    if not entities:
        return None
    name = " ".join(document_name.split())
    # CUAD's name spans sometimes end in a cut-off fragment: "Agreement (", "AGREEMENT, d".
    name = re.sub(r"\s*\([^)]*$", "", name)
    name = re.sub(r",\s*\w{0,2}$", "", name).strip(" .,")
    if name.isupper():
        name = name.title()
    if not name.lower().startswith("the "):
        name = f"the {name}"
    if len(entities) == 1:
        return f"{name} with {entities[0]}"
    return f"{name} between {entities[0]} and {entities[1]}"


def name_question(question: str, reference: str) -> str:
    body = re.sub(r"\bthis agreement\b", "the agreement", question, flags=re.I)
    return f"In {reference}, {body[0].lower()}{body[1:]}"


def build(cuad: Path) -> tuple[list[dict], int]:
    golden = [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines()]
    annotations = {}
    for doc in json.loads((cuad / "CUAD_v1.json").read_text())["data"]:
        qas = {q["id"].split("__")[-1]: q for q in doc["paragraphs"][0]["qas"]}
        names = [a["text"] for a in qas["Document Name"]["answers"]]
        parties = [a["text"] for a in qas["Parties"]["answers"]]
        annotations[doc["title"]] = (names[0] if names else "", parties)

    out, skipped = [], 0
    for g in golden:
        if not g["answerable"]:
            continue
        document_name, parties = annotations[g["contract"]]
        reference = contract_reference(document_name or "agreement", parties)
        if reference is None:
            skipped += 1
            continue
        out.append(
            {
                "id": "c" + g["id"][1:],
                "source_id": g["id"],
                "question": name_question(g["question"], reference),
                "category": g["category"],
                "answerable": True,
                "contract": g["contract"],
                "evidence": g["evidence"],
            }
        )
    return out, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cuad", type=Path, required=True, help="extracted CUAD_v1 directory")
    args = parser.parse_args()

    questions, skipped = build(args.cuad)
    with CORPUS_GOLDEN_PATH.open("w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    contracts = len({q["contract"] for q in questions})
    print(
        f"{len(questions)} contract-named questions over {contracts} contracts "
        f"({skipped} skipped: no identifiable party) -> "
        f"{CORPUS_GOLDEN_PATH.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
