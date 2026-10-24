import json

import httpx

from app.corpus.cfr import fetch
from app.ingest.bulk import cfr_files
from tests.test_parsers import make_pdf


def test_cfr_fetch_downloads_volumes_until_govinfo_has_no_more(tmp_path):
    pdf = make_pdf(["Part 1. General provisions.", "Part 2. Definitions."])
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if "vol3" in request.url.path:
            # govinfo answers a missing volume with a redirect to an HTML page.
            return httpx.Response(302, headers={"location": "https://www.govinfo.gov/"})
        return httpx.Response(200, content=pdf)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    entries = fetch(2025, [17], client=client, data_dir=tmp_path, pause_s=0)

    assert [e["volume"] for e in entries] == [1, 2]
    assert all(e["pages"] == 2 and len(e["sha256"]) == 64 for e in entries)
    assert entries[0]["url"].endswith("CFR-2025-title17-vol1/pdf/CFR-2025-title17-vol1.pdf")
    assert len(requested) == 3  # stopped at the first missing volume
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "CFR-2025-title17-vol1.pdf",
        "CFR-2025-title17-vol2.pdf",
    ]


def test_cfr_files_map_manifest_entries_to_titled_files(tmp_path):
    manifest = tmp_path / "cfr.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "name": "2025 CFR Title 17, Volume 1",
                        "file": "CFR-2025-title17-vol1.pdf",
                        "url": "https://example.test/v1.pdf",
                    },
                ]
            }
        )  # fmt: skip
    )
    [(path, title, url)] = cfr_files(manifest)

    assert path.name == "CFR-2025-title17-vol1.pdf"
    assert title == "2025 CFR Title 17, Volume 1"
    assert url == "https://example.test/v1.pdf"


def test_cfr_fetch_leaves_a_caller_supplied_client_open(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    fetch(2025, [17], client=client, data_dir=tmp_path, pause_s=0)

    assert not client.is_closed
    client.close()


async def test_bulk_wait_times_out_naming_unfinished_documents(clean_db):
    import pytest

    from app.db import SessionLocal
    from app.ingest.bulk import wait
    from app.models import Document

    async with SessionLocal() as session:
        doc = Document(title="Stuck MSA", mime_type="application/pdf", sha256="stuck-sha",
                       status="processing")  # fmt: skip
        session.add(doc)
        await session.commit()

    with pytest.raises(TimeoutError, match=r"1 documents unfinished.*Stuck MSA \(processing\)"):
        await wait([doc.id], interval=0, max_wait_s=0)


async def test_bulk_enqueue_dedupes_identical_files_so_wait_can_finish(clean_db, client, tmp_path):
    from app.db import SessionLocal
    from app.ingest.bulk import enqueue, wait
    from app.models import Document

    data = make_pdf(["Identical contract text."])
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    a.write_bytes(data)
    b.write_bytes(data)

    ids = await enqueue([(a, "A", None), (b, "B", None)])
    assert len(ids) == 1

    async with SessionLocal() as session:
        (await session.get(Document, ids[0])).status = "ready"
        await session.commit()
    await wait(ids, interval=0, max_wait_s=0)  # returns instead of timing out


def test_legal_entities_keep_company_names_and_drop_roles_and_defined_terms():
    from app.evals.golden_corpus import legal_entities

    parties = [
        "Valeant", "Dova Pharmaceuticals, Inc.", "Company", "The Company", "Party A",
        'Dova and Valeant are each referred to individually as a "Party"',
        "_______________________ and its affiliates", "Valeant Pharmaceuticals North America LLC",
        "DOVA PHARMACEUTICALS, INC.",
    ]  # fmt: skip
    assert legal_entities(parties) == [
        "Dova Pharmaceuticals, Inc.",
        "Valeant Pharmaceuticals North America LLC",
    ]


def test_contract_reference_and_question_naming():
    from app.evals.golden_corpus import contract_reference, name_question

    ref = contract_reference("MASTER SUPPLY AGREEMENT", ["Seller", "PACTIV LLC", "Acme Corp."])
    assert ref == "the Master Supply Agreement between PACTIV LLC and Acme Corp."
    assert contract_reference("STRATEGIC ALLIANCE AGREEMENT, d", ["Edible Arrangements, LLC"]) == (
        "the Strategic Alliance Agreement with Edible Arrangements, LLC"
    )
    assert contract_reference("Master Services Agreement (", ["RadialSpark, LLC"]).startswith(
        "the Master Services Agreement with"
    )
    assert contract_reference("Agreement", ["Seller", "Buyer"]) is None
    assert name_question("Which jurisdiction's law governs this agreement?", ref) == (
        "In the Master Supply Agreement between PACTIV LLC and Acme Corp., "
        "which jurisdiction's law governs the agreement?"
    )
