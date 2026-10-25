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
