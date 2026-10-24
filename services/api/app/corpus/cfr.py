"""Fetch Code of Federal Regulations volumes (public domain) from govinfo.gov.

Each title's volumes are numbered from 1; we fetch until govinfo has no next volume. PDFs land
in data/cfr/ (gitignored), and corpora/cfr-<year>.json records each file's URL, SHA-256 and
page count so the corpus is reproducible and checkable.

Usage: uv run python -m app.corpus.cfr --year 2025 --titles 12 17 26
"""

import argparse
import contextlib
import hashlib
import json
import time
from pathlib import Path

import httpx
import pymupdf

from app.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "cfr"
MANIFEST_DIR = REPO_ROOT / "corpora"
BASE = "https://www.govinfo.gov/content/pkg/{pkg}/pdf/{pkg}.pdf"
USER_AGENT = "LexIQ corpus fetch (+https://github.com/404priyanshu/lexiq)"
MAX_VOLUMES = 60


def fetch(
    year: int,
    titles: list[int],
    client: httpx.Client | None = None,
    data_dir: Path = DATA_DIR,
    pause_s: float = 0.5,
) -> list[dict]:
    data_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    # Close only a client we created; a caller's client stays usable after fetch returns.
    owned = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120)
    with client if owned else contextlib.nullcontext(client):
        for title in titles:
            for volume in range(1, MAX_VOLUMES + 1):
                pkg = f"CFR-{year}-title{title}-vol{volume}"
                url = BASE.format(pkg=pkg)
                path = data_dir / f"{pkg}.pdf"
                if not path.exists():
                    # govinfo answers a missing volume with a redirect to an HTML page.
                    resp = client.get(url, follow_redirects=False)
                    if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
                        break
                    # Write then rename, so an interrupted download never leaves a truncated
                    # PDF that later runs would treat as cached.
                    tmp = path.with_suffix(".part")
                    tmp.write_bytes(resp.content)
                    tmp.replace(path)
                    time.sleep(pause_s)  # be gentle with govinfo
                data = path.read_bytes()
                with pymupdf.open(stream=data, filetype="pdf") as doc:
                    pages = doc.page_count
                entries.append(
                    {
                        "title": title,
                        "volume": volume,
                        "name": f"{year} CFR Title {title}, Volume {volume}",
                        "url": url,
                        "file": path.name,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "pages": pages,
                    }
                )
                print(f"title {title:>2} vol {volume:>2}: {pages:>5} pages", flush=True)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--titles", type=int, nargs="+", required=True)
    args = parser.parse_args()

    entries = fetch(args.year, args.titles)
    MANIFEST_DIR.mkdir(exist_ok=True)
    out = MANIFEST_DIR / f"cfr-{args.year}.json"
    out.write_text(
        json.dumps(
            {
                "source": "Code of Federal Regulations (annual edition), govinfo.gov",
                "license": "US government work, public domain",
                "year": args.year,
                "documents": entries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(entries)} volumes, {sum(e['pages'] for e in entries)} pages -> {out}")


if __name__ == "__main__":
    main()
