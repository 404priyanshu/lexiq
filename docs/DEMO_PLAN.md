# Public demo plan

Goal: a URL a recruiter can open, ask a question about the 52,650-page corpus, and get a
streamed, cited answer in about two seconds, without being able to run up the LLM bill or
change the data.

## What it runs on

**Recommended: one small VM running everything with Docker Compose, behind Caddy for HTTPS.**

```
browser ──https──▶ Caddy ─┬─ /        ▶ Next.js (web)
                          └─ /api/*   ▶ FastAPI (search, answer)
                                          ├─ ParadeDB Postgres (BM25 + HNSW), not exposed
                                          └─ Redis (caches, rate limits), not exposed
```

- **Why a VM and not managed services:** retrieval needs ParadeDB's `pg_search` (BM25) next
  to pgvector, and I couldn't confirm that Neon or Supabase offer it, so Postgres would be
  self-hosted anyway. The reranker and embedder run in-process with the API, which needs a
  steady 1–2 GB of RAM and real CPU; per-request serverless platforms are a poor fit.
- **One origin:** serving the web app and the API from the same domain removes CORS and the
  need for a second certificate. The browser already calls the API directly, so the web app
  needs only its API URL changed to `/api`.
- **Size:** Hetzner CX33 (4 vCPU, 8 GB RAM, 80 GB disk), €8.49/month plus the IPv4 fee, from
  Hetzner's price table effective 15 June 2026. The index is 0.7 GB of Postgres data (the
  237 MB HNSW index should stay in RAM), the ONNX models 237 MB. The 2 vCPU / 4 GB CX23
  (€5.49) would fit in memory, but reranking on 2 shared cores will likely push retrieval
  past 0.5 s. Hetzner changed prices twice in 2026, so check at order time.
- **Alternative:** web on Vercel's free Hobby plan and only the API on the VM. Same cost, but
  two origins (CORS, a separate API domain) for no real gain.

## Data

Load the existing local index rather than re-ingesting on the server: a data-only `pg_dump`
of `documents` and `chunks`, the same approach the CI eval cache uses, restored after
migrations. Uploaded source files (598 MB) aren't needed for search and stay local. The
worker isn't deployed.

## Code changes for a public deployment (done)

All off by default, so local development is unchanged; the demo's `.env` turns them on.

1. **Read-only mode** (`READ_ONLY`): `POST /documents` and `/reingest` return 403, `/answer`
   is capped at the default number of passages, and the web app replaces the upload box with
   a description of the corpus.
2. **Rate limits** in Redis, per client address (`ANSWER_LIMIT_PER_MINUTE`, `_PER_DAY`,
   `SEARCH_LIMIT_PER_MINUTE`): HTTP 429 with `Retry-After` and a readable message. Behind
   Caddy, uvicorn's `--proxy-headers` makes the visitor's address the key, and Caddy sets
   `X-Forwarded-For` itself, so it can't be spoofed.
3. **A global daily answer budget** (`DAILY_ANSWER_BUDGET`): counts model calls per UTC day,
   so cached answers are free. Once spent, `/answer` still streams the retrieved passages and
   then explains that answers resume at midnight UTC.
4. **Example questions** (`app/demo.py`): six questions across both corpora whose evidence
   retrieval ranks first on the 52k-page index. `GET /config` serves them to the web app;
   `python -m app.demo warm` caches their answers, so clicking one is instant and free.
5. **Deployment files** in `infra/demo`: Dockerfiles for the API (models baked in, 2.1 GB) and
   web app (Next.js standalone, 293 MB), `compose.yml`, `Caddyfile`, `.env.example`,
   `deploy.sh`, and `restore-index.sh`; `make demo-dump` writes the 298 MB index dump.
6. **Attribution footer** for CUAD and the CFR, plus a note that answers can be wrong.

Rehearsed locally with the production compose file (`DOMAIN=localhost`): the dump restored and
the examples warmed in 2 min 23 s; through Caddy over HTTPS a cached example returned in 35 ms
and a new question streamed its first token at 0.97 s; uploads got 403; the 11th answer in a
minute got 429. That run also caught the mobile layout overflowing on long contract titles,
now fixed.

## Deploy runbook

On your machine:

```bash
make demo-dump        # data/demo-index.dump (298 MB)
scp data/demo-index.dump root@SERVER:/root/
```

On the server (Ubuntu 24.04, as root, after pointing the domain's DNS at it):

```bash
apt-get update && apt-get install -y docker.io docker-compose-v2 git ufw
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw --force enable
git clone https://github.com/404priyanshu/lexiq.git && cd lexiq/infra/demo
cp .env.example .env && nano .env     # DOMAIN, POSTGRES_PASSWORD, OPENAI_API_KEY
./deploy.sh
./restore-index.sh /root/demo-index.dump
```

The repository is private, so the server needs read access to clone it (a deploy key, or
make the repository public). Updates later: `./deploy.sh` again; after a corpus change, a new
dump and `./restore-index.sh`.

## Hardening

- Postgres and Redis publish no ports; strong generated passwords; firewall allows 22, 80
  and 443 only; SSH by key, password login off.
- A separate OpenAI project and key for the demo, with a monthly budget limit set in the
  OpenAI dashboard. With prepaid credit the spend also can't exceed the balance.
- Containers restart on failure. A free uptime check pings `/api/health`.
- `CORS_ORIGINS` is the demo domain only.

## Costs

| Item | Cost |
|---|---|
| Hetzner CX33 | €8.49/month + IPv4 fee |
| Domain (needed for HTTPS) | ~$10/year, or a subdomain of one you own |
| OpenAI | ~$0.0016 per new answer; cached answers are free. $5 of credit ≈ 3,000 answers |
| **Total** | **≈ €10/month plus answers actually asked** |

`gpt-4.1-nano` would cut answer cost about 4×, but its answer quality hasn't been measured; a
graded run on the golden set (~$0.10) would settle that before switching.

## Rollout

1. Code changes 1–6 above: done.
2. You create the Hetzner account and server, add your SSH key, and point a domain's DNS at
   it. I can't create accounts or enter payment details.
3. Provision over SSH: Docker, firewall, secrets in a server-side `.env`, deploy script.
4. Restore the index dump, warm the example questions, run `make bench` against the server
   to confirm latency.
5. Add the URL to the README and your resume.

## Decisions for you

- **Host:** Hetzner CX33 (recommended), or something you already pay for.
- **Domain:** buy one, or use a subdomain of a domain you own.
- **Budget:** the daily answer cap and the OpenAI top-up (for example, 300 a day and $5).
- **Model:** keep `gpt-4.1-mini`, or measure `gpt-4.1-nano` first.

## Risks

- **Scrambled CFR excerpts:** CFR volumes are printed in two columns and the PDF parser reads
  straight across them, so regulation passages interleave mid-sentence ("Compliance with tain
  these records..."). Retrieval and answers mostly cope, but the source excerpts a visitor
  expands look broken. Column-aware parsing (PyMuPDF block sorting) and a re-ingest of the CFR
  volumes would fix it; worth doing before sharing the URL widely.
- **Slower hardware:** retrieval is 0.2 s on an M5 Pro; on 4 shared vCPUs expect 2–3× that.
  The step-4 benchmark measures it. If it's too slow, a larger CAX/CX size is the safe fix;
  fewer rerank candidates would need its own eval run (10 vs 30 cost 0.003 recall@5, but
  fewer than 10 is unmeasured).
- **Abuse:** bots asking many unique questions hit the per-IP and daily caps; the worst case is
  the daily budget, about $0.50.
- **Prices:** the Hetzner figures come from Hetzner's June 2026 price table; recheck before
  ordering.
