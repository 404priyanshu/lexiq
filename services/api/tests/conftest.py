"""Tests run against a separate database and Redis DB so they never touch dev data.

Env vars are set before anything imports app.config / app.db (engine is built at import).
"""

import asyncio
import os
import tempfile

os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://lexiq:lexiq@localhost:5433/lexiq_test"
)
os.environ["REDIS_URL"] = os.environ.get("TEST_REDIS_URL", "redis://localhost:6380/15")
os.environ["UPLOADS_DIR"] = tempfile.mkdtemp(prefix="lexiq-test-uploads-")

import asyncpg  # noqa: E402
import httpx  # noqa: E402
import pytest  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

from app.config import API_ROOT, get_settings  # noqa: E402
from app.db import engine  # noqa: E402
from app.ingest.pipeline import Models  # noqa: E402
from app.main import app, lifespan  # noqa: E402
from app.ml.embedder import Embedder, load_tokenizer, model_path_exists  # noqa: E402


async def _ensure_database(url: str) -> None:
    u = make_url(url)
    conn = await asyncpg.connect(
        user=u.username, password=u.password, host=u.host, port=u.port, database="postgres"
    )
    try:
        if not await conn.fetchval("select 1 from pg_database where datname = $1", u.database):
            await conn.execute(f'create database "{u.database}"')
    finally:
        await conn.close()


@pytest.fixture(scope="session", autouse=True)
def migrated_db():
    asyncio.run(_ensure_database(get_settings().database_url))
    command.upgrade(Config(str(API_ROOT / "alembic.ini")), "head")


@pytest.fixture
async def clean_db():
    async with engine.begin() as conn:
        await conn.execute(text("truncate documents cascade"))
    yield


@pytest.fixture(scope="session")
async def client():
    async with lifespan(app):
        await app.state.redis.flushdb()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.fixture(scope="session")
def models() -> Models:
    if not model_path_exists():
        pytest.skip("embedder not downloaded (uv run python -m app.ml.download)")
    return Models(embedder=Embedder(), tokenizer=load_tokenizer())
