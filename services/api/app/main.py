import logging
from contextlib import asynccontextmanager

from anthropic import AsyncAnthropic
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.cache import QueryCache
from app.config import get_settings
from app.db import engine
from app.generation.generator import ClaudeGenerator
from app.ml.registry import EMBEDDER, RERANKER
from app.ml.runtime import model_path_exists
from app.retrieval.retriever import Retriever
from app.routes import answer, documents, health, search

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # ArqRedis is a redis.asyncio.Redis that can also enqueue jobs.
    app.state.redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    app.state.cache = QueryCache(app.state.redis, settings) if settings.cache_enabled else None
    if model_path_exists(EMBEDDER) and model_path_exists(RERANKER):
        app.state.retriever = Retriever.from_settings(engine, settings, app.state.cache)
    else:
        app.state.retriever = None
        log.warning("ONNX models missing; /search disabled until `make models` is run")
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
    if key:
        app.state.generator = ClaudeGenerator(AsyncAnthropic(api_key=key), settings)
    else:
        app.state.generator = None
        log.warning("ANTHROPIC_API_KEY not set; /answer disabled")
    yield
    await app.state.redis.aclose()
    await engine.dispose()


app = FastAPI(title="LexIQ API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(answer.router)
