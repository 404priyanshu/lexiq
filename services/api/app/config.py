from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
API_ROOT = REPO_ROOT / "services" / "api"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"),
        extra="ignore",
    )

    database_url: str = "postgresql+asyncpg://lexiq:lexiq@localhost:5433/lexiq"
    redis_url: str = "redis://localhost:6380/0"

    models_dir: Path = API_ROOT / "models"
    # Uploaded source files, stored by content hash.
    uploads_dir: Path = REPO_ROOT / "data" / "uploads"
    max_upload_mb: int = 100

    # Chunking, in embedder tokens. bge-small truncates at 512, so max stays below that.
    chunk_target_tokens: int = 400
    chunk_max_tokens: int = 480
    chunk_overlap_tokens: int = 64
    embed_batch_size: int = 32

    # Retrieval. Candidates from each first-stage list are fused with RRF; the top
    # `rerank_candidates` go through the cross-encoder; `top_k` are returned.
    lexical_k: int = 50
    dense_k: int = 50
    hnsw_ef_search: int = 100
    rrf_k: int = 60
    # Golden-set sweep (evals/README.md): 10 candidates keeps recall@5 within 0.003 of 30
    # and halves rerank time; the reranker rarely lifts a deeper candidate into the top 8.
    rerank_candidates: int = 10
    # Cross-encoder input length (query + passage, in tokens). Keep the full 512: cutting to
    # 256 dropped recall@5 by ~0.07 because many clauses sit past a chunk's first 256 tokens.
    rerank_max_tokens: int = 512
    # int8-quantized cross-encoder: same recall on the golden set, faster on CPU.
    reranker_int8: bool = True
    top_k: int = 8
    # Abstain before generation when the best reranked passage's sigmoid(logit) is below this.
    # Only a floor for clearly off-topic questions: on the CUAD golden set the reranker's score
    # barely separates answerable from unanswerable questions (at 0.1 it refused 33% of
    # answerable ones), so the model's INSUFFICIENT_EVIDENCE path does the real abstaining.
    abstain_min_relevance: float = 0.0001

    # Generation. Read from .env here (not os.environ), so it's passed to the client explicitly.
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-opus-5-5"
    # Grounded QA over a handful of passages is latency-sensitive and needs little reasoning.
    llm_effort: str = "low"
    # Includes adaptive thinking tokens, so leave headroom above the visible answer.
    llm_max_tokens: int = 16000
    llm_fallbacks: bool = True
    judge_model: str = "claude-opus-5-5"

    # Redis caches (app.cache). Keys carry the corpus version, so TTLs only bound memory.
    cache_enabled: bool = True
    cache_retrieval_ttl_s: int = 3600
    cache_answer_ttl_s: int = 86400

    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def settings_with(overrides: list[str]) -> Settings:
    """Settings with KEY=VALUE overrides applied (for eval and benchmark sweeps)."""
    updates = {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or key not in Settings.model_fields:
            raise ValueError(f"bad override {item!r}; expected KEY=VALUE with a Settings field")
        updates[key] = value
    # Init kwargs take precedence over env and .env, and are validated/coerced like them.
    return Settings(**{**get_settings().model_dump(), **updates})
