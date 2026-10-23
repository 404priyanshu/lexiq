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
    rerank_candidates: int = 30
    top_k: int = 8
    # Abstain when the best reranked passage's sigmoid(logit) is below this. Provisional;
    # calibrated against the golden set's unanswerable questions in Phase 5.
    abstain_min_relevance: float = 0.1

    # Generation. Read from .env here (not os.environ), so it's passed to the client explicitly.
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-opus-5-5"
    # Grounded QA over a handful of passages is latency-sensitive and needs little reasoning.
    llm_effort: str = "low"
    # Includes adaptive thinking tokens, so leave headroom above the visible answer.
    llm_max_tokens: int = 16000
    llm_fallbacks: bool = True

    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
