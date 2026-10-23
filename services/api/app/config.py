from functools import lru_cache
from pathlib import Path

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

    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
