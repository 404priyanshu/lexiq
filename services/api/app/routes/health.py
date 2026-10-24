from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text

from app.db import engine

router = APIRouter()

REQUIRED_EXTENSIONS = {"vector", "pg_search"}


@router.get("/health")
async def health(request: Request, response: Response) -> dict:
    checks: dict[str, str] = {}

    try:
        async with engine.connect() as conn:
            rows = await conn.execute(text("select extname from pg_extension"))
            missing = REQUIRED_EXTENSIONS - {r[0] for r in rows}
            checks["postgres"] = f"missing extensions: {sorted(missing)}" if missing else "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {exc.__class__.__name__}"

    try:
        await request.app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc.__class__.__name__}"

    healthy = all(v == "ok" for v in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if healthy else "degraded", "checks": checks}
