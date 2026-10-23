from fastapi.testclient import TestClient

from app.main import app


def test_health_reports_postgres_and_redis_ok():
    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200, resp.json()
    assert resp.json() == {"status": "ok", "checks": {"postgres": "ok", "redis": "ok"}}
