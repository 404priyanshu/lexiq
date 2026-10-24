async def test_health_reports_postgres_and_redis_ok(client):
    resp = await client.get("/health")

    assert resp.status_code == 200, resp.json()
    assert resp.json() == {"status": "ok", "checks": {"postgres": "ok", "redis": "ok"}}
