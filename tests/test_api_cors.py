def test_cors_allows_configured_frontend_origin(api_client):
    resp = api_client.get("/runs", headers={"Origin": "http://localhost:3000"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
