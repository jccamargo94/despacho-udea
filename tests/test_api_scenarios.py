def test_create_scenario_returns_id(api_client):
    body = {
        "mode": "generator",
        "penetration_level": "low",
        "units": [
            {
                "name": "B1",
                "mwh_nom": 10,
                "hours_to_deplete": 2,
                "initial_soc": 5,
                "min_soc": 0,
                "max_soc": 10,
                "efficiency": 0.9,
                "discharge_bid": 100.0,
            }
        ],
    }
    resp = api_client.post("/scenarios", json=body)
    assert resp.status_code == 200
    assert "id" in resp.json()


def test_create_scenario_rejects_arbitrage_without_charge_bid(api_client):
    body = {
        "mode": "arbitrage",
        "penetration_level": "low",
        "units": [
            {
                "name": "B1",
                "mwh_nom": 10,
                "hours_to_deplete": 2,
                "initial_soc": 5,
                "min_soc": 0,
                "max_soc": 10,
                "efficiency": 0.9,
                "discharge_bid": 100.0,
            }
        ],
    }
    resp = api_client.post("/scenarios", json=body)
    assert resp.status_code == 422
