from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root():
    assert client.get("/").status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_prediction():
    response = client.post("/predict", json={"distance_km": 10, "raining": 1, "heavy_traffic": 1})
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in (0, 1)
    assert result["status"] == ("Late Delivery" if result["prediction"] == 1 else "On Time")


def test_negative_distance():
    assert client.post("/predict", json={"distance_km": -10, "raining": 1, "heavy_traffic": 1}).status_code == 422


def test_missing_field():
    assert client.post("/predict", json={"distance_km": 10, "raining": 1}).status_code == 422


def test_invalid_binary():
    assert client.post("/predict", json={"distance_km": 10, "raining": 3, "heavy_traffic": 1}).status_code == 422
