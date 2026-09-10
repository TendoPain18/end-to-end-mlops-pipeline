from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "VPN Traffic Classification API is running"
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_predict():
    payload = {
        "duration": 0.0,
        "total_fiat": 0.0,
        "total_biat": 0.0,
        "min_fiat": -1.0,
        "min_biat": -1.0,
        "max_fiat": -1.0,
        "max_biat": -1.0,
        "mean_fiat": 0.0,
        "mean_biat": 0.0,
        "flowPktsPerSecond": 0.0,
        "flowBytesPerSecond": 0.0,
        "min_flowiat": -1.0,
        "max_flowiat": -1.0,
        "mean_flowiat": 0.0,
        "std_flowiat": 0.0,
        "min_active": -1.0,
        "mean_active": 0.0,
        "max_active": -1.0,
        "std_active": 0.0,
        "min_idle": -1.0,
        "mean_idle": 0.0,
        "max_idle": -1.0,
        "std_idle": 0.0,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert isinstance(result["prediction"], str)
    assert result["prediction"] != ""


def test_predict_validation():
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422
