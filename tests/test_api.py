import pytest
from fastapi.testclient import TestClient
from app.main import app, MODEL_PATH
from ml.train import IrisTrainer

if not MODEL_PATH.exists():
    trainer = IrisTrainer(model_path=MODEL_PATH)
    trainer.execute()

@pytest.fixture
def api_client():
    with TestClient(app) as client:
        yield client

def test_system_info_route(api_client):
    """Тест ендпоінту / (root)"""
    response = api_client.get("/")
    assert response.status_code == 200
    assert response.json()["model_status"] == "active"

def test_monitoring_endpoint(api_client):
    """Тест ендпоінту /check-up"""
    response = api_client.get("/check-up")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True
    assert data["drift_ready"] is True

def test_inference_logic_setosa(api_client):
    """Тест передбачення для сорту setosa"""
    sample_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = api_client.post("/predict", json=sample_data)
    assert response.status_code == 200
    
    result = response.json()
    assert result["class_name"] == "setosa"
    assert "probability" in result
    assert result["class_id"] == 0

def test_validation_error_handling(api_client):
    """Тест на некоректні типи даних"""
    bad_payload = {"sepal_length": "not-a-number"}
    response = api_client.post("/predict", json=bad_payload)
    assert response.status_code == 422