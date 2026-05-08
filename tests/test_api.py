import pytest
from fastapi.testclient import TestClient
from ml.train import IrisTrainer, DEFAULT_SAVE_PATH
from app.main import app, FILE_LOCATION

if not FILE_LOCATION.exists():
    trainer = IrisTrainer(target_path=FILE_LOCATION)
    trainer.execute()

@pytest.fixture
def api_client():
    with TestClient(app) as client:
        yield client

def test_system_info_route(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    assert response.json()["ready"] is True

def test_monitoring_endpoint(api_client):
    response = api_client.get("/check-up")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "storage_path" in data

def test_inference_logic_setosa(api_client):
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
    bad_payload = {"sepal_length": "invalid_data"}
    response = api_client.post("/predict", json=bad_payload)
    assert response.status_code == 422