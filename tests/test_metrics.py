import pytest
from fastapi.testclient import TestClient
from app.main import app
from ml.train import IrisTrainer

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="module", autouse=True)
def prepare_model():
    """Гарантуємо, що модель та артефакти існують перед початком тестів"""
    trainer = IrisTrainer()
    trainer.execute()

def test_latency_metric_exists(client):
    """Перевірка, чи реєструється час обробки запиту"""
    response = client.get("/metrics")
    
    assert "model_inference_latency_seconds_count" in response.text

def test_inference_updates_prometheus_counter(client):
    """Перевірка, чи інкрементується лічильник після запиту на передбачення"""
    initial_response = client.get("/metrics").text
    
    sample_payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    client.post("/predict", json=sample_payload)
    
    updated_response = client.get("/metrics").text
    
    assert 'result_status="success"' in updated_response
    assert 'species="setosa"' in updated_response
    assert updated_response != initial_response
