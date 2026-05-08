import joblib
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status

from . import schemas 


FILE_LOCATION = Path(__file__).resolve().parents[1] / "model.joblib"
LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}


ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Сучасний спосіб ініціалізації ресурсів у FastAPI"""
    if FILE_LOCATION.exists():
        ml_models["iris_classifier"] = joblib.load(FILE_LOCATION)
        print(f"--- Artifact loaded from {FILE_LOCATION} ---")
    else:
        print(f"--- Warning: {FILE_LOCATION} not found ---")
    yield
    ml_models.clear()

app = FastAPI(
    title="Botanical Prediction Service",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/")
async def get_system_info():
    return {
        "app_name": "Iris Classifier",
        "ready": "iris_classifier" in ml_models
    }

@app.get("/check-up")
async def health_status():
    is_ready = "iris_classifier" in ml_models
    return {
        "status": "online" if is_ready else "degraded",
        "storage_path": str(FILE_LOCATION)
    }

@app.post("/predict", response_model=schemas.PredictionResponse)
async def perform_inference(payload: schemas.IrisFeatures):
    classifier = ml_models.get("iris_classifier")
    
    if not classifier:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Machine Learning model is unavailable"
        )
    
    input_data = np.array([
        [payload.sepal_length, payload.sepal_width, 
         payload.petal_length, payload.petal_width]
    ])
    
    try:
        idx = int(classifier.predict(input_data)[0])
        confidences = classifier.predict_proba(input_data)[0]
        score = float(confidences[idx])
        
        return schemas.PredictionResponse(
            class_id=idx,
            class_name=LABELS[idx],
            probability=round(score, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")