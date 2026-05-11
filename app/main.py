import logging
import time
import joblib
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from . import schemas
from . import metrics
from .drift import DriftAnalyzer
from .logging_config import setup_logging


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model.joblib"
REFERENCE_PATH = BASE_DIR / "reference_stats.joblib"

LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

setup_logging()
logger = logging.getLogger("iris-service")


state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Сучасне управління станом застосунку (Lifespan)"""
    logger.info("Initializing application resources...")
    
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
        metrics.IS_MODEL_ACTIVE.set(1)
        logger.info(f"Model loaded from {MODEL_PATH}")
    else:
        metrics.IS_MODEL_ACTIVE.set(0)
        logger.error(f"Critical error: Model artifact not found at {MODEL_PATH}")

    if REFERENCE_PATH.exists():
        ref_data = joblib.load(REFERENCE_PATH)
        state["drift_engine"] = DriftAnalyzer(
            baseline_data=ref_data["X"],
            feature_names=ref_data["feature_names"]
        )
        logger.info("Drift analyzer is ready to use")
    else:
        logger.warning("Reference stats missing. Drift detection endpoint will be limited.")
    
    yield
    state.clear()
    logger.info("Application resources cleared.")

app = FastAPI(
    title="Botanical Monitoring API",
    description="ML service with Prometheus tracking and Statistical Drift detection",
    version="3.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Проміжний шар для вимірювання затримки відповідей"""
    start_point = time.perf_counter()
    response = await call_next(request)
    
    if request.url.path in ["/predict", "/check-drift"]:
        duration = time.perf_counter() - start_point
        metrics.INFERENCE_SPEED.observe(duration)
        
    return response

@app.get("/")
async def service_info():
    return {
        "service": "Iris ML API",
        "model_status": "active" if "model" in state else "inactive",
        "monitoring": "enabled"
    }

@app.get("/check-up")
async def health_status():
    return {
        "status": "online" if "model" in state else "degraded",
        "model_loaded": "model" in state,
        "drift_ready": "drift_engine" in state
    }

@app.get("/metrics")
async def get_metrics():
    """Ендпоінт для збору даних системою Prometheus"""
    return Response(
        content=generate_latest(metrics.REGISTRY), 
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/predict", response_model=schemas.PredictionResponse)
async def perform_inference(payload: schemas.IrisFeatures):
    model = state.get("model")
    if not model:
        metrics.SYSTEM_ERRORS.labels(exception_type="MissingModel").inc()
        raise HTTPException(status_code=503, detail="Model is unavailable")
    
    try:
        # Підготовка даних
        input_vector = np.array([[
            payload.sepal_length, payload.sepal_width, 
            payload.petal_length, payload.petal_width
        ]])
        
        prediction_id = int(model.predict(input_vector)[0])
        confidences = model.predict_proba(input_vector)[0]
        score = float(confidences[prediction_id])
        species_name = LABELS[prediction_id]
        
        metrics.INFERENCE_COUNT.labels(species=species_name, result_status="success").inc()
        metrics.MODEL_CONFIDENCE_LEVEL.observe(score)
        
        logger.info("Inference successful", extra={
            "species": species_name, 
            "score": round(score, 3)
        })
        
        return schemas.PredictionResponse(
            class_id=prediction_id,
            class_name=species_name,
            probability=round(score, 4)
        )
        
    except Exception as err:
        metrics.SYSTEM_ERRORS.labels(exception_type=type(err).__name__).inc()
        logger.error(f"Prediction failure: {str(err)}")
        raise HTTPException(status_code=500, detail="Internal inference error")

@app.post("/check-drift", response_model=schemas.DriftResponse)
async def run_drift_analysis(request_data: schemas.DriftRequest):
    analyzer = state.get("drift_engine")
    if not analyzer:
        metrics.SYSTEM_ERRORS.labels(exception_type="DriftEngineOffline").inc()
        raise HTTPException(status_code=503, detail="Drift engine not initialized")
    
    metrics.DATA_DRIFT_RUNS.inc()
    
    raw_data = np.array(request_data.samples)
    report = analyzer.analyze(current_data=raw_data, threshold=request_data.alpha)
    
    for f_name, info in report["per_feature"].items():
        if info["drift_detected"]:
            metrics.DATA_DRIFT_ALERTS.labels(feature_name=f_name).inc()
            
    logger.info("Drift analysis completed", extra={"drift_found": report["drift_detected"]})
    
    return schemas.DriftResponse(**report)