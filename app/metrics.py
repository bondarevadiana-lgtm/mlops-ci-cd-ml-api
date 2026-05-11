from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

REGISTRY = CollectorRegistry()

INFERENCE_COUNT = Counter(
    "model_inference_total",
    "Total number of predictions made",
    labelnames=["species", "result_status"],
    registry=REGISTRY,
)

INFERENCE_SPEED = Histogram(
    "model_inference_latency_seconds",
    "Time taken to process a single request",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0),
    registry=REGISTRY,
)

MODEL_CONFIDENCE_LEVEL = Histogram(
    "model_confidence_score",
    "Distribution of prediction probability scores",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=REGISTRY,
)

SYSTEM_ERRORS = Counter(
    "model_system_errors_total",
    "Total count of failed requests",
    labelnames=["exception_type"],
    registry=REGISTRY,
)

IS_MODEL_ACTIVE = Gauge(
    "model_is_active",
    "Current status of the ML model (1: Ready, 0: Not Loaded)",
    registry=REGISTRY,
)

DATA_DRIFT_RUNS = Counter(
    "model_drift_analysis_total",
    "Number of drift detection checks performed",
    registry=REGISTRY,
)

DATA_DRIFT_ALERTS = Counter(
    "model_drift_alerts_total",
    "Count of detected feature drifts",
    labelnames=["feature_name"],
    registry=REGISTRY,
)