from typing import Dict, List, Annotated
from pydantic import BaseModel, Field, ConfigDict

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0.1, le=10, description="Sepal length in cm", examples=[5.1])
    sepal_width: float = Field(..., ge=0.1, le=10, description="Sepal width in cm", examples=[3.5])
    petal_length: float = Field(..., ge=0.1, le=10, description="Petal length in cm", examples=[1.4])
    petal_width: float = Field(..., ge=0.1, le=10, description="Petal width in cm", examples=[0.2])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }
    )

class PredictionResponse(BaseModel):
    class_id: int = Field(..., description="Numeric ID of the predicted class")
    class_name: str = Field(..., description="Human-readable label")
    probability: float = Field(..., description="Model confidence score (0.0 to 1.0)")

class FeatureDriftInfo(BaseModel):
    statistic: float
    p_value: float
    drift_detected: bool

class DriftRequest(BaseModel):
    samples: Annotated[List[Annotated[List[float], Field(min_length=4, max_length=4)]], Field(min_length=5)]
    alpha: float = Field(default=0.05, ge=0.001, le=0.5, description="Significance threshold for KS-test")

class DriftResponse(BaseModel):
    drift_detected: bool
    n_drifted_features: int
    drifted_features: List[str]
    per_feature: Dict[str, FeatureDriftInfo]
    n_samples: int
    alpha: float
    
    model_config = ConfigDict(title="Data Drift Analysis Result")