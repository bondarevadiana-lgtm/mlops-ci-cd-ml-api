import numpy as np
from scipy import stats
from typing import Dict, List, Any

class DriftAnalyzer:
    
    def __init__(self, baseline_data: np.ndarray, feature_names: List[str]):
        if baseline_data.ndim != 2:
            raise ValueError("Baseline data must be a 2D array (samples, features)")
        
        if len(feature_names) != baseline_data.shape[1]:
            raise ValueError("Feature names length must match columns in baseline data")

        self.baseline = baseline_data
        self.feature_names = feature_names

    def analyze(self, current_data: np.ndarray, threshold: float = 0.05) -> Dict[str, Any]:
        """
        Виконує тест Колмогорова-Смирнова для кожної ознаки.
        """
        current_data = np.atleast_2d(current_data)
        
        if current_data.shape[1] != self.baseline.shape[1]:
            raise ValueError(
                f"Feature mismatch: expected {self.baseline.shape[1]} columns, "
                f"got {current_data.shape[1]}"
            )

        feature_metrics = {}
        drifted_columns = []

        for idx, name in enumerate(self.feature_names):
            ref_values = self.baseline[:, idx]
            current_values = current_data[:, idx]

            statistic, p_value = stats.ks_2samp(ref_values, current_values)
            
            has_drift = bool(p_value < threshold)
            
            feature_metrics[name] = {
                "statistic": round(float(statistic), 4),
                "p_value": round(float(p_value), 5),
                "drift_detected": has_drift
            }
            
            if has_drift:
                drifted_columns.append(name)

        return {
            "drift_detected": len(drifted_columns) > 0,
            "n_drifted_features": len(drifted_columns),
            "drifted_features": drifted_columns,
            "per_feature": feature_metrics,
            "n_samples": int(current_data.shape[0]),
            "alpha": threshold
        }