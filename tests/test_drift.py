import numpy as np
import pytest
from app.drift import DriftAnalyzer

FEATURE_LABELS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

def test_no_drift_on_consistent_data():
    """Тест: якщо розподіли однакові, дрифт не має бути виявлений"""
    seed_gen = np.random.default_rng(42)

    baseline = seed_gen.normal(loc=5.0, scale=1.0, size=(100, 4))
    current = seed_gen.normal(loc=5.0, scale=1.0, size=(100, 4))
    
    analyzer = DriftAnalyzer(baseline, FEATURE_LABELS)
    report = analyzer.analyze(current, threshold=0.05)
    
    assert report["drift_detected"] is False
    assert report["n_drifted_features"] == 0
    assert len(report["per_feature"]) == 4

def test_drift_detection_on_shifted_data():
    """Тест: суттєвий зсув середнього значення має тригернути дрифт"""
    seed_gen = np.random.default_rng(42)
    baseline = seed_gen.normal(loc=5.0, scale=1.0, size=(100, 4))

    current = seed_gen.normal(loc=15.0, scale=1.0, size=(100, 4))
    
    analyzer = DriftAnalyzer(baseline, FEATURE_LABELS)
    report = analyzer.analyze(current, threshold=0.01)
    
    assert report["drift_detected"] is True
    assert report["n_drifted_features"] == 4

    assert report["per_feature"]["sepal_length"]["p_value"] < 0.01

def test_analyzer_output_integrity():
    """Перевірка структури відповіді та коректності типів даних"""
    base = np.zeros((20, 4))
    curr = np.ones((20, 4))
    
    analyzer = DriftAnalyzer(base, FEATURE_LABELS)
    report = analyzer.analyze(curr)
    
    expected_keys = ["drift_detected", "n_drifted_features", "per_feature", "n_samples", "alpha"]
    for key in expected_keys:
        assert key in report
    
    feature_info = report["per_feature"]["sepal_length"]
    assert "statistic" in feature_info
    assert "p_value" in feature_info
    assert isinstance(feature_info["drift_detected"], bool)

def test_mismatched_dimensions_error():
    """Тест: аналізатор має видавати помилку, якщо кількість колонок не збігається"""
    base = np.zeros((10, 4))
    curr_wrong = np.zeros((10, 3)) 
    
    analyzer = DriftAnalyzer(base, FEATURE_LABELS)
    with pytest.raises(ValueError, match="Feature mismatch"):
        analyzer.analyze(curr_wrong)