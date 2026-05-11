import joblib
from pathlib import Path
from ml.train import IrisTrainer

def test_model_generation_process(tmp_path: Path):
    """Перевірка створення файлу моделі"""
    target_file = tmp_path / "model_v2.joblib"
    
    trainer = IrisTrainer(model_path=target_file)
    accuracy = trainer.execute()
    
    assert target_file.is_file()
    assert isinstance(accuracy, float)
    assert 0.9 <= accuracy <= 1.0

def test_classifier_output_format(tmp_path: Path):
    """Перевірка формату передбачення"""
    temp_model_path = tmp_path / "test_model.joblib"
    
    orchestrator = IrisTrainer(model_path=temp_model_path)
    orchestrator.execute()
    
    loaded_clf = joblib.load(temp_model_path)
    mock_input = [[5.1, 3.5, 1.4, 0.2]]
    prediction = loaded_clf.predict(mock_input)
    
    assert prediction[0] in {0, 1, 2}