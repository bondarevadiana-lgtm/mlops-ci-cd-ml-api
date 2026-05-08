import joblib
from pathlib import Path
from ml.train import IrisTrainer

def test_model_generation_process(tmp_path: Path):
    target_file = tmp_path / "model_v2.joblib"
    
    trainer = IrisTrainer(target_path=target_file)
    accuracy = trainer.execute()

    assert target_file.is_file(), "Артефакт моделі не був створений за вказаним шляхом"
    assert isinstance(accuracy, float), "Метод execute має повертати число (float)"
    assert 0.9 <= accuracy <= 1.0, f"Точність занадто низька або некоректна: {accuracy}"

def test_classifier_output_format(tmp_path: Path):
    temp_model_path = tmp_path / "test_model.joblib"
    
    orchestrator = IrisTrainer(target_path=temp_model_path)
    orchestrator.execute()
    
    loaded_clf = joblib.load(temp_model_path)
    
    mock_input = [[5.1, 3.5, 1.4, 0.2]]
    prediction = loaded_clf.predict(mock_input)
    
    assert prediction[0] in {0, 1, 2}, f"Неочікуваний ID класу: {prediction[0]}"
    
    probabilities = loaded_clf.predict_proba(mock_input)
    assert probabilities.shape == (1, 3), "Модель повинна повертати ймовірності для 3 класів"