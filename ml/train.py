import joblib
import numpy as np
from pathlib import Path
from sklearn import datasets, model_selection, linear_model, metrics

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_ARTIFACT = BASE_DIR / "model.joblib"
STATS_ARTIFACT = BASE_DIR / "reference_stats.joblib"

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

class IrisTrainer:
    def __init__(self, model_path: Path = MODEL_ARTIFACT, stats_path: Path = STATS_ARTIFACT):
        self.model_path = model_path
        self.stats_path = stats_path
 
        self.estimator = linear_model.LogisticRegression(max_iter=1000, solver='lbfgs')

    def execute(self) -> float:
        """
        Завантажує дані, тренує модель та зберігає артефакти для API та Drift Detection.
        """
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            X, y, 
            test_size=0.25, 
            random_state=123, 
            stratify=y
        )

        self.estimator.fit(x_train, y_train)

        predictions = self.estimator.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        joblib.dump(self.estimator, self.model_path)

        reference_payload = {
            "X": x_train,
            "feature_names": FEATURES
        }
        joblib.dump(reference_payload, self.stats_path)

        return float(accuracy)

def train_and_save(model_path: Path = MODEL_ARTIFACT) -> float:
    """Обгортка для сумісності зі старими тестами, якщо вони її вимагають."""
    trainer = IrisTrainer(model_path=model_path)
    return trainer.execute()

if __name__ == "__main__":
    print("--- Starting ML Training Pipeline ---")
    orchestrator = IrisTrainer()
    final_acc = orchestrator.execute()
    
    print(f"Status: Success")
    print(f"Accuracy Score: {final_acc:.4%}")
    print(f"Model saved to: {MODEL_ARTIFACT.name}")
    print(f"Stats saved to: {STATS_ARTIFACT.name}")