import os
import joblib
from pathlib import Path
from sklearn import datasets, model_selection, linear_model, metrics

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SAVE_PATH = BASE_DIR / "model.joblib"

class IrisTrainer:
    def __init__(self, target_path: Path = DEFAULT_SAVE_PATH):
        self.save_path = target_path
        self.model = linear_model.LogisticRegression(max_iter=1000, solver='lbfgs')

    def execute(self) -> float:
        iris = datasets.load_iris()
        data, target = iris.data, iris.target

        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(
            data, 
            target, 
            test_size=0.25,     
            random_state=123,   
            stratify=target
        )

        self.model.fit(features_train, labels_train)

        predictions = self.model.predict(features_test)
        score = metrics.accuracy_score(labels_test, predictions)

        joblib.dump(self.model, self.save_path)
        
        return float(score)

if __name__ == "__main__":
    trainer = IrisTrainer()
    perf_metrics = trainer.execute()
    
    print(f"--- Training Completed ---")
    print(f"Accuracy: {perf_metrics:.2%}")
    print(f"Model location: {DEFAULT_SAVE_PATH}")