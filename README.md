# Лабораторна робота №3
## Моніторинг ML API та детекція проблем (Data Drift)

[![CI](https://github.com/bondarevadiana-lgtm/mlops-ci-cd-ml-api/actions/workflows/ci.yml/badge.svg)](https://github.com/bondarevadiana-lgtm/mlops-ci-cd-ml-api/actions/workflows/ci.yml)

Цей репозиторій містить розширену версію MLOps-конвеєра, де основну увагу приділено спостережуваності (observability) та автоматичному аналізу якості вхідних даних. Система дозволяє відстежувати технічні метрики та бізнес-показники моделі в реальному часі.

## Опис системи
Сервіс розпізнає сорти ірисів на основі моделі `LogisticRegression`. В третій версії додано:
* **Інструментарій Prometheus**: збір метрик продуктивності та стабільності.
* **Drift Detection**: статистичний аналіз вхідних даних (тест Колмогорова-Смирнова) для виявлення деградації моделі.
* **Structured Logging**: JSON-логи для зручного аналізу в ELK-стеку.

## Реалізовані метрики (Prometheus)
* `model_inference_total`: загальна кількість прогнозів (з розподілом за сортами та статусом).
* `model_inference_latency_seconds`: час обробки запитів (Histogram).
* `model_confidence_level`: рівень впевненості моделі у своїх прогнозах.
* `model_is_active`: бінарний показник доступності моделі (Gauge).
* `model_drift_alerts_total`: лічильник виявлених статистичних аномалій у даних.

## Інструментарій
* **Моніторинг**: Prometheus 2.55.0
* **Аналіз даних**: Numpy, Scikit-learn (KS-test)
* **Візуалізація**: Evidently AI (HTML-звіти)
* **Контейнеризація**: Docker Compose

## Інструкція з розгортання

### 1. Локальний запуск (Development)
```bash
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate для Windows
pip install -r requirements.txt
python -m ml.train
uvicorn app.main:app --reload
```

### 2. Запуск повного стеку моніторингу (Docker)
Для підняття API разом із сервером Prometheus виконайте:
```bash
docker-compose up -d --build
```

Після цього сервіси будуть доступні за адресами:
- ML API: http://localhost:8000
- Prometheus: http://localhost:9090
- Metrics Endpoint: http://localhost:8000/metrics

## Приклади запитів
### Перевірка на Data Drift (нормальні дані)
```bash
curl -s -X POST http://localhost:8000/check-drift \
-H "Content-Type: application/json" \
-d '{"samples": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]], "alpha": 0.05}' | python -m json.tool
```
### Перевірка на Data Drift (аномальні дані)
```bash
curl -s -X POST http://localhost:8000/check-drift \
-H "Content-Type: application/json" \
-d '{"samples": [[9.5, 8.5, 8.0, 5.0], [9.1, 8.2, 7.9, 5.2]], "alpha": 0.05}' | python -m json.tool
```
## Тестування
Проєкт покритий тестами (API, ML-логіка, Drift, Metrics). Запуск:
```bash
pytest
```
## Висновки
В ході роботи було реалізовано повноцінну систему спостереження за ML-сервісом. На відміну від класичних веб-сервісів, тут відстежується не тільки працездатність коду, а й стабільність розподілу вхідних даних (Data Drift), що дозволяє вчасно виявити потребу в перенавчанні моделі.