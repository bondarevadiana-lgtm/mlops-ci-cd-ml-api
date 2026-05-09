# Лабораторна робота №2
## CI/CD та ML API

[![CI](https://github.com/bondarevadiana-lgtm/mlops-ci-cd-ml-api/actions/workflows/ci.yml/badge.svg)](https://github.com/bondarevadiana-lgtm/mlops-ci-cd-ml-api/actions/workflows/ci.yml)

Даний репозиторій містить реалізацію автоматизованого конвеєра (MLOps) для розгортання моделі класифікації (Iris dataset). Система включає навчання моделі, створення API-інтерфейсу на базі FastAPI, пакування у Docker-контейнер та автоматичний деплой.

## Використані інструменти
* **Програмування**: Python 3.11
* **Машинне навчання**: Scikit-learn, Joblib
* **Веб-сервіс**: FastAPI, Pydantic, Uvicorn
* **CI/CD**: GitHub Actions
* **Інфраструктура**: Docker, Render

## Інструкція з локального розгортання

1. Підготовка середовища
   Створіть та активуйте ізольоване віртуальне середовище :

   ```bash
   python -m venv venv

   venv\Scripts\activate     # для Windows

   source venv/bin/activate  # для Linux/macOS
   

2. Встановіть компонентів

   Завантажте необхідні бібліотеки :

   ```bash
   pip install -r requirements.txt

3. Навчання та запуск
   Виконайте скрипт для генерації файлу моделі, після чого запустіть локальний сервер :
   ```bash
   python -m ml.train
   uvicorn app.maiin:app --reload
  

## Доступ до розгорнутого сервісу
Проект публікується за наступними посиланнями

- Api endpoint : https://mlops-ci-cd-ml-api.onrender.com
- Перевірка працездатності : https://mlops-ci-cd-ml-api.onrender.com/health
- Документація : https://mlops-ci-cd-ml-api.onrender.com/docs