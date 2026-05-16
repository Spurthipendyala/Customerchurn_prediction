# Churn MLOps Pipeline Makefile

.PHONY: help setup install venv ingest validate preprocess train monitor docker-up docker-down

VENV := venv
PYTHON := $(VENV)/Scripts/python
PIP := $(VENV)/Scripts/pip

help:
	@echo ""
	@echo "  Telco Churn MLOps Pipeline"
	@echo "  ─────────────────────────────────────────────"
	@echo "  setup          Create venv + install deps + init DVC"
	@echo "  install        Install Python requirements"
	@echo "  ingest         Run data ingestion (OpenLineage)"
	@echo "  validate       Run Great Expectations validation + store in ClickHouse"
	@echo "  preprocess     Run feature engineering + store in ClickHouse"
	@echo "  featurize      Materialize features to Feast"
	@echo "  train          Train all models with MLflow"
	@echo "  pipeline       Run full DVC pipeline"
	@echo "  monitor        Run Evidently drift detection"
	@echo "  api            Start FastAPI server"
	@echo "  frontend       Start React frontend"
	@echo "  docker-up      Start all services (ClickHouse, Marquez, MLflow, etc.)"
	@echo "  docker-down    Stop all services"
	@echo "  test           Run unit tests"
	@echo ""

setup: venv install dvc-init

venv:
	python -m venv $(VENV)
	@echo "✅ Virtual environment created"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

dvc-init:
	$(PYTHON) -m dvc init --no-scm || true
	@echo "✅ DVC initialized"

ingest:
	$(PYTHON) -m src.data.ingest

validate:
	$(PYTHON) -m src.data.validate

preprocess:
	$(PYTHON) -m src.data.preprocess

featurize:
	$(PYTHON) -m src.features.feature_store

train:
	$(PYTHON) -m src.models.train

pipeline:
	$(PYTHON) -m dvc repro

monitor:
	$(PYTHON) -m src.monitoring.drift_detection

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm start

docker-up:
	docker-compose -f docker/docker-compose.yml up -d
	@echo "✅ All services started"
	@echo "   ClickHouse:  http://localhost:8123"
	@echo "   Marquez:     http://localhost:3000"
	@echo "   MLflow:      http://localhost:5001"
	@echo "   Prometheus:  http://localhost:9090"
	@echo "   Grafana:     http://localhost:3002 (admin/admin)"
	@echo "   API:         http://localhost:8000/docs"
	@echo "   Frontend:    http://localhost:3001"

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-build:
	docker-compose -f docker/docker-compose.yml build

test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov=api

all: ingest validate preprocess featurize train monitor
	@echo "🎉 Full pipeline complete!"
