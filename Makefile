.PHONY: help setup install lint format test benchmark serve docker-build docker-up docker-down clean

PYTHON ?= python
PIP ?= pip
PORT ?= 8000

help:
	@echo "Applied DS & ML Curriculum — Makefile Commands"
	@echo "------------------------------------------------"
	@echo "make setup        : Install locked dependencies"
	@echo "make lint         : Lint codebase with Ruff"
	@echo "make format       : Auto-format with Ruff"
	@echo "make test         : Run pytest suite with coverage"
	@echo "make benchmark    : Run Ingestion & Analytical SQL benchmarks"
	@echo "make serve        : Run FastAPI inference microservice"
	@echo "make docker-build : Build production multi-stage Docker image"
	@echo "make docker-up    : Launch API and Jupyter with Docker Compose"
	@echo "make docker-down  : Stop Docker Compose services"
	@echo "make clean        : Remove caches and temporary build artifacts"

setup: install

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

benchmark:
	$(PYTHON) -m src.benchmark --samples 25000 --runs 3

serve:
	uvicorn src.app:app --host 0.0.0.0 --port $(PORT) --reload

docker-build:
	docker build -t applied-ds-ml-api:latest .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f .coverage coverage.xml
