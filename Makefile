# Flower Classification - Makefile
# =================================

.PHONY: help install dev train test lint format clean docker-build docker-run api cli

# Default target
help:
	@echo "Flower Classification ML Project"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install production dependencies"
	@echo "  dev         - Install development dependencies"
	@echo "  train       - Train the model with sample data"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linters"
	@echo "  format      - Format code with black"
	@echo "  clean       - Clean build artifacts"
	@echo "  api         - Start the FastAPI server"
	@echo "  cli         - Run CLI tool"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev]"

# Training
train:
	python backend/scripts/train.py --sample

train-custom:
	python backend/scripts/train.py --data $(DATA)

# Testing
test:
	pytest backend/tests/ -v --cov=backend/src

test-cov:
	pytest backend/tests/ -v --cov=backend/src --cov-report=html

# Code quality
lint:
	flake8 backend/src backend/app
	mypy backend/src backend/app

format:
	black backend/src backend/app

check: lint format

# Docker
docker-build:
	docker build -f backend/docker/Dockerfile -t flower-classification:latest .

docker-run:
	docker run -p 8000:8000 -v $(pwd)/backend/models:/app/backend/models flower-classification:latest

docker-compose-up:
	docker-compose -f backend/docker/docker-compose.yml up --build

docker-compose-down:
	docker-compose -f backend/docker/docker-compose.yml down

# API
api:
	python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# CLI
cli:
	python backend/scripts/cli.py $(ARGS)

demo:
	python backend/scripts/cli.py demo

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/
	@echo "Cleaned up build artifacts"

# Model management
save-model:
	@echo "Model artifacts are saved in backend/models/artifacts/"

load-sample-data:
	@echo "Sample data is loaded automatically when using --sample flag"
