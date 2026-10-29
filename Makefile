.PHONY: help install db-up db-down dev test lint clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install Python dependencies"
	@echo "  make db-up      - Start PostgreSQL pgvector container in background"
	@echo "  make db-down    - Stop PostgreSQL container"
	@echo "  make dev        - Run FastAPI app with live reload"
	@echo "  make test       - Run end-to-end pipeline verification test"
	@echo "  make lint       - Run syntax and code quality checks"

install:
	pip install -r requirements.txt

db-up:
	docker compose up -d

db-down:
	docker compose down

dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	python run_test.py

lint:
	python -m flake8 app --count --select=E9,F63,F7,F82 --show-source --statistics

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

