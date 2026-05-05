.PHONY: help install install-dev test test-core test-frontend api frontend docker-up docker-down clean

help:
	@echo "Medication Evidence Chat"
	@echo ""
	@echo "Targets:"
	@echo "  install        Install backend runtime dependencies"
	@echo "  install-dev    Install backend runtime + dev dependencies"
	@echo "  test           Run backend and frontend tests"
	@echo "  test-core      Run backend tests"
	@echo "  test-frontend  Run frontend tests"
	@echo "  api            Run FastAPI locally on :8000"
	@echo "  frontend       Run Vite locally on :5173"
	@echo "  docker-up      Build and run Docker product stack"
	@echo "  docker-down    Stop Docker product stack"
	@echo "  clean          Remove generated local artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

test: test-core test-frontend

test-core:
	pytest

test-frontend:
	cd frontend && npm test

api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	rm -rf frontend/dist frontend/tsconfig.tsbuildinfo
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
