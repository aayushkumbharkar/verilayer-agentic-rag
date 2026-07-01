# VeriLayer — Developer Makefile
# Usage: make <target>
.PHONY: help up down logs test test-unit test-live lint format typecheck migrate ingest-sample

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Docker Compose ─────────────────────────────────────────────────────────────

up:  ## Start all services (postgres, opensearch, redis, api, ui)
	docker compose up -d --build

down:  ## Stop all services
	docker compose down

logs:  ## Tail logs from all services
	docker compose logs -f

logs-api:  ## Tail API logs
	docker compose logs -f api

logs-ui:  ## Tail Gradio UI logs
	docker compose logs -f ui

restart-api:  ## Restart only the API container
	docker compose restart api

# ── Testing ────────────────────────────────────────────────────────────────────

test:  ## Run all unit + integration tests (no live services required)
	python -m pytest tests/ -v --tb=short -m "not live"

test-unit:  ## Run unit tests only
	python -m pytest tests/unit/ -v --tb=short

test-live:  ## Run ALL tests including live E2E (requires docker compose up)
	python -m pytest tests/ -v --tb=short -m live

# ── Code Quality ───────────────────────────────────────────────────────────────

lint:  ## Run ruff linter
	python -m ruff check src/ tests/ ui/

format:  ## Auto-format with ruff
	python -m ruff format src/ tests/ ui/

typecheck:  ## Run mypy type-checking
	python -m mypy src/ --ignore-missing-imports

# ── Database ───────────────────────────────────────────────────────────────────

migrate:  ## Apply Alembic migrations
	python -m alembic upgrade head

migrate-create:  ## Create a new migration (usage: make migrate-create MSG="add index")
	python -m alembic revision --autogenerate -m "$(MSG)"

# ── Dev Helpers ────────────────────────────────────────────────────────────────

ingest-sample:  ## Ingest a sample contract text via the API
	curl -s -X POST http://localhost:8000/ingest \
	  -H "Content-Type: application/json" \
	  -d '{"source_name": "sample_contract.txt", "content": "Force majeure is a clause in contracts that frees parties from obligations when extraordinary events prevent performance, such as natural disasters, war, or government actions."}' | python -m json.tool

health:  ## Check API health
	curl -s http://localhost:8000/health/services | python -m json.tool

run-ui:  ## Run Gradio UI locally (requires API running on localhost:8000)
	python -m ui.gradio_app

run-api:  ## Run FastAPI locally (requires services on localhost)
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
