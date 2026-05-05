# Medication Evidence Assistant

A personal project exploring medication-information retrieval with a Python/FastAPI backend, a React chat UI, and a tested evidence pipeline for source-linked answers.

I built this project to answer a practical question: can a medication assistant be useful without pretending to be a doctor? The backend retrieves consumer health information and biomedical evidence, normalizes it into one response schema, and returns guarded answers with citations, disclaimers, and structured fallback behavior when a source or model path fails.

## What It Does

- Runs a local product stack with Docker Compose.
- Serves a React/Vite chat UI on `http://localhost:3000`.
- Serves the FastAPI backend on `http://localhost:8000`.
- Uses MedlinePlus for consumer-friendly medication information.
- Supports PubMed/E-utilities and a medical retrieval pipeline for evidence-oriented workflows.
- Streams agent responses with stage events, citations, usage metadata, and terminal error handling.
- Includes rate limiting, CORS controls, medical guardrail hooks, monitoring helpers, and session persistence.

## Technical Highlights

- Containerized a Python/FastAPI backend and React frontend with Docker Compose for reproducible local deployment.
- Designed fallback and partial-response paths so retrieval or generation failures return structured errors or abstention responses instead of crashing the API.
- Wrote 617 pytest tests reaching 87.2% coverage across the agent, API, evidence, and medical retrieval packages.
- Implemented structured logging and monitoring utilities for endpoint health, alerts, quotas, and API troubleshooting.
- Built an adapter layer that normalizes results from multiple retrieval backends into a single schema, allowing sources to be swapped without modifying consumer code.

## Architecture

```text
frontend/              React chat client
src/api/               FastAPI app, routes, rate limiting
src/agent/             Planner, evaluator, tools, streaming loop, sessions
src/evidence/          Evidence policy, scoring, query building, schemas
src/medical_retrieval/ Medication retrieval, claim generation, verification
src/clients/           OpenAI-compatible LLM clients
src/monitoring/        Alert, quota, credit, and endpoint health helpers
tests/                 Backend unit, integration, API, and domain tests
```

The main Docker path runs `uvicorn src.api.app:app` for the API and builds the frontend from `frontend/`.

## Quick Start

```bash
cp .env.example .env
docker compose up --build
```

Then open:

- Frontend: `http://localhost:3000`
- API health: `http://localhost:8000/api/health`
- API docs: `http://localhost:8000/docs`

## Local Development

Backend:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm ci
npm run dev
```

## Tests

```bash
python -m pytest -q
python -m pytest --cov=src --cov-report=term
cd frontend && npm test && npm run typecheck
```

Current verified backend result from the cleaned repo:

- `617 passed, 1 skipped`
- `84.6%` coverage across `src/agent` and `src/api`
- `87.2%` coverage across `src/agent`, `src/api`, `src/evidence`, and `src/medical_retrieval`

## Safety

This project provides general medication information with supporting references. It is not a diagnostic, prescribing, or emergency-care system. Responses are designed to include source context and medical disclaimers, and the API exposes structured error or abstention paths instead of silently fabricating an answer when evidence is unavailable.

## License

MIT. See `LICENSE`.
