# Architecture

The product is a medication evidence chat system with a FastAPI backend and frontend client.

## Backend

- `src/api/` exposes HTTP and SSE routes.
- `src/agent/` owns the planning, retrieval, evaluation, synthesis, translation, streaming, and session loop.
- `src/evidence/` contains evidence schemas, source scoring, query construction, and RxNorm helpers.
- `src/medical_retrieval/` contains consumer health retrieval orchestration and MedlinePlus/PubMed-facing schema normalization.
- `src/clients/` wraps external LLM clients.
- `src/monitoring/` tracks quotas, credits, endpoint health, and alerts.

## Agent Pipeline

```text
Planner -> Retrieval -> Evaluator -> Summarizer -> Translator
```

Session state is persisted at stage boundaries. Terminal paths call `finalize_session` before emitting `finished`.
