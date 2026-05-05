# API Documentation

The FastAPI app is defined in `src/api/app.py`.

## Main Routes

| Route | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Runtime health check |
| `/api/chat` | `POST` | Server-sent event stream from the core evidence agent |
| `/api/agent/chat` | `POST` | OpenAI Agents SDK chat route |
| `/api/tool/retrieve_evidence` | `POST` | Legacy evidence retrieval compatibility route |
| `/api/tool/retrieve_medical_evidence` | `POST` | Consumer health retrieval via MedlinePlus |

OpenAPI documentation is available at `/docs` when the API server is running.

## Streaming Contract

`/api/chat` emits SSE events from the agent pipeline. A successful stream follows:

```text
started -> status(planning) -> status(searching) -> status(evaluating)
-> status(synthesizing) -> status(translating) -> answer_delta -> sources -> finished
```

Terminal errors are followed by `finished` and include the session id, usage, and stop reason.
