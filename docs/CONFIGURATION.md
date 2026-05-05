# Configuration

Configuration is read from environment variables and `.env`.

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required for OpenAI-backed planning and translation | unset |
| `AGENT_MODE_ENABLED` | Enables `/api/chat` and `/api/agent/chat` | `true` |
| `AGENT_MODEL` | Planner and evaluator model | `gpt-5.4-mini` |
| `AGENT_SESSION_DIR` | Writable session persistence directory | `.agent_sessions` |
| `ENABLE_MEDICAL_GUARDRAILS` | Enables medical query/response validation | `false` in tests |
| `MEDLINEPLUS_EMAIL` | Optional NLM contact email for MedlinePlus calls | unset |
| `MEDLINEPLUS_MAX_RESULTS` | Max MedlinePlus results per request | `5` |
| `HTTP_RATE_LIMIT_PER_MINUTE` | API rate limit per client IP | `60` |

Medical guardrails can run in lightweight regex mode. Install `requirements-medical.txt` only when the heavier Presidio or NeMo integrations are needed.
