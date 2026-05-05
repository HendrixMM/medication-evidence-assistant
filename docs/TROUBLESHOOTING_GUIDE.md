# Troubleshooting

## API Will Not Start

- Confirm `OPENAI_API_KEY` is set.
- Run `python -m compileall src` to catch import errors.
- Check that `AGENT_SESSION_DIR` is writable.

## Docker Build Fails

- Rebuild from a clean image cache with `docker compose build --no-cache`.
- Confirm `.env` exists and does not contain quoted multiline values.

## Chat Stream Stops Early

- Check API logs for terminal `pubmed_failure`, `translator_failure`, or `safety_check` events.
- Verify outbound network access to OpenAI and NLM/MedlinePlus.

## Medical Guardrails Dependencies

The default product path uses lightweight validators. Install `requirements-medical.txt` only when enabling the optional Presidio or NeMo integrations.
