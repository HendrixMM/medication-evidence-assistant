# Contributing

This is primarily a portfolio project, but issues and small improvements are welcome.

## Development

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
cd frontend && npm ci && npm test && npm run typecheck
```

## Pull Requests

- Keep changes scoped.
- Do not commit secrets or generated cache files.
- Preserve public API, SSE, session, and response-schema behavior unless the change intentionally updates those contracts.
- Add or update tests for backend behavior changes.
