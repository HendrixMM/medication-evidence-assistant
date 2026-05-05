# Testing

Run the backend suite:

```bash
python -m pytest
```

Run coverage:

```bash
python -m pytest --cov=src --cov-report=term-missing --cov-report=xml
```

Run the frontend checks:

```bash
cd frontend
npm install
npm test
npm run typecheck
```

The kept suite focuses on the finished product: the agent loop, API routes, evidence retrieval, medical retrieval, guardrails, synthesis, ranking, monitoring, and utility modules.
