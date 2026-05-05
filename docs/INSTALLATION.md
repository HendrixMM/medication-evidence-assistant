# Installation

## Local Python

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` for the API and agent routes.

## Docker

```bash
docker compose up --build
```

The API runs on `http://localhost:8000`. The frontend runs on the port configured by `frontend/`.
