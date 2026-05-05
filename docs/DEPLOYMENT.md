# Deployment

The supported deployment path is Docker Compose.

```bash
cp .env.example .env
docker compose up --build
```

Required production settings:

- `OPENAI_API_KEY`
- `AGENT_SESSION_DIR` pointing at a writable mounted volume
- explicit CORS origins for the frontend domain
- rate limit settings appropriate for the deployment target

The API image is built from `Dockerfile.api`. Runtime session data should be mounted outside the image so sessions survive container restarts.
