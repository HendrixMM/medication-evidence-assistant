"""API contract tests for /api/chat, /api/session, and /api/health endpoints."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.agent.config import AgentConfig
from src.agent.loop import AgentLoop
from src.agent.models import SessionState, UsageStats
from src.agent.retrieval_quality import RetrievalQualityScorer
import src.api.app as app_module
from src.api.app import app, get_agent_loop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_loop():
    loop = Mock(spec=AgentLoop)
    loop.session_manager = Mock()

    async def _stream(*_args, **_kwargs):
        yield 'data: {"event": "finished", "data": {"session_id": "s1", "usage": {}, "stop_reason": "sufficient"}}\n\n'

    loop.answer_streaming.side_effect = _stream
    return loop


@pytest.fixture()
def mock_rate_limiter(monkeypatch):
    """Patch get_http_limiter to return a deterministic limiter instance."""
    from src.api import rate_limit as rl_module

    limiter = Mock(spec=rl_module.HTTPClientRateLimiter)
    limiter.get_client_ip.return_value = "127.0.0.1"
    limiter.status.return_value = {
        "requests_last_minute": 0,
        "max_requests_per_minute": 10,
        "remaining_minute": 10,
        "retry_after_seconds": 0.0,
        "daily_count": 0,
        "daily_limit": None,
        "remaining_daily": None,
    }
    limiter.check_and_consume.return_value = None
    monkeypatch.setattr(rl_module, "_http_limiter", limiter)
    return limiter


@pytest.fixture()
def client(mock_loop, mock_rate_limiter):
    app_module._agent_loop = None
    app.dependency_overrides[app_module.get_agent_loop_composer] = lambda: (lambda: mock_loop)
    yield TestClient(app)
    app.dependency_overrides.clear()
    app_module._agent_loop = None


def _make_session(session_id: str = "abc123") -> SessionState:
    return SessionState(
        session_id=session_id,
        query="What is aspirin?",
        turns=[],
        all_articles=[],
        usage=UsageStats(),
    )


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------


def test_chat_returns_event_stream(client: TestClient) -> None:
    with client.stream("POST", "/api/chat", json={"question": "test query"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.read().decode()
    assert "data:" in body


def test_chat_emits_data_frames(client: TestClient) -> None:
    with client.stream("POST", "/api/chat", json={"question": "aspirin interactions"}) as resp:
        lines = resp.read().decode().splitlines()
    data_lines = [ln for ln in lines if ln.startswith("data:")]
    assert len(data_lines) >= 1


def test_rate_limit_status_returns_remaining_for_client(client: TestClient, mock_rate_limiter: Mock) -> None:
    mock_rate_limiter.status.return_value = {
        "requests_last_minute": 3,
        "max_requests_per_minute": 10,
        "remaining_minute": 7,
        "retry_after_seconds": 0.0,
        "daily_count": 3,
        "daily_limit": None,
        "remaining_daily": None,
    }

    resp = client.get("/api/rate-limit/status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["remaining_minute"] == 7
    mock_rate_limiter.check_and_consume.assert_not_called()


def test_rate_limit_status_does_not_consume_slot(client: TestClient, mock_rate_limiter: Mock) -> None:
    for _ in range(3):
        resp = client.get("/api/rate-limit/status")
        assert resp.status_code == 200

    assert mock_rate_limiter.check_and_consume.call_count == 0


def test_chat_calls_rate_limiter_before_streaming(client: TestClient, mock_rate_limiter: Mock) -> None:
    with client.stream("POST", "/api/chat", json={"question": "test query"}) as resp:
        assert resp.status_code == 200
        resp.read()

    mock_rate_limiter.check_and_consume.assert_called_once_with("127.0.0.1")


def test_chat_returns_429_when_rate_limited(client: TestClient, mock_rate_limiter: Mock) -> None:
    from src.api.rate_limit import RateLimitExceeded

    mock_rate_limiter.check_and_consume.side_effect = RateLimitExceeded(
        retry_after_seconds=42.0,
        detail="Rate limit exceeded",
    )

    resp = client.post("/api/chat", json={"question": "second"})

    assert resp.status_code == 429
    body = resp.json()
    assert body["detail"] == "Rate limit exceeded"
    assert body["retry_after_seconds"] == 42.0
    assert resp.headers["Retry-After"] == "42"


def test_chat_rate_limit_keyed_per_client_host(
    mock_loop: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.api import rate_limit as rl_module

    monkeypatch.setenv("API_CHAT_MAX_REQUESTS_PER_MINUTE", "1")
    monkeypatch.delenv("API_CHAT_MAX_REQUESTS_PER_DAY", raising=False)
    monkeypatch.setattr(rl_module, "_http_limiter", None)
    app_module._agent_loop = None
    app.dependency_overrides[app_module.get_agent_loop_composer] = lambda: (lambda: mock_loop)

    client_one = TestClient(app, client=("10.0.0.1", 50000))
    client_two = TestClient(app, client=("10.0.0.2", 50000))

    with client_one.stream("POST", "/api/chat", json={"question": "first"}) as resp:
        assert resp.status_code == 200
        resp.read()
    with client_two.stream("POST", "/api/chat", json={"question": "other client"}) as resp:
        assert resp.status_code == 200
        resp.read()

    resp = client_one.post("/api/chat", json={"question": "same client again"})

    assert resp.status_code == 429
    app.dependency_overrides.clear()
    app_module._agent_loop = None
    monkeypatch.setattr(rl_module, "_http_limiter", None)


def test_chat_options_preflight_returns_cors_headers(client: TestClient) -> None:
    resp = client.options(
        "/api/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers


# ---------------------------------------------------------------------------
# GET /api/session/{session_id}
# ---------------------------------------------------------------------------


def test_get_session_unknown_returns_404(client: TestClient, mock_loop: Mock) -> None:
    mock_loop.session_manager.load_session.return_value = None
    resp = client.get("/api/session/unknown-id")
    assert resp.status_code == 404


def test_get_session_known_returns_schema(client: TestClient, mock_loop: Mock) -> None:
    session = _make_session("abc123")
    mock_loop.session_manager.load_session.return_value = session
    resp = client.get("/api/session/abc123")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "abc123"
    assert body["query"] == "What is aspirin?"
    assert isinstance(body["turns"], list)
    assert isinstance(body["all_articles"], list)


# ---------------------------------------------------------------------------
# AGENT_MODE_ENABLED=false enforcement
# ---------------------------------------------------------------------------


def test_chat_returns_503_when_agent_mode_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.api.app as app_module

    monkeypatch.setattr(app_module, "AGENT_MODE_ENABLED", False)
    c = TestClient(app)
    resp = c.post("/api/chat", json={"question": "test"})
    assert resp.status_code == 503
    assert "disabled" in resp.json()["detail"].lower()


def test_session_returns_503_when_agent_mode_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.api.app as app_module

    monkeypatch.setattr(app_module, "AGENT_MODE_ENABLED", False)
    c = TestClient(app)
    resp = c.get("/api/session/some-id")
    assert resp.status_code == 503
    assert "disabled" in resp.json()["detail"].lower()


def test_chat_returns_503_when_agent_initialization_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "AGENT_MODE_ENABLED", True)
    monkeypatch.setattr(app_module, "_agent_loop", None)
    app.dependency_overrides[app_module.get_agent_loop_composer] = lambda: (
        lambda: (_ for _ in ()).throw(RuntimeError("NVIDIA_API_KEY is missing"))
    )
    c = TestClient(app)

    resp = c.post("/api/chat", json={"question": "test"})

    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()
    assert "nvidia" not in resp.json()["detail"].lower()
    app.dependency_overrides.clear()
    app_module._agent_loop = None


def test_session_returns_503_when_agent_initialization_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "AGENT_MODE_ENABLED", True)
    monkeypatch.setattr(app_module, "_agent_loop", None)
    app.dependency_overrides[app_module.get_agent_loop_composer] = lambda: (
        lambda: (_ for _ in ()).throw(RuntimeError("OpenAI SDK not installed"))
    )
    c = TestClient(app)

    resp = c.get("/api/session/some-id")

    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()
    assert "openai" not in resp.json()["detail"].lower()
    app.dependency_overrides.clear()


def test_get_agent_loop_initializes_once_under_concurrent_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "AGENT_MODE_ENABLED", True)
    monkeypatch.setattr(app_module, "_agent_loop", None)

    created_loop = Mock(spec=AgentLoop)
    create_call_count = 0
    start_barrier = threading.Barrier(4)
    errors: list[BaseException] = []
    results: list[AgentLoop] = []

    def _slow_create_agent_loop():
        nonlocal create_call_count
        create_call_count += 1
        time.sleep(0.05)
        return created_loop

    def _resolve_loop() -> None:
        try:
            start_barrier.wait()
            results.append(app_module._get_or_create_agent_loop(_slow_create_agent_loop))
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    threads = [threading.Thread(target=_resolve_loop) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert create_call_count == 1
    assert results == [created_loop] * 4


def test_chat_supports_agent_loop_composer_dependency_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "AGENT_MODE_ENABLED", True)
    monkeypatch.setattr(app_module, "_agent_loop", None)

    composer_loop = Mock(spec=AgentLoop)

    async def _stream(*_args, **_kwargs):
        yield 'data: {"event": "finished", "data": {"session_id": "s2", "usage": {}, "stop_reason": "sufficient"}}\n\n'

    composer_loop.answer_streaming.side_effect = _stream
    app.dependency_overrides[app_module.get_agent_loop_composer] = lambda: (lambda: composer_loop)
    c = TestClient(app)

    with c.stream("POST", "/api/chat", json={"question": "override test"}) as resp:
        assert resp.status_code == 200
        body = resp.read().decode()

    assert "data:" in body
    app.dependency_overrides.clear()
    app_module._agent_loop = None


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------


def test_health_ok_fields(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["agent_mode_enabled"], bool)
    assert isinstance(body["version"], str) and body["version"]


def test_health_returns_200_when_agent_mode_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "AGENT_MODE_ENABLED", False)
    app_module._agent_loop = None

    resp = TestClient(app).get("/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["agent_mode_enabled"] is False


def test_health_does_not_force_agent_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "AGENT_MODE_ENABLED", True)
    monkeypatch.setattr(app_module, "_agent_loop", None)
    app.dependency_overrides[app_module.get_agent_loop_composer] = lambda: (
        lambda: (_ for _ in ()).throw(RuntimeError("guardrails config missing"))
    )

    resp = TestClient(app).get("/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["agent_mode_enabled"] is True
    app.dependency_overrides.clear()
    app_module._agent_loop = None


def test_medical_guardrails_factory_uses_resolved_config_path() -> None:
    config = app_module.AgentConfig(enable_guardrails=False)
    expected_path = "/tmp/api-medical-guardrails.json"

    with patch("src.medical_guardrails.MedicalGuardrails") as mock_guardrails:
        with patch("src.medical_guardrails.resolve_medical_guardrails_config_path", return_value=Path(expected_path)):
            factory = app_module.get_medical_guardrails_factory()
            factory(config)

    mock_guardrails.assert_called_once_with(expected_path, enabled=config.enable_guardrails)


def test_retrieval_quality_scorer_factory_passes_relevance_scorer() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    relevance_scorer = object()

    factory = app_module.get_retrieval_quality_scorer_factory()
    scorer = factory(config, relevance_scorer)

    assert isinstance(scorer, RetrievalQualityScorer)
    assert scorer.source_relevance_scorer is relevance_scorer


def test_evaluator_factory_requires_retrieval_quality_scorer() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    retrieval_quality_scorer = object()

    factory = app_module.get_evaluator_factory()
    evaluator = factory(config, retrieval_quality_scorer)

    assert evaluator.retrieval_quality_scorer is retrieval_quality_scorer
