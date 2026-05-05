from __future__ import annotations

import threading

import pytest
from fastapi import Request

import src.api.rate_limit as api_rate_limit
from src.api.rate_limit import HTTPClientRateLimiter, RateLimitExceeded


class TimeStub:
    def __init__(self) -> None:
        self.current = 0.0
        self._lock = threading.Lock()

    def time(self) -> float:
        with self._lock:
            return self.current

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.current += seconds


def make_request(headers: dict[str, str], client_host: str = "172.18.0.10") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/chat",
            "headers": [(name.lower().encode(), value.encode()) for name, value in headers.items()],
            "client": (client_host, 54321),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )


@pytest.mark.parametrize(
    ("env_value", "message"),
    [
        ("0", "API_CHAT_MAX_REQUESTS_PER_MINUTE must be an integer >= 1"),
        ("-1", "API_CHAT_MAX_REQUESTS_PER_MINUTE must be an integer >= 1"),
        ("not-an-int", "API_CHAT_MAX_REQUESTS_PER_MINUTE must be an integer >= 1"),
    ],
)
def test_http_rate_limiter_rejects_invalid_minute_limit(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    message: str,
) -> None:
    monkeypatch.setenv("API_CHAT_MAX_REQUESTS_PER_MINUTE", env_value)
    monkeypatch.delenv("API_CHAT_MAX_REQUESTS_PER_DAY", raising=False)

    with pytest.raises(ValueError, match=message):
        HTTPClientRateLimiter()


def test_http_rate_limiter_treats_unset_daily_limit_as_unlimited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_CHAT_MAX_REQUESTS_PER_MINUTE", "1")
    monkeypatch.delenv("API_CHAT_MAX_REQUESTS_PER_DAY", raising=False)

    limiter = HTTPClientRateLimiter()

    assert limiter.daily_limit is None


def test_get_client_ip_uses_trusted_forwarded_ip_when_forwarded_headers_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUST_FORWARDED_HEADERS", "true")
    request = make_request({"X-Forwarded-For": "203.0.113.25"}, client_host="172.18.0.10")

    assert HTTPClientRateLimiter.get_client_ip(request) == "203.0.113.25"


def test_get_client_ip_ignores_untrusted_forwarded_ip_when_forwarded_headers_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUST_FORWARDED_HEADERS", "false")
    request = make_request({"X-Forwarded-For": "198.51.100.99"}, client_host="172.18.0.10")

    assert HTTPClientRateLimiter.get_client_ip(request) == "172.18.0.10"


@pytest.mark.parametrize("env_value", ["", "   "])
def test_http_rate_limiter_treats_blank_daily_limit_as_unlimited(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
) -> None:
    monkeypatch.setenv("API_CHAT_MAX_REQUESTS_PER_MINUTE", "1")
    monkeypatch.setenv("API_CHAT_MAX_REQUESTS_PER_DAY", env_value)

    limiter = HTTPClientRateLimiter()

    assert limiter.daily_limit is None


@pytest.mark.parametrize("env_value", ["-1", "not-an-int"])
def test_http_rate_limiter_rejects_invalid_daily_limit(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
) -> None:
    monkeypatch.setenv("API_CHAT_MAX_REQUESTS_PER_MINUTE", "1")
    monkeypatch.setenv("API_CHAT_MAX_REQUESTS_PER_DAY", env_value)

    with pytest.raises(ValueError, match="API_CHAT_MAX_REQUESTS_PER_DAY must be an integer >= 0"):
        HTTPClientRateLimiter()


def test_http_rate_limiter_allows_zero_daily_limit_as_zero_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_CHAT_MAX_REQUESTS_PER_MINUTE", "1")
    monkeypatch.setenv("API_CHAT_MAX_REQUESTS_PER_DAY", "0")
    limiter = HTTPClientRateLimiter()

    assert limiter.daily_limit == 0
    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.check_and_consume("127.0.0.1")

    assert exc_info.value.detail == "Daily API chat limit exceeded."


def test_check_and_consume_defensively_handles_empty_minute_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_CHAT_MAX_REQUESTS_PER_MINUTE", "1")
    monkeypatch.delenv("API_CHAT_MAX_REQUESTS_PER_DAY", raising=False)
    limiter = HTTPClientRateLimiter()
    limiter.max_requests_per_minute = 0

    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.check_and_consume("127.0.0.1")

    assert exc_info.value.detail == "Per-minute API chat rate limit exceeded."
    assert exc_info.value.retry_after_seconds >= 0.0


def test_http_rate_limiter_uses_monotonic_time_for_minute_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monotonic_stub = TimeStub()
    wall_clock_values = iter([100.0, 50.0])

    monkeypatch.setenv("API_CHAT_MAX_REQUESTS_PER_MINUTE", "1")
    monkeypatch.delenv("API_CHAT_MAX_REQUESTS_PER_DAY", raising=False)
    monkeypatch.setattr(api_rate_limit, "_now_monotonic", monotonic_stub.time, raising=False)
    monkeypatch.setattr(api_rate_limit.time, "time", lambda: next(wall_clock_values))

    limiter = HTTPClientRateLimiter()
    limiter.check_and_consume("127.0.0.1")

    monotonic_stub.sleep(30.0)

    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.check_and_consume("127.0.0.1")

    assert exc_info.value.detail == "Per-minute API chat rate limit exceeded."
    assert exc_info.value.retry_after_seconds == pytest.approx(30.0)
