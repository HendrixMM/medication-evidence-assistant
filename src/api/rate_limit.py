"""HTTP client rate limiting for API chat endpoints."""
from __future__ import annotations

from collections import deque
from datetime import UTC, date, datetime, timedelta
import os
import threading
import time

from fastapi import Request


def _now_monotonic() -> float:
    """Module-level monotonic clock for minute-window calculations."""
    return time.monotonic()


class RateLimitExceeded(Exception):
    """Raised when an HTTP client exceeds an API rate limit."""

    def __init__(self, *, retry_after_seconds: float, detail: str) -> None:
        super().__init__(detail)
        self.retry_after_seconds = retry_after_seconds
        self.detail = detail


class HTTPClientRateLimiter:
    """Sliding-window per-client limiter for the public HTTP API."""

    def __init__(self) -> None:
        self.max_requests_per_minute = self._parse_required_positive_int(
            "API_CHAT_MAX_REQUESTS_PER_MINUTE",
            os.getenv("API_CHAT_MAX_REQUESTS_PER_MINUTE", "10"),
        )
        raw_daily_limit = os.getenv("API_CHAT_MAX_REQUESTS_PER_DAY")
        self.daily_limit = self._parse_optional_non_negative_int(
            "API_CHAT_MAX_REQUESTS_PER_DAY",
            raw_daily_limit,
        )
        self._minute_windows: dict[str, deque[float]] = {}
        self._daily_counts: dict[str, int] = {}
        self._daily_reset_dates: dict[str, date] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _parse_required_positive_int(name: str, raw_value: str | None) -> int:
        try:
            value = int(raw_value or "")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer >= 1") from exc
        if value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
        return value

    @staticmethod
    def _parse_optional_non_negative_int(name: str, raw_value: str | None) -> int | None:
        if raw_value is None:
            return None
        normalized = raw_value.strip()
        if normalized == "":
            return None
        try:
            value = int(normalized)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer >= 0 when set") from exc
        if value < 0:
            raise ValueError(f"{name} must be an integer >= 0 when set")
        return value

    @staticmethod
    def get_client_ip(request: Request) -> str:
        trust_forwarded = os.getenv("TRUST_FORWARDED_HEADERS", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if trust_forwarded:
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                forwarded_ip = forwarded_for.split(",", maxsplit=1)[0].strip()
                if forwarded_ip:
                    return forwarded_ip
        if request.client is not None and request.client.host:
            return request.client.host
        return "unknown"

    def status(self, client_ip: str) -> dict:
        now_monotonic = _now_monotonic()
        today = datetime.now(UTC).date()
        with self._lock:
            window = self._minute_windows.setdefault(client_ip, deque())
            self._prune_window(window, now_monotonic)
            self._reset_daily_if_needed(client_ip, today)
            daily_count = self._daily_counts.get(client_ip, 0)

            remaining_minute = max(0, self.max_requests_per_minute - len(window))
            remaining_daily = (
                None
                if self.daily_limit is None
                else max(0, self.daily_limit - daily_count)
            )
            retry_after_seconds = self._retry_after_seconds(window, now_monotonic, daily_count)

            return {
                "requests_last_minute": len(window),
                "max_requests_per_minute": self.max_requests_per_minute,
                "remaining_minute": remaining_minute,
                "retry_after_seconds": retry_after_seconds,
                "daily_count": daily_count,
                "daily_limit": self.daily_limit,
                "remaining_daily": remaining_daily,
            }

    def check_and_consume(self, client_ip: str) -> None:
        now_monotonic = _now_monotonic()
        today = datetime.now(UTC).date()
        with self._lock:
            window = self._minute_windows.setdefault(client_ip, deque())
            self._prune_window(window, now_monotonic)
            self._reset_daily_if_needed(client_ip, today)
            daily_count = self._daily_counts.get(client_ip, 0)

            if self.daily_limit is not None and daily_count >= self.daily_limit:
                raise RateLimitExceeded(
                    retry_after_seconds=self._seconds_until_midnight_utc(),
                    detail="Daily API chat limit exceeded.",
                )
            if len(window) >= self.max_requests_per_minute:
                retry_after = self._retry_after_seconds(window, now_monotonic, daily_count)
                raise RateLimitExceeded(
                    retry_after_seconds=retry_after,
                    detail="Per-minute API chat rate limit exceeded.",
                )

            window.append(now_monotonic)
            self._daily_counts[client_ip] = daily_count + 1
            self._daily_reset_dates[client_ip] = today

    def _retry_after_seconds(self, window: deque[float], now_monotonic: float, daily_count: int) -> float:
        if self.daily_limit is not None and daily_count >= self.daily_limit:
            return self._seconds_until_midnight_utc()
        if len(window) >= self.max_requests_per_minute and window:
            return max(0.0, 60.0 - (now_monotonic - window[0]))
        if len(window) >= self.max_requests_per_minute:
            return 0.0
        return 0.0

    @staticmethod
    def _prune_window(window: deque[float], now_monotonic: float) -> None:
        while window and now_monotonic - window[0] >= 60.0:
            window.popleft()

    def _reset_daily_if_needed(self, client_ip: str, today: date) -> None:
        if self._daily_reset_dates.get(client_ip) != today:
            self._daily_counts[client_ip] = 0
            self._daily_reset_dates[client_ip] = today

    @staticmethod
    def _seconds_until_midnight_utc() -> float:
        now = datetime.now(UTC)
        tomorrow = now.date() + timedelta(days=1)
        midnight = datetime.combine(tomorrow, datetime.min.time(), tzinfo=UTC)
        return max(0.0, (midnight - now).total_seconds())


_http_limiter: HTTPClientRateLimiter | None = None
_http_limiter_lock = threading.Lock()


def get_http_limiter() -> HTTPClientRateLimiter:
    global _http_limiter
    if _http_limiter is None:
        with _http_limiter_lock:
            if _http_limiter is None:
                _http_limiter = HTTPClientRateLimiter()
    return _http_limiter


__all__ = ["HTTPClientRateLimiter", "RateLimitExceeded", "get_http_limiter"]
