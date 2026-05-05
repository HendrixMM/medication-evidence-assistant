"""OpenAI LLM client adapter for the agent workflow.

Mirrors the public surface of :class:`src.nvidia_llm_client.NVIDIALLMClient` so the
agent (planner, translator, source relevance scorer, etc.) can swap providers
without code changes. Exposes ``generate(...)``, ``generate_simple(...)``, and
``test_connection(...)`` returning plain strings / status dicts.

Model identifiers are passed through verbatim to the OpenAI Chat Completions API.
The caller (agent config) is responsible for choosing a model the account
supports.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OpenAILLMError(Exception):
    """Base exception for OpenAILLMClient errors."""


class OpenAILLMConfigError(OpenAILLMError):
    """Raised when the client is misconfigured (missing key, SDK, etc.)."""


class OpenAILLMAPIError(OpenAILLMError):
    """Raised when an OpenAI API call fails."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


def _parse_env_float(var_name: str, default: float) -> float:
    value = os.getenv(var_name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_env_int(var_name: str, default: int) -> int:
    value = os.getenv(var_name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_openai():
    try:
        from openai import (
            APIConnectionError,
            APIError,
            APIStatusError,
            OpenAI,
            RateLimitError,
        )
    except ImportError as exc:
        raise OpenAILLMConfigError(
            "OpenAI SDK is required. Install with: pip install openai>=1.0.0"
        ) from exc
    return OpenAI, APIError, APIStatusError, RateLimitError, APIConnectionError


class OpenAILLMClient:
    """OpenAI chat-completions adapter with the same interface as NVIDIALLMClient."""

    DEFAULT_TEMPERATURE = _parse_env_float("LLM_TEMPERATURE", 0.3)
    DEFAULT_MAX_TOKENS = _parse_env_int("LLM_MAX_TOKENS", 800)
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 1.0
    DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_logging: bool = True,
    ) -> None:
        self.enable_logging = enable_logging
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise OpenAILLMConfigError(
                "OpenAI API key not found. Provide api_key or set OPENAI_API_KEY."
            )

        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.default_temperature = _parse_env_float("LLM_TEMPERATURE", self.DEFAULT_TEMPERATURE)
        self.default_max_tokens = _parse_env_int("LLM_MAX_TOKENS", self.DEFAULT_MAX_TOKENS)
        self.last_usage_tokens: Optional[int] = None
        self.model: Optional[str] = None

        OpenAI, APIError, APIStatusError, RateLimitError, APIConnectionError = _load_openai()
        self.APIError = APIError
        self.APIStatusError = APIStatusError
        self.RateLimitError = RateLimitError
        self.APIConnectionError = APIConnectionError

        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        try:
            self.client = OpenAI(**kwargs)
        except Exception as exc:
            raise OpenAILLMConfigError(f"Failed to initialize OpenAI client: {exc}") from exc

        if self.enable_logging:
            logger.info("OpenAI LLM client initialized (base_url=%s)", self.base_url or "default")

    def _resolve_model(self, model: Optional[str]) -> str:
        if model and model.strip():
            return model.strip()
        return self.DEFAULT_MODEL

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs: Any,
    ) -> str:
        resolved_model = self._resolve_model(model)
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        self.model = resolved_model
        self.last_usage_tokens = None
        start = time.time()

        if self.enable_logging:
            logger.info(
                "OpenAI generate: model=%s temp=%s max_tokens=%s messages=%d",
                resolved_model,
                temperature,
                max_tokens,
                len(messages),
            )

        # Newer OpenAI chat models (gpt-5.x, o-series, etc.) reject the legacy
        # `max_tokens` parameter and require `max_completion_tokens`. Try the
        # modern name first, fall back to the legacy one if the model/endpoint
        # rejects it.
        call_kwargs = dict(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **kwargs,
        )
        try:
            response = self._retry_with_backoff(
                self.client.chat.completions.create, **call_kwargs
            )
        except Exception as exc:
            err_msg = str(exc).lower()
            if "max_completion_tokens" in err_msg and "unsupported" in err_msg:
                call_kwargs.pop("max_completion_tokens", None)
                call_kwargs["max_tokens"] = max_tokens
                try:
                    response = self._retry_with_backoff(
                        self.client.chat.completions.create, **call_kwargs
                    )
                except Exception as exc2:
                    exc = exc2
                else:
                    exc = None
            if exc is not None:
                if isinstance(exc, OpenAILLMError):
                    raise exc
                status = getattr(exc, "status_code", None) or getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                raise OpenAILLMAPIError(
                    f"OpenAI API request failed: {exc}", status_code=status
                ) from exc

        if not getattr(response, "choices", None):
            raise OpenAILLMAPIError("OpenAI API returned empty response")

        content = response.choices[0].message.content
        if isinstance(content, list):
            parts: list[str] = []
            for seg in content:
                if isinstance(seg, str):
                    parts.append(seg)
                elif isinstance(seg, dict):
                    text = seg.get("text") or seg.get("value")
                    if text:
                        parts.append(str(text))
            content = "".join(parts) if parts else None

        if not content:
            raise OpenAILLMAPIError("OpenAI API returned empty content")

        if hasattr(response, "usage") and response.usage is not None:
            self.last_usage_tokens = getattr(response.usage, "total_tokens", None)

        if self.enable_logging:
            elapsed_ms = (time.time() - start) * 1000
            logger.info(
                "OpenAI generated %d chars in %.0fms using %s",
                len(content),
                elapsed_ms,
                resolved_model,
            )
        return content

    def generate_simple(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.generate(
            messages=messages, model=model, temperature=temperature, **kwargs
        )

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                retryable_types = tuple(
                    t for t in (self.RateLimitError, self.APIConnectionError) if t is not None
                )
                is_retryable = bool(retryable_types) and isinstance(exc, retryable_types)

                if status is not None:
                    if status in (401, 403, 404):
                        raise
                    if status == 429 or (500 <= status < 600):
                        is_retryable = True

                if not is_retryable and status is None:
                    err = str(exc).lower()
                    is_retryable = ("timeout" in err) or ("timed out" in err) or ("connection" in err)

                if not is_retryable or attempt == self.MAX_RETRIES - 1:
                    raise

                delay = self.RETRY_DELAY_SECONDS * (2 ** attempt)
                if self.enable_logging:
                    logger.warning(
                        "OpenAI retry %d/%d after %.1fs: %s",
                        attempt + 1,
                        self.MAX_RETRIES,
                        delay,
                        exc,
                    )
                time.sleep(delay)

    def test_connection(self, model: str = None) -> Dict[str, Any]:
        resolved = self._resolve_model(model)
        start = time.time()
        try:
            self.generate_simple("Say 'ok' in one word.", model=resolved)
            return {
                "success": True,
                "model_tested": resolved,
                "response_time_ms": (time.time() - start) * 1000,
                "base_url": self.base_url,
                "status_code": None,
                "error": None,
            }
        except Exception as exc:
            return {
                "success": False,
                "model_tested": resolved,
                "response_time_ms": (time.time() - start) * 1000,
                "base_url": self.base_url,
                "status_code": getattr(exc, "status_code", None),
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }


__all__ = [
    "OpenAILLMClient",
    "OpenAILLMError",
    "OpenAILLMAPIError",
    "OpenAILLMConfigError",
]
