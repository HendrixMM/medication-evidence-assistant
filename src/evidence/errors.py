from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from .schemas import ErrorRecord

T = TypeVar("T")


def capture_expected_failure(layer: str, operation: Callable[[], T], fallback: T) -> tuple[T, ErrorRecord | None]:
    try:
        return operation(), None
    except (ConnectionError, TimeoutError, ValueError, OSError, RuntimeError) as exc:
        return fallback, ErrorRecord(layer=layer, reason=str(exc))
