from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass

from src.medical_retrieval.schemas import MedicalEvidenceRequest, MedicalEvidenceResponse

DEFAULT_CACHE_TTL_SECONDS = 600
DEFAULT_CACHE_MAX_ENTRIES = 256


@dataclass
class _CacheEntry:
    response: MedicalEvidenceResponse
    expires_at: float


class MedicalRetrievalCache:
    """In-memory request cache for medication-centric retrieval responses."""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    def key_for(self, request: MedicalEvidenceRequest) -> str:
        canonical = json.dumps(
            {
                "question": request.question.strip().lower(),
                "hints": request.hints.model_dump(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> MedicalEvidenceResponse | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at < time.monotonic():
                self._entries.pop(key, None)
                return None
            return entry.response.model_copy(deep=True)

    def set(self, key: str, response: MedicalEvidenceResponse) -> None:
        with self._lock:
            if len(self._entries) >= self.max_entries and key not in self._entries:
                oldest_key = min(self._entries, key=lambda k: self._entries[k].expires_at)
                self._entries.pop(oldest_key, None)
            self._entries[key] = _CacheEntry(
                response=response.model_copy(deep=True),
                expires_at=time.monotonic() + self.ttl_seconds,
            )


__all__ = ["DEFAULT_CACHE_TTL_SECONDS", "MedicalRetrievalCache"]
