from __future__ import annotations

import hashlib
import time
from copy import deepcopy

try:
    from cachetools import TTLCache
except Exception:  # pragma: no cover - fallback when dependency is not installed
    TTLCache = None  # type: ignore[assignment]


class ResponseCache:
    def __init__(self, maxsize: int = 512, ttl_seconds: int = 86400) -> None:
        self._ttl_seconds = ttl_seconds
        if TTLCache is not None:
            self._cache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
            self._timestamps: dict[str, float] = {}
        else:  # pragma: no cover
            self._cache = {}
            self._timestamps = {}

    def make_key(self, question: str, intent: str, drug_names: list[str]) -> str:
        normalized_question = " ".join(question.lower().split())
        normalized_drugs = ",".join(sorted(name.strip().lower() for name in drug_names if name.strip()))
        payload = f"{normalized_question}|{intent}|{normalized_drugs}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str):
        if key not in self._cache:
            return None
        if TTLCache is None:  # pragma: no cover
            created = self._timestamps.get(key, 0.0)
            if time.time() - created > self._ttl_seconds:
                self._cache.pop(key, None)
                self._timestamps.pop(key, None)
                return None
        return deepcopy(self._cache[key])

    def set(self, key: str, value) -> None:
        self._cache[key] = deepcopy(value)
        self._timestamps[key] = time.time()
