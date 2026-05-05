from __future__ import annotations

import os
import random
import threading
import time
from typing import Any

from src.pubmed_eutils_client import PubMedEutilsClient


class TokenBucket:
    def __init__(self, rate_per_second: float, capacity: int) -> None:
        self.rate_per_second = rate_per_second
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_seconds = max(0.01, (1.0 - self._tokens) / self.rate_per_second)
            time.sleep(wait_seconds)


class RateLimitedPubMedClient:
    def __init__(self, client: PubMedEutilsClient | None = None) -> None:
        self.client = client or PubMedEutilsClient()
        rate = 10.0 if (os.getenv("PUBMED_EUTILS_API_KEY") or os.getenv("PUBMED_API_KEY")) else 3.0
        self.bucket = TokenBucket(rate_per_second=rate, capacity=max(1, int(rate)))

    def esearch(self, query: str, retmax: int = 30) -> list[str]:
        return self._call_with_backoff(self.client.esearch, query, retmax=retmax)

    def efetch_xml(self, pmids: list[str]) -> str:
        return self._call_with_backoff(self.client.efetch_xml, pmids)

    def elink_pmc(self, pmid: str) -> str | None:
        return self._call_with_backoff(self.client.elink_pmc, pmid)

    def parse_efetch(self, xml_text: str) -> list[dict[str, Any]]:
        return self.client.parse_efetch(xml_text)

    def search_and_fetch(self, query: str, max_items: int = 30) -> list[dict[str, Any]]:
        pmids = self.esearch(query, retmax=max_items)
        if not pmids:
            return []
        return self.parse_efetch(self.efetch_xml(pmids))

    def _call_with_backoff(self, method, *args, **kwargs):
        delay = 0.25
        last_error = None
        for _attempt in range(3):
            self.bucket.acquire()
            try:
                return method(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                if status_code not in (429, 503) and status_code is not None:
                    raise
                time.sleep(delay + random.uniform(0.0, 0.1))
                delay *= 2
        if last_error is not None:
            raise last_error
        raise RuntimeError("pubmed call failed")
