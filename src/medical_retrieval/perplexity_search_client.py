from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import requests


DEFAULT_BASE_URL = "https://api.perplexity.ai"
DEFAULT_SEARCH_DOMAINS = ("pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov")
DEFAULT_MAX_TOKENS_PER_PAGE = 512
MAX_QUERIES_PER_REQUEST = 5
MAX_DOMAINS_PER_REQUEST = 20


class PerplexitySearchError(RuntimeError):
    """Expected operational failure from Perplexity search transport."""


@dataclass(frozen=True)
class PerplexitySearchResult:
    source_url: str
    title: str | None
    snippet: str | None
    rank: int
    query: str
    strategy: str | None = None


class PerplexitySearchClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        session: requests.Session | None = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    def search(
        self,
        query: str | list[str],
        *,
        domain_filter: list[str] | tuple[str, ...] | None = None,
        strategy: str | None = None,
        max_tokens_per_page: int = DEFAULT_MAX_TOKENS_PER_PAGE,
    ) -> list[PerplexitySearchResult]:
        queries = self._normalize_queries(query)
        domains = self._validate_domain_filter(domain_filter or DEFAULT_SEARCH_DOMAINS)
        payload: dict[str, Any] = {
            "search_domain_filter": domains,
            "max_tokens_per_page": max_tokens_per_page,
        }
        if len(queries) == 1:
            payload["query"] = queries[0]
        else:
            payload["query"] = queries

        try:
            response = self.session.post(
                f"{self.base_url}/search",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise PerplexitySearchError("Perplexity search request failed") from exc
        except ValueError as exc:
            raise PerplexitySearchError("Perplexity search response was not valid JSON") from exc

        return self._map_results(data, queries, strategy)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _normalize_queries(self, query: str | list[str]) -> list[str]:
        if isinstance(query, str):
            queries = [query]
        else:
            queries = list(query)
        if not queries:
            raise ValueError("At least one query is required")
        if len(queries) > MAX_QUERIES_PER_REQUEST:
            raise ValueError("Perplexity Search API accepts at most 5 queries per request")
        if any(not isinstance(item, str) or not item.strip() for item in queries):
            raise ValueError("Queries must be non-empty strings")
        return [item.strip() for item in queries]

    def _validate_domain_filter(self, domain_filter: list[str] | tuple[str, ...]) -> list[str]:
        domains = list(domain_filter)
        if len(domains) > MAX_DOMAINS_PER_REQUEST:
            raise ValueError("Perplexity Search API accepts at most 20 domains per request")
        if not domains:
            raise ValueError("At least one search domain is required")

        denylist = [domain for domain in domains if domain.startswith("-")]
        allowlist = [domain for domain in domains if not domain.startswith("-")]
        if denylist and allowlist:
            raise ValueError("Do not mix Perplexity domain allowlist and denylist modes")

        for domain in domains:
            bare_domain = domain[1:] if domain.startswith("-") else domain
            if "://" in bare_domain:
                raise ValueError("Perplexity search domains must not include a protocol")
            if not bare_domain.strip():
                raise ValueError("Perplexity search domains must be non-empty")
        return domains

    def _map_results(
        self,
        data: dict[str, Any],
        queries: list[str],
        strategy: str | None,
    ) -> list[PerplexitySearchResult]:
        grouped_results = self._extract_grouped_results(data, queries)
        mapped: list[PerplexitySearchResult] = []
        for query, raw_results in grouped_results:
            for index, raw_result in enumerate(raw_results, start=1):
                url = raw_result.get("url") or raw_result.get("source_url")
                if not url:
                    continue
                mapped.append(
                    PerplexitySearchResult(
                        source_url=str(url),
                        title=self._optional_str(raw_result.get("title") or raw_result.get("name")),
                        snippet=self._optional_str(
                            raw_result.get("snippet")
                            or raw_result.get("description")
                            or raw_result.get("text")
                        ),
                        rank=index,
                        query=str(raw_result.get("query") or query),
                        strategy=str(raw_result.get("strategy") or strategy) if raw_result.get("strategy") or strategy else None,
                    )
                )
        return mapped

    def _extract_grouped_results(
        self,
        data: dict[str, Any],
        queries: list[str],
    ) -> list[tuple[str, list[dict[str, Any]]]]:
        if isinstance(data.get("queries"), list):
            groups: list[tuple[str, list[dict[str, Any]]]] = []
            for index, item in enumerate(data["queries"]):
                if not isinstance(item, dict):
                    continue
                query = str(item.get("query") or queries[min(index, len(queries) - 1)])
                results = item.get("results") or []
                groups.append((query, [r for r in results if isinstance(r, dict)]))
            return groups

        results = data.get("results") or []
        if self._is_grouped_results(results):
            groups = []
            for index, item in enumerate(results):
                query = queries[min(index, len(queries) - 1)]
                if isinstance(item, list):
                    groups.append((query, [r for r in item if isinstance(r, dict)]))
                    continue
                if isinstance(item, dict):
                    group_query = str(item.get("query") or query)
                    group_results = item.get("results") or []
                    groups.append((group_query, [r for r in group_results if isinstance(r, dict)]))
            return groups

        return [(queries[0], [r for r in results if isinstance(r, dict)])]

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    def _is_grouped_results(self, results: Any) -> bool:
        if not isinstance(results, list) or not results:
            return False
        first = results[0]
        return isinstance(first, list) or (
            isinstance(first, dict) and isinstance(first.get("results"), list)
        )
