from __future__ import annotations

import pytest
import requests

from src.medical_retrieval.perplexity_search_client import (
    DEFAULT_SEARCH_DOMAINS,
    PerplexitySearchClient,
    PerplexitySearchError,
)


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse | None = None) -> None:
        self.response = response or FakeResponse({"results": []})
        self.posts: list[dict] = []

    def post(self, url: str, **kwargs):
        self.posts.append({"url": url, **kwargs})
        return self.response


def test_search_posts_default_pubmed_and_pmc_domain_allowlist() -> None:
    session = FakeSession()
    client = PerplexitySearchClient(api_key="test-key", session=session)

    client.search("aspirin bleeding risk")

    assert session.posts[0]["url"] == "https://api.perplexity.ai/search"
    assert session.posts[0]["headers"]["Authorization"] == "Bearer test-key"
    assert session.posts[0]["json"]["query"] == "aspirin bleeding risk"
    assert session.posts[0]["json"]["search_domain_filter"] == list(DEFAULT_SEARCH_DOMAINS)
    assert session.posts[0]["json"]["max_tokens_per_page"] == 512


def test_search_posts_multi_query_request_using_query_list_field() -> None:
    session = FakeSession()
    client = PerplexitySearchClient(api_key="test-key", session=session)

    client.search(["aspirin", "warfarin"])

    assert session.posts[0]["json"]["query"] == ["aspirin", "warfarin"]
    assert "queries" not in session.posts[0]["json"]


def test_search_rejects_more_than_five_queries() -> None:
    client = PerplexitySearchClient(api_key="test-key", session=FakeSession())

    with pytest.raises(ValueError, match="5 queries"):
        client.search(["q1", "q2", "q3", "q4", "q5", "q6"])


def test_search_rejects_more_than_twenty_domains() -> None:
    client = PerplexitySearchClient(api_key="test-key", session=FakeSession())

    domains = [f"example{i}.org" for i in range(21)]
    with pytest.raises(ValueError, match="20 domains"):
        client.search("query", domain_filter=domains)


def test_search_rejects_domain_filters_with_protocols() -> None:
    client = PerplexitySearchClient(api_key="test-key", session=FakeSession())

    with pytest.raises(ValueError, match="protocol"):
        client.search("query", domain_filter=["https://pubmed.ncbi.nlm.nih.gov"])


def test_search_maps_single_query_results_with_query_and_strategy_provenance() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "results": [
                    {
                        "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
                        "title": "Aspirin review",
                        "snippet": "Focused PubMed snippet",
                    },
                    {
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
                        "name": "PMC article",
                        "description": "Open access text",
                    },
                ]
            }
        )
    )
    client = PerplexitySearchClient(api_key="test-key", session=session)

    results = client.search("aspirin", strategy="broad-discovery")

    assert [result.rank for result in results] == [1, 2]
    assert results[0].source_url == "https://pubmed.ncbi.nlm.nih.gov/123/"
    assert results[0].title == "Aspirin review"
    assert results[0].snippet == "Focused PubMed snippet"
    assert results[0].query == "aspirin"
    assert results[0].strategy == "broad-discovery"
    assert results[1].title == "PMC article"
    assert results[1].snippet == "Open access text"


def test_search_maps_multi_query_nested_results_lists_by_input_query_index() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "results": [
                    [
                        {
                            "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
                            "title": "Aspirin",
                            "snippet": "first",
                        }
                    ],
                    [
                        {
                            "url": "https://pubmed.ncbi.nlm.nih.gov/2/",
                            "title": "Warfarin",
                            "snippet": "second",
                        }
                    ],
                ]
            }
        )
    )
    client = PerplexitySearchClient(api_key="test-key", session=session)

    results = client.search(["aspirin", "warfarin"])

    assert [(result.query, result.rank, result.title) for result in results] == [
        ("aspirin", 1, "Aspirin"),
        ("warfarin", 1, "Warfarin"),
    ]


def test_search_maps_multi_query_grouped_results_objects_by_group_query() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "results": [
                    {
                        "query": "aspirin",
                        "results": [
                            {
                                "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
                                "title": "Aspirin",
                                "snippet": "first",
                            }
                        ],
                    },
                    {
                        "query": "warfarin",
                        "results": [
                            {
                                "url": "https://pubmed.ncbi.nlm.nih.gov/2/",
                                "title": "Warfarin",
                                "snippet": "second",
                            }
                        ],
                    },
                ]
            }
        )
    )
    client = PerplexitySearchClient(api_key="test-key", session=session)

    results = client.search(["fallback aspirin", "fallback warfarin"])

    assert [(result.query, result.rank, result.title) for result in results] == [
        ("aspirin", 1, "Aspirin"),
        ("warfarin", 1, "Warfarin"),
    ]


def test_search_keeps_legacy_multi_query_response_results_with_query_provenance() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "queries": [
                    {
                        "query": "aspirin",
                        "results": [
                            {
                                "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
                                "title": "Aspirin",
                                "snippet": "first",
                            }
                        ],
                    },
                    {
                        "query": "warfarin",
                        "results": [
                            {
                                "url": "https://pubmed.ncbi.nlm.nih.gov/2/",
                                "title": "Warfarin",
                                "snippet": "second",
                            }
                        ],
                    },
                ]
            }
        )
    )
    client = PerplexitySearchClient(api_key="test-key", session=session)

    results = client.search(["aspirin", "warfarin"])

    assert [(result.query, result.rank, result.title) for result in results] == [
        ("aspirin", 1, "Aspirin"),
        ("warfarin", 1, "Warfarin"),
    ]


def test_search_wraps_operational_request_errors() -> None:
    class ErrorSession:
        def post(self, url: str, **kwargs):
            raise requests.Timeout("timed out")

    client = PerplexitySearchClient(api_key="test-key", session=ErrorSession())

    with pytest.raises(PerplexitySearchError, match="Perplexity search request failed"):
        client.search("aspirin")
