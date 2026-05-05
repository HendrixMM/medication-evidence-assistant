from __future__ import annotations

from typing import Any

import pytest

from src.medical_retrieval.pubmed_fetcher import (
    PubmedFetcher,
    PubmedFetcherError,
)


class FakePubmedClient:
    def __init__(self, records: dict[str, dict[str, Any]] | None = None, raise_on_fetch: Exception | None = None) -> None:
        self.records = records or {}
        self.raise_on_fetch = raise_on_fetch
        self.efetch_calls: list[list[str]] = []

    def efetch_xml(self, pmids: list[str]) -> str:
        if self.raise_on_fetch is not None:
            raise self.raise_on_fetch
        self.efetch_calls.append(list(pmids))
        return "<xml/>"

    def parse_efetch(self, xml_text: str) -> list[dict[str, Any]]:
        if not self.efetch_calls:
            return []
        last = self.efetch_calls[-1]
        return [self.records[p] for p in last if p in self.records]


class FakePMCMapper:
    def __init__(self, mappings: dict[str, str]) -> None:
        self.mappings = mappings
        self.calls: list[list[str]] = []

    def pmcids_to_pmids(self, pmcids: list[str]) -> dict[str, str]:
        self.calls.append(list(pmcids))
        return {p: self.mappings[p] for p in pmcids if p in self.mappings}


def _record(pmid: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "pmid": pmid,
        "title": f"Title {pmid}",
        "abstract": f"Abstract {pmid}",
        "authors": "Doe, Jane, Roe, John",
        "journal": "Journal X",
        "publication_date": "2023-01-01",
        "publication_types": ["Journal Article"],
        "doi": f"10.0/{pmid}",
    }
    base.update(overrides)
    return base


def test_normalize_and_fetch_returns_canonical_pubmed_records_for_pubmed_urls() -> None:
    client = FakePubmedClient({"123": _record("123")})
    fetcher = PubmedFetcher(pubmed_client=client)

    out = fetcher.normalize_and_fetch({
        "candidates": [{
            "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
            "snippet": "snippet",
            "query": "aspirin warfarin",
            "strategy_family": "interaction",
        }],
    })

    assert len(out["normalized"]) == 1
    record = out["normalized"][0]
    assert record["pmid"] == "123"
    assert record["url"] == "https://pubmed.ncbi.nlm.nih.gov/123/"
    assert record["title"] == "Title 123"
    assert record["authors"] == ["Doe", "Jane", "Roe", "John"]
    assert record["provenance"][0]["strategy_family"] == "interaction"
    assert out["fetch_count"] == 1


def test_normalize_and_fetch_maps_pmcid_via_official_converter_then_fetches_pubmed() -> None:
    client = FakePubmedClient({"999": _record("999")})
    mapper = FakePMCMapper({"PMC555": "999"})
    fetcher = PubmedFetcher(pubmed_client=client, pmc_mapper=mapper)

    out = fetcher.normalize_and_fetch({
        "candidates": [{
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC555/",
            "snippet": "pmc snippet",
            "query": "ozempic adverse",
            "strategy_family": "mechanism_outcome",
        }],
    })

    assert mapper.calls == [["PMC555"]]
    assert len(out["normalized"]) == 1
    assert out["normalized"][0]["pmid"] == "999"
    assert out["normalized"][0]["pmcid"] == "PMC555"


def test_normalize_and_fetch_drops_unmapped_pmcids_without_llm_inference() -> None:
    client = FakePubmedClient({})
    mapper = FakePMCMapper({})
    fetcher = PubmedFetcher(pubmed_client=client, pmc_mapper=mapper)

    out = fetcher.normalize_and_fetch({
        "candidates": [{
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC404/",
            "snippet": "x",
            "query": "q",
            "strategy_family": "literal_med",
        }],
    })

    assert out["normalized"] == []
    assert out["unmapped"] == [{
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC404/",
        "reason": "pmcid_unmapped",
        "pmcid": "PMC404",
    }]
    assert client.efetch_calls == []


def test_normalize_and_fetch_drops_unsupported_urls() -> None:
    client = FakePubmedClient({})
    fetcher = PubmedFetcher(pubmed_client=client)

    out = fetcher.normalize_and_fetch({
        "candidates": [
            {"url": "https://example.com/article", "snippet": "", "query": "q", "strategy_family": "x"},
            {"url": "", "snippet": "", "query": "q", "strategy_family": "x"},
        ],
    })

    assert out["normalized"] == []
    assert {u["reason"] for u in out["unmapped"]} == {"unsupported_url", "missing_url"}


def test_normalize_and_fetch_dedupes_pmid_and_merges_provenance() -> None:
    client = FakePubmedClient({"42": _record("42")})
    fetcher = PubmedFetcher(pubmed_client=client)

    out = fetcher.normalize_and_fetch({
        "candidates": [
            {
                "url": "https://pubmed.ncbi.nlm.nih.gov/42/",
                "snippet": "s1",
                "query": "q1",
                "strategy_family": "literal_med",
            },
            {
                "url": "https://pubmed.ncbi.nlm.nih.gov/42/",
                "snippet": "s2",
                "query": "q2",
                "strategy_family": "ingredient_or_class",
            },
        ],
    })

    assert len(out["normalized"]) == 1
    provenance = out["normalized"][0]["provenance"]
    assert len(provenance) == 2
    assert {p["strategy_family"] for p in provenance} == {"literal_med", "ingredient_or_class"}
    assert client.efetch_calls == [["42"]]


def test_normalize_and_fetch_enforces_candidate_cap() -> None:
    client = FakePubmedClient({"1": _record("1"), "2": _record("2")})
    fetcher = PubmedFetcher(pubmed_client=client, max_candidates=1)

    out = fetcher.normalize_and_fetch({
        "candidates": [
            {"url": "https://pubmed.ncbi.nlm.nih.gov/1/", "snippet": "", "query": "q", "strategy_family": "f"},
            {"url": "https://pubmed.ncbi.nlm.nih.gov/2/", "snippet": "", "query": "q", "strategy_family": "f"},
        ],
    })

    assert len(out["normalized"]) == 1
    assert out["normalized"][0]["pmid"] == "1"
    assert any(u["reason"] == "candidate_cap_reached" for u in out["unmapped"])


def test_normalize_and_fetch_raises_expected_error_when_pubmed_fetch_fails() -> None:
    client = FakePubmedClient({}, raise_on_fetch=RuntimeError("network down"))
    fetcher = PubmedFetcher(pubmed_client=client)

    with pytest.raises(PubmedFetcherError, match="canonical fetch failed"):
        fetcher.normalize_and_fetch({
            "candidates": [{
                "url": "https://pubmed.ncbi.nlm.nih.gov/77/",
                "snippet": "",
                "query": "q",
                "strategy_family": "f",
            }],
        })


def test_normalize_and_fetch_records_pubmed_fetch_misses_in_unmapped() -> None:
    client = FakePubmedClient({})
    fetcher = PubmedFetcher(pubmed_client=client)

    out = fetcher.normalize_and_fetch({
        "candidates": [{
            "url": "https://pubmed.ncbi.nlm.nih.gov/55/",
            "snippet": "s",
            "query": "q",
            "strategy_family": "f",
        }],
    })

    assert out["normalized"] == []
    assert out["unmapped"] == [{
        "url": "https://pubmed.ncbi.nlm.nih.gov/55/",
        "reason": "pubmed_fetch_missing",
        "pmid": "55",
    }]


def test_normalize_and_fetch_rejects_non_list_candidates() -> None:
    fetcher = PubmedFetcher(pubmed_client=FakePubmedClient({}))

    with pytest.raises(PubmedFetcherError, match="must be a list"):
        fetcher.normalize_and_fetch({"candidates": "not-a-list"})


def test_normalize_and_fetch_pass_index_recorded_in_provenance() -> None:
    client = FakePubmedClient({"7": _record("7")})
    fetcher = PubmedFetcher(pubmed_client=client)

    out = fetcher.normalize_and_fetch(
        {"candidates": [{
            "url": "https://pubmed.ncbi.nlm.nih.gov/7/",
            "snippet": "s",
            "query": "q",
            "strategy_family": "f",
        }]},
        pass_index=2,
    )

    assert out["normalized"][0]["provenance"][0]["pass_index"] == 2


def test_pubmed_fetcher_rejects_invalid_max_candidates() -> None:
    with pytest.raises(ValueError, match="max_candidates"):
        PubmedFetcher(pubmed_client=FakePubmedClient({}), max_candidates=0)
