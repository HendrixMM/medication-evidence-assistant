from __future__ import annotations

import pytest
import requests

from src.medical_retrieval.pmc_mapper import PMCIDMapper
from src.medical_retrieval.source_normalizer import normalize_source_url


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
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.gets: list[dict] = []

    def get(self, url: str, **kwargs):
        self.gets.append({"url": url, **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_normalizes_pubmed_url_variants_to_pmid() -> None:
    urls = [
        "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        "https://pubmed.ncbi.nlm.nih.gov/12345678",
        "https://pubmed.ncbi.nlm.nih.gov/12345678/?from_term=aspirin",
        "https://pubmed.ncbi.nlm.nih.gov/12345678/#abstract",
    ]

    for url in urls:
        normalized = normalize_source_url(url)
        assert normalized.is_supported is True
        assert normalized.pmid == "12345678"
        assert normalized.pmcid is None
        assert normalized.dedupe_key == "pmid:12345678"


def test_normalizes_pubmed_default_port_url_using_hostname() -> None:
    normalized = normalize_source_url("https://pubmed.ncbi.nlm.nih.gov:443/12345678/")

    assert normalized.is_supported is True
    assert normalized.pmid == "12345678"
    assert normalized.dedupe_key == "pmid:12345678"


def test_normalizes_current_and_legacy_pmc_url_forms_to_uppercase_pmcid() -> None:
    urls = [
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567",
        "https://pmc.ncbi.nlm.nih.gov/articles/pmc1234567/",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/?from=search",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/#abstract",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/",
    ]

    for url in urls:
        normalized = normalize_source_url(url)
        assert normalized.is_supported is True
        assert normalized.pmid is None
        assert normalized.pmcid == "PMC1234567"
        assert normalized.dedupe_key == "pmcid:PMC1234567"


def test_current_pmc_url_requires_articles_path() -> None:
    normalized = normalize_source_url("https://pmc.ncbi.nlm.nih.gov/foo/PMC1234567/")

    assert normalized.is_supported is False
    assert normalized.pmcid is None
    assert normalized.dedupe_key is None


def test_current_pmc_url_rejects_extra_path_segments_after_pmcid() -> None:
    normalized = normalize_source_url(
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/extra"
    )

    assert normalized.is_supported is False
    assert normalized.pmcid is None
    assert normalized.dedupe_key is None


def test_current_pmc_url_rejects_empty_internal_or_extra_path_segments() -> None:
    urls = [
        "https://pmc.ncbi.nlm.nih.gov//articles/PMC1234567",
        "https://pmc.ncbi.nlm.nih.gov/articles//PMC1234567",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567//",
    ]

    for url in urls:
        normalized = normalize_source_url(url)
        assert normalized.is_supported is False
        assert normalized.pmcid is None
        assert normalized.dedupe_key is None


def test_legacy_www_ncbi_pmc_url_requires_pmc_articles_path() -> None:
    normalized = normalize_source_url("https://www.ncbi.nlm.nih.gov/foo/PMC1234567/")

    assert normalized.is_supported is False
    assert normalized.pmcid is None
    assert normalized.dedupe_key is None


def test_unsupported_urls_have_no_identifier_or_dedupe_key() -> None:
    normalized = normalize_source_url("https://example.org/article/123")

    assert normalized.is_supported is False
    assert normalized.pmid is None
    assert normalized.pmcid is None
    assert normalized.dedupe_key is None


def test_dedupe_key_prefers_pmid_over_pmcid() -> None:
    normalized = normalize_source_url("https://pubmed.ncbi.nlm.nih.gov/12345678/")
    normalized.pmcid = "PMC1234567"

    assert normalized.dedupe_key == "pmid:12345678"


def test_pmcid_mapper_calls_nih_pmc_id_converter_endpoint() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "status": "ok",
                "records": [
                    {"pmcid": "PMC123", "pmid": "111"},
                ]
            }
        )
    )
    mapper = PMCIDMapper(session=session)

    assert mapper.pmcid_to_pmid("pmc123") == "111"
    assert "/tools/idconv/api/v1/articles/" in session.gets[0]["url"]
    assert session.gets[0]["params"]["ids"] == "PMC123"
    assert session.gets[0]["params"]["idtype"] == "pmcid"
    assert session.gets[0]["params"]["format"] == "json"


def test_pmcid_mapper_returns_batch_mapping_from_mocked_converter_response() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "status": "ok",
                "records": [
                    {"pmcid": "PMC123", "pmid": "111"},
                    {"pmcid": "PMC456", "pmid": "222"},
                    {"pmcid": "PMC789", "status": "error", "errmsg": "not found"},
                ],
            }
        )
    )
    mapper = PMCIDMapper(session=session)

    assert mapper.pmcids_to_pmids(["PMC123", "pmc456"]) == {
        "PMC123": "111",
        "PMC456": "222",
    }
    assert session.gets[0]["params"]["ids"] == "PMC123,PMC456"


def test_single_pmcid_mapper_delegates_to_batch_mapping() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "status": "ok",
                "records": [
                    {"pmcid": "PMC123", "pmid": "111"},
                ],
            }
        )
    )
    mapper = PMCIDMapper(session=session)

    assert mapper.pmcid_to_pmid("PMC123") == "111"


def test_pmcid_mapper_returns_none_for_unmapped_or_error_records_without_inference() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "status": "ok",
                "records": [
                    {"pmcid": "PMC123"},
                    {"pmcid": "PMC456", "status": "error", "errmsg": "not found"},
                ],
            }
        )
    )
    mapper = PMCIDMapper(session=session)

    assert mapper.pmcids_to_pmids(["PMC123", "PMC456"]) == {}
    assert mapper.pmcid_to_pmid("PMC456") is None


def test_pmcid_mapper_returns_empty_mapping_for_expected_operational_failure() -> None:
    session = FakeSession(requests.Timeout("timed out"))
    mapper = PMCIDMapper(session=session)

    assert mapper.pmcids_to_pmids(["PMC123"]) == {}
    assert mapper.pmcid_to_pmid("PMC123") is None


def test_pmcid_mapper_rejects_invalid_pmcid_strings() -> None:
    mapper = PMCIDMapper(session=FakeSession(FakeResponse({"status": "ok", "records": []})))

    with pytest.raises(ValueError):
        mapper.pmcid_to_pmid("not-a-pmcid")

    with pytest.raises(ValueError):
        mapper.pmcid_to_pmid("123")

    with pytest.raises(ValueError):
        mapper.pmcids_to_pmids(["PMC123", "bad"])

    with pytest.raises(ValueError):
        mapper.pmcids_to_pmids(["123"])
