from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from src.medlineplus_client import MedlinePlusClient, expand_consumer_query


MEDLINEPLUS_XML = """
<nlmSearchResult>
  <term>acetaminophen ibuprofen</term>
  <count>1</count>
  <list>
    <document rank="0" url="https://medlineplus.gov/painrelievers.html">
      <content name="title"><span class="qt0">Pain</span> Relievers</content>
      <content name="organizationName">National Library of Medicine</content>
      <content name="FullSummary"><p>Acetaminophen and NSAIDs can help with pain and fever.</p></content>
      <content name="groupName">Drug Therapy</content>
      <content name="snippet">Pain relievers include acetaminophen and ibuprofen.</content>
    </document>
  </list>
</nlmSearchResult>
""".strip()


def test_parse_response_normalizes_medlineplus_documents() -> None:
    results = MedlinePlusClient.parse_response(MEDLINEPLUS_XML)

    assert len(results) == 1
    result = results[0]
    assert result.identifier == "medlineplus:painrelievers"
    assert result.title == "Pain Relievers"
    assert result.summary == "Acetaminophen and NSAIDs can help with pain and fever."
    assert result.url == "https://medlineplus.gov/painrelievers.html"
    assert result.sections["groupName"] == ["Drug Therapy"]
    assert result.sections["snippet"] == ["Pain relievers include acetaminophen and ibuprofen."]


def test_search_calls_medlineplus_web_service_with_expected_params() -> None:
    client = MedlinePlusClient(base_url="https://example.test/ws/query", email="team@example.com")
    response = Mock()
    response.text = MEDLINEPLUS_XML
    response.raise_for_status.return_value = None

    with patch("src.medlineplus_client.requests.get", return_value=response) as mock_get:
        results = client.search("Tylenol vs Advil side effects", max_results=3)

    assert len(results) == 1
    params = mock_get.call_args.kwargs["params"]
    assert mock_get.call_args.args[0] == "https://example.test/ws/query"
    assert params["db"] == "healthTopics"
    assert params["term"] == "Tylenol vs Advil side effects"
    assert params["retmax"] == 3
    assert params["rettype"] == "all"
    assert params["tool"] == "rag_template_medlineplus"
    assert params["email"] == "team@example.com"


def test_parse_response_rejects_invalid_xml() -> None:
    with pytest.raises(ValueError, match="MedlinePlus response is not valid XML"):
        MedlinePlusClient.parse_response("<not-closed")


def test_expand_consumer_query_adds_common_generic_terms() -> None:
    assert expand_consumer_query("Tylenol vs Advil side effects") == (
        "acetaminophen ibuprofen pain relievers side effects"
    )
