from __future__ import annotations

import json
from typing import Any

import pytest

from src.medical_retrieval.ranker import (
    DEFAULT_RANKER_MODEL,
    LLMReranker,
    RerankerError,
)


class FakeLLMClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(self, *, messages, model, temperature, max_tokens) -> str:
        self.calls.append({
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, str):
            return response
        return json.dumps(response)


def _candidate(pmid: str, *, abstract: str = "abs", strategy: str = "literal_med") -> dict[str, Any]:
    return {
        "pmid": pmid,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "title": f"Title {pmid}",
        "abstract": abstract,
        "journal": "JAMA",
        "publication_date": "2023-01-01",
        "publication_types": ["Journal Article"],
        "authors": ["Doe, Jane"],
        "provenance": [{"strategy_family": strategy, "query": "q"}],
        "full_text": "should not be sent",
    }


def _score(pmid: str, *, include: bool = True, base: float = 0.8) -> dict[str, Any]:
    return {
        "pmid": pmid,
        "relevance": base,
        "directness": base,
        "design_strength": base,
        "population_match": base,
        "medication_match": base,
        "outcome_match": base,
        "include": include,
        "reason": f"score for {pmid}",
    }


def test_rank_batches_candidates_and_compact_payload_omits_full_text() -> None:
    candidates = [_candidate(str(i)) for i in range(1, 6)]
    batch1_response = {"scores": [_score(str(i)) for i in range(1, 4)]}
    batch2_response = {"scores": [_score(str(i)) for i in range(4, 6)]}
    arbitration_response = {
        "selected": [
            {"pmid": "1", "rank": 1, "reason": "best"},
            {"pmid": "2", "rank": 2, "reason": "next"},
        ],
        "thin_evidence": False,
        "rationale": "good coverage",
    }

    client = FakeLLMClient([batch1_response, batch2_response, arbitration_response])
    reranker = LLMReranker(llm_client=client, batch_size=3, top_n=2)

    result = reranker.rank("Does aspirin interact with warfarin?", {"question_type": "interaction"}, candidates)

    assert [r.pmid for r in result.ranked] == ["1", "2"]
    assert result.ranked[0].rank == 1
    assert result.ranked[0].include is True
    assert result.thin_evidence is False
    assert len(client.calls) == 3

    first_user_payload = json.loads(client.calls[0]["messages"][1]["content"])
    sent_candidate = first_user_payload["candidates"][0]
    assert "full_text" not in sent_candidate
    assert sent_candidate["pmid"] == "1"
    assert "abstract" in sent_candidate


def test_rank_uses_default_model_gpt_5_4_mini() -> None:
    candidates = [_candidate("1")]
    client = FakeLLMClient([
        {"scores": [_score("1")]},
        {"selected": [{"pmid": "1", "rank": 1, "reason": "ok"}], "thin_evidence": False, "rationale": "ok"},
    ])
    LLMReranker(llm_client=client).rank("q", {}, candidates)

    assert client.calls[0]["model"] == DEFAULT_RANKER_MODEL == "gpt-5.4-mini"


def test_rank_returns_thin_evidence_when_no_candidates() -> None:
    reranker = LLMReranker(llm_client=FakeLLMClient([]))

    result = reranker.rank("q", {}, [])

    assert result.ranked == []
    assert result.thin_evidence is True


def test_rank_marks_thin_evidence_when_arbitration_says_so() -> None:
    candidates = [_candidate("1")]
    client = FakeLLMClient([
        {"scores": [_score("1", include=False, base=0.2)]},
        {"selected": [], "thin_evidence": True, "rationale": "weak coverage"},
    ])
    result = LLMReranker(llm_client=client).rank("q", {}, candidates)

    assert result.ranked == []
    assert result.thin_evidence is True
    assert "weak" in result.rationale


def test_rank_caps_at_top_n_even_when_arbitration_returns_more() -> None:
    candidates = [_candidate(str(i)) for i in range(1, 4)]
    client = FakeLLMClient([
        {"scores": [_score(str(i)) for i in range(1, 4)]},
        {
            "selected": [
                {"pmid": "1", "rank": 1, "reason": "a"},
                {"pmid": "2", "rank": 2, "reason": "b"},
                {"pmid": "3", "rank": 3, "reason": "c"},
            ],
            "thin_evidence": False,
            "rationale": "ok",
        },
    ])
    result = LLMReranker(llm_client=client, top_n=2).rank("q", {}, candidates)

    assert [r.pmid for r in result.ranked] == ["1", "2"]


def test_rank_drops_arbitration_pmids_not_in_scored_set() -> None:
    candidates = [_candidate("1")]
    client = FakeLLMClient([
        {"scores": [_score("1")]},
        {
            "selected": [
                {"pmid": "999", "rank": 1, "reason": "hallucinated"},
                {"pmid": "1", "rank": 2, "reason": "real"},
            ],
            "thin_evidence": False,
            "rationale": "ok",
        },
    ])
    result = LLMReranker(llm_client=client).rank("q", {}, candidates)

    assert [r.pmid for r in result.ranked] == ["1"]
    assert result.ranked[0].arbitration_reason == "real"


def test_rank_raises_expected_error_when_llm_response_is_not_json() -> None:
    candidates = [_candidate("1")]
    client = FakeLLMClient(["not-json"])
    reranker = LLMReranker(llm_client=client)

    with pytest.raises(RerankerError, match="not valid JSON"):
        reranker.rank("q", {}, candidates)


def test_rank_raises_expected_error_when_batch_response_fails_validation() -> None:
    candidates = [_candidate("1")]
    client = FakeLLMClient([{"scores": [{"pmid": "1"}]}])
    reranker = LLMReranker(llm_client=client)

    with pytest.raises(RerankerError, match="batch scoring response failed validation"):
        reranker.rank("q", {}, candidates)


def test_rank_raises_expected_error_when_arbitration_response_fails_validation() -> None:
    candidates = [_candidate("1")]
    client = FakeLLMClient([
        {"scores": [_score("1")]},
        {"selected": [{"pmid": "1", "rank": 1, "reason": "x"}], "thin_evidence": "not-bool", "rationale": "x"},
    ])
    reranker = LLMReranker(llm_client=client)

    with pytest.raises(RerankerError, match="arbitration response failed validation"):
        reranker.rank("q", {}, candidates)


def test_rank_wraps_expected_llm_transport_errors() -> None:
    candidates = [_candidate("1")]
    client = FakeLLMClient([TimeoutError("slow")])
    reranker = LLMReranker(llm_client=client)

    with pytest.raises(RerankerError, match="Reranker LLM request failed"):
        reranker.rank("q", {}, candidates)


def test_rank_propagates_unexpected_programming_errors() -> None:
    candidates = [_candidate("1")]
    client = FakeLLMClient([AttributeError("bug")])
    reranker = LLMReranker(llm_client=client)

    with pytest.raises(AttributeError, match="bug"):
        reranker.rank("q", {}, candidates)


def test_rank_raises_when_no_candidate_has_pmid() -> None:
    reranker = LLMReranker(llm_client=FakeLLMClient([]))

    with pytest.raises(RerankerError, match="must include pmid"):
        reranker.rank("q", {}, [{"url": "x"}])


def test_reranker_rejects_invalid_construction_args() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        LLMReranker(llm_client=FakeLLMClient([]), batch_size=0)
    with pytest.raises(ValueError, match="top_n"):
        LLMReranker(llm_client=FakeLLMClient([]), top_n=0)


def test_compact_payload_includes_only_documented_fields() -> None:
    candidates = [_candidate("1")]
    client = FakeLLMClient([
        {"scores": [_score("1")]},
        {"selected": [{"pmid": "1", "rank": 1, "reason": "ok"}], "thin_evidence": False, "rationale": "ok"},
    ])
    LLMReranker(llm_client=client).rank("q", {}, candidates)

    sent = json.loads(client.calls[0]["messages"][1]["content"])["candidates"][0]
    assert set(sent.keys()) == {
        "pmid", "url", "title", "abstract", "journal",
        "publication_date", "publication_types", "authors", "provenance",
    }
