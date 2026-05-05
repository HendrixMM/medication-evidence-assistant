from __future__ import annotations

import pytest

from src.medical_retrieval.cache import MedicalRetrievalCache
from src.medical_retrieval.errors import (
    LAYER_AGENT_DISCOVERY,
    LAYER_PUBMED_FETCH,
    LAYER_QUESTION_ANALYSIS,
    LAYER_RANKING,
    PerplexityAgentError,
    PubmedFetcherError,
    QuestionAnalysisError,
    RerankerError,
    layer_for_exception,
)
from src.medical_retrieval.metrics import RetrievalMetrics
from src.medical_retrieval.schemas import MedicalEvidenceRequest, MedicalRetrievalHints


def test_layer_for_exception_classifies_known_layer_errors() -> None:
    assert layer_for_exception(QuestionAnalysisError("x")) == LAYER_QUESTION_ANALYSIS
    assert layer_for_exception(PerplexityAgentError("x")) == LAYER_AGENT_DISCOVERY
    assert layer_for_exception(PubmedFetcherError("x")) == LAYER_PUBMED_FETCH
    assert layer_for_exception(RerankerError("x")) == LAYER_RANKING
    assert layer_for_exception(RuntimeError("?")) == "orchestration"


def test_metrics_to_meta_serializes_errors_and_strength_hint() -> None:
    metrics = RetrievalMetrics(agent_passes=2, perplexity_searches=4, pubmed_fetches=10, candidate_articles=20, ranked_articles=5)
    metrics.add_error("ranking", "ranker boom")

    meta = metrics.to_meta("medium")
    assert meta.agent_passes == 2
    assert meta.evidence_strength_hint == "medium"
    assert meta.errors[0].layer == "ranking"
    assert meta.errors[0].reason == "ranker boom"


def test_cache_key_is_stable_for_equivalent_requests() -> None:
    cache = MedicalRetrievalCache()
    a = MedicalEvidenceRequest(question="Aspirin?", hints=MedicalRetrievalHints(max_passes=3))
    b = MedicalEvidenceRequest(question="aspirin?", hints=MedicalRetrievalHints(max_passes=3))

    assert cache.key_for(a) == cache.key_for(b)


def test_cache_key_changes_when_hints_differ() -> None:
    cache = MedicalRetrievalCache()
    a = MedicalEvidenceRequest(question="aspirin?", hints=MedicalRetrievalHints(max_passes=3))
    b = MedicalEvidenceRequest(question="aspirin?", hints=MedicalRetrievalHints(max_passes=4))

    assert cache.key_for(a) != cache.key_for(b)


def test_cache_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="ttl_seconds"):
        MedicalRetrievalCache(ttl_seconds=0)
    with pytest.raises(ValueError, match="max_entries"):
        MedicalRetrievalCache(max_entries=0)


def test_cache_evicts_oldest_when_full() -> None:
    cache = MedicalRetrievalCache(max_entries=2, ttl_seconds=60)
    from src.medical_retrieval.schemas import (
        MedicalEvidenceResponse,
        MedicalQuestionAnalysis,
        MedicalQuestionEntities,
        MedicalRetrievalMeta,
    )

    def _resp() -> MedicalEvidenceResponse:
        return MedicalEvidenceResponse(
            question_analysis=MedicalQuestionAnalysis(
                question_type="x", entities=MedicalQuestionEntities(), optional_drug_normalizations=[]
            ),
            strategies_executed=[],
            articles=[],
            meta=MedicalRetrievalMeta(
                agent_passes=0, perplexity_searches=0, pubmed_fetches=0,
                candidate_articles=0, ranked_articles=0, evidence_strength_hint="unknown",
                cached=False, errors=[],
            ),
        )

    cache.set("a", _resp())
    cache.set("b", _resp())
    cache.set("c", _resp())

    # one of the original entries must be evicted
    present = sum(1 for k in ("a", "b", "c") if cache.get(k) is not None)
    assert present == 2
