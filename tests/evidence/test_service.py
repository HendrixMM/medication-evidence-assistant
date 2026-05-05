from __future__ import annotations

import pytest

from src.evidence.schemas import (
    ArticleRecord,
    EvidenceRequest,
    EvidenceResponse,
    NormalizedDrug,
    QueryExecution,
    ResponseMeta,
)
from src.evidence.service import EvidenceService


class StubPolicy:
    def __init__(self, response: EvidenceResponse) -> None:
        self.response = response
        self.calls = 0

    async def run(self, request: EvidenceRequest, *, query_budget_remaining: int | None = None) -> EvidenceResponse:
        self.calls += 1
        return self.response


@pytest.mark.asyncio
async def test_service_caches_responses_by_question_intent_and_drugs() -> None:
    response = EvidenceResponse(
        normalized_drugs=[
            NormalizedDrug(raw_name="Tylenol", generic_name="acetaminophen", match_type="exact", rxnorm_cui="1")
        ],
        queries_executed=[QueryExecution(query="acetaminophen safety", strategy_label="narrow", result_count=5)],
        articles=[
            ArticleRecord(
                pmid="1",
                title="Study",
                abstract=None,
                authors=["Author"],
                journal=None,
                publication_date=None,
                doi=None,
                url=None,
                publication_types=[],
                study_type="review",
                heuristic_score=0.5,
                llm_relevance_score=0.8,
                llm_reason="Direct",
            )
        ],
        meta=ResponseMeta(
            pubmed_calls=1,
            articles_considered=1,
            articles_heuristic_filtered=0,
            articles_llm_scored=1,
            normalization_used=True,
            normalization_complete=True,
            evidence_strength_hint="high",
            cached=False,
            cost_estimate_usd=0.02,
            query_budget_used=1,
            query_budget_max=12,
            errors=[],
        ),
    )
    policy = StubPolicy(response)
    service = EvidenceService(policy=policy)
    request = EvidenceRequest(question="Is Tylenol safe?", intent="safety", drug_names=["Tylenol"])

    first = await service.run_request(request)
    second = await service.run_request(request)

    assert first.meta.cached is False
    assert second.meta.cached is True
    assert policy.calls == 1
