from __future__ import annotations

import pytest

from src.evidence.policy import EvidencePolicy
from src.evidence.schemas import ArticleRecord, EvidenceRequest, NormalizedDrug


class FailingResolver:
    def resolve_names(self, names: list[str]):
        raise RuntimeError("rxnorm unavailable")


class FailingQueryBuilder:
    def build_queries(self, **_kwargs):
        raise RuntimeError("llm output malformed")


class EmptyPubMedClient:
    def search_and_fetch(self, query: str, max_items: int = 30):
        raise RuntimeError("pubmed unavailable")


class EmptyFilter:
    def filter_and_rank(self, articles, **_kwargs):
        return []


class NonEmptyPubMedClient:
    def search_and_fetch(self, query: str, max_items: int = 30):
        return [
            {
                "pmid": "1",
                "title": "Article",
                "abstract": "Abstract",
                "authors": ["Author"],
                "journal": "Journal",
                "publication_date": "2024-01-01",
                "doi": None,
                "url": None,
                "publication_types": ["Review"],
                "study_type": "review",
                "heuristic_score": 0.7,
                "llm_relevance_score": None,
                "llm_reason": None,
            }
        ]


class PassThroughFilter:
    def filter_and_rank(self, articles, **_kwargs):
        return [ArticleRecord.model_validate(article) for article in articles]


class EmptyScorer:
    def score_articles(self, **_kwargs):
        return [], True


class HeuristicFallbackScorer:
    def score_articles(self, **_kwargs):
        return [
            ArticleRecord(
                pmid="1",
                title="Broad review",
                abstract="Background article",
                authors=["Author"],
                journal="Journal",
                publication_date="2024-01-01",
                doi=None,
                url=None,
                publication_types=["Review"],
                study_type="review",
                heuristic_score=0.7,
                llm_relevance_score=None,
                llm_reason=None,
            )
        ], True


class BuggyScorer:
    def score_articles(self, **_kwargs):
        raise TypeError("unexpected scorer bug")


@pytest.mark.asyncio
async def test_policy_returns_structurally_valid_low_evidence_response_when_layers_fail() -> None:
    policy = EvidencePolicy(
        resolver=FailingResolver(),
        query_builder=FailingQueryBuilder(),
        pubmed_client=EmptyPubMedClient(),
        ranking_filter=EmptyFilter(),
        llm_scorer=EmptyScorer(),
        query_budget_max=12,
        llm_scoring_candidates=15,
    )

    response = await policy.run(EvidenceRequest(question="Is creatine safe?", intent="safety", drug_names=["creatine"]))

    assert response.articles == []
    assert response.meta.evidence_strength_hint == "low"
    assert response.meta.errors
    assert response.normalized_drugs == [
        NormalizedDrug(raw_name="creatine", generic_name="creatine", match_type="none", rxnorm_cui=None)
    ]


@pytest.mark.asyncio
async def test_policy_preserves_heuristic_fallback_contract_when_llm_scoring_fails() -> None:
    policy = EvidencePolicy(
        resolver=FailingResolver(),
        query_builder=FailingQueryBuilder(),
        pubmed_client=EmptyPubMedClient(),
        ranking_filter=EmptyFilter(),
        llm_scorer=HeuristicFallbackScorer(),
        query_budget_max=12,
        llm_scoring_candidates=15,
    )

    response = await policy.run(EvidenceRequest(question="Is creatine safe?", intent="safety", drug_names=["creatine"]))

    assert [article.pmid for article in response.articles] == ["1"]
    assert response.articles[0].llm_relevance_score is None
    assert response.meta.evidence_strength_hint == "low"


@pytest.mark.asyncio
async def test_policy_surfaces_unexpected_scorer_errors() -> None:
    policy = EvidencePolicy(
        resolver=FailingResolver(),
        query_builder=FailingQueryBuilder(),
        pubmed_client=NonEmptyPubMedClient(),
        ranking_filter=PassThroughFilter(),
        llm_scorer=BuggyScorer(),
        query_budget_max=12,
        llm_scoring_candidates=15,
    )

    with pytest.raises(TypeError, match="unexpected scorer bug"):
        await policy.run(EvidenceRequest(question="Is creatine safe?", intent="safety", drug_names=["creatine"]))
