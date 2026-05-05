from __future__ import annotations

import pytest

from src.evidence.policy import EvidencePolicy
from src.evidence.schemas import ArticleRecord, EvidenceRequest, EvidenceResponse, NormalizedDrug, QueryExecution, ResponseMeta
from src.evidence.policy import compute_evidence_strength_hint, enforce_query_budget
from src.evidence.ranking_filter import RankingFilter


class StubResolver:
    def resolve_names(self, names: list[str]) -> list[NormalizedDrug]:
        return [
            NormalizedDrug(raw_name=name, generic_name="acetaminophen", match_type="exact", rxnorm_cui="161")
            for name in names
        ]


class StubResolverWithContext:
    def __init__(
        self,
        *,
        normalized_drugs: list[NormalizedDrug],
        contexts: list[dict[str, object]],
    ) -> None:
        self.normalized_drugs = normalized_drugs
        self.contexts = contexts

    def resolve_names(self, _names: list[str]) -> list[NormalizedDrug]:
        return list(self.normalized_drugs)

    def resolve_names_with_context(self, _names: list[str]) -> list[dict[str, object]]:
        return list(self.contexts)


class StubQueryBuilder:
    def build_queries(self, **_kwargs):
        return [QueryExecution(query="acetaminophen safety", strategy_label="direct", result_count=0)]


class CapturingQueryBuilder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def build_queries(self, **kwargs):
        self.calls.append(kwargs)
        return [QueryExecution(query="acetaminophen safety", strategy_label="direct", result_count=0)]


class StubPubMedClient:
    def __init__(self, articles: list[dict[str, object]]) -> None:
        self.articles = articles

    def search_and_fetch(self, query: str, max_items: int = 30):
        return list(self.articles)


class PassThroughFilter:
    def __init__(self, articles: list[ArticleRecord]) -> None:
        self.articles = articles

    def filter_and_rank(self, _articles, **_kwargs):
        return list(self.articles)


class StubScorer:
    def __init__(self, articles: list[ArticleRecord], *, fallback_used: bool = False) -> None:
        self.articles = articles
        self.fallback_used = fallback_used

    def score_articles(self, **_kwargs):
        return list(self.articles), self.fallback_used


class EchoScorer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def score_articles(self, **kwargs):
        self.calls.append(kwargs)
        scored_articles = [
            article.model_copy(update={"llm_relevance_score": 0.75, "llm_reason": "rescued"})
            for article in kwargs["articles"]
        ]
        return scored_articles, False


class ContextObject:
    def __init__(
        self,
        *,
        raw_name: str,
        canonical_name: str,
        match_type: str,
        rxnorm_cui: str | None,
        confidence: float,
        aliases: list[str],
    ) -> None:
        self.raw_name = raw_name
        self.canonical_name = canonical_name
        self.match_type = match_type
        self.rxnorm_cui = rxnorm_cui
        self.confidence = confidence
        self.aliases = aliases

    def model_dump(self) -> dict[str, object]:
        return {
            "raw_name": self.raw_name,
            "canonical_name": self.canonical_name,
            "match_type": self.match_type,
            "rxnorm_cui": self.rxnorm_cui,
            "confidence": self.confidence,
            "aliases": list(self.aliases),
        }


class ContextResolver:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def resolve_names_with_context(self, _names: list[str]) -> list[object]:
        return [self.payload]


def _response_with_scores(*scores: float | None) -> EvidenceResponse:
    articles = [
        ArticleRecord(
            pmid=str(index),
            title=f"Article {index}",
            abstract=None,
            authors=["Author"],
            journal=None,
            publication_date=None,
            doi=None,
            url=None,
            publication_types=[],
            study_type="review",
            heuristic_score=0.1,
            llm_relevance_score=score,
            llm_reason=None,
        )
        for index, score in enumerate(scores, start=1)
    ]
    return EvidenceResponse(
        normalized_drugs=[
            NormalizedDrug(raw_name="Tylenol", generic_name="acetaminophen", match_type="exact", rxnorm_cui="1")
        ],
        queries_executed=[QueryExecution(query="acetaminophen safety", strategy_label="narrow", result_count=3)],
        articles=articles,
        meta=ResponseMeta(
            pubmed_calls=1,
            articles_considered=len(articles),
            articles_heuristic_filtered=0,
            articles_llm_scored=len(articles),
            normalization_used=True,
            normalization_complete=True,
            evidence_strength_hint="low",
            cached=False,
            cost_estimate_usd=0.01,
            query_budget_used=1,
            query_budget_max=12,
            errors=[],
        ),
    )


def test_compute_evidence_strength_hint_uses_mean_llm_score_thresholds() -> None:
    assert compute_evidence_strength_hint(_response_with_scores(0.9, 0.8, 0.75)) == "high"
    assert compute_evidence_strength_hint(_response_with_scores(0.6, 0.45, 0.4)) == "medium"
    assert compute_evidence_strength_hint(_response_with_scores(0.2, 0.3, 0.4)) == "low"


def test_enforce_query_budget_rejects_runs_over_cap() -> None:
    assert enforce_query_budget(current_used=11, additional_queries=1, maximum=12) == 12
    assert enforce_query_budget(current_used=11, additional_queries=2, maximum=12) == 12


def test_policy_prefilter_removes_articles_missing_drug_anchor_tokens() -> None:
    ranking_filter = RankingFilter()
    article = {
        "pmid": "1",
        "title": "Cannabinoid Effects of Metamizol/Dipyrone",
        "abstract": "Analgesic outcomes unrelated to the requested medication.",
        "authors": [],
        "journal": "Journal",
        "publication_date": "2026-01-01",
        "publication_types": ["Review"],
    }

    kept = ranking_filter.filter_and_rank([article], drug_terms=["acetaminophen", "tylenol"], intent="safety")

    assert kept == []


def test_policy_prefilter_requires_all_drug_anchor_groups_for_interactions() -> None:
    ranking_filter = RankingFilter()
    article = {
        "pmid": "1",
        "title": "Ibuprofen use in adults",
        "abstract": "Safety profile of ibuprofen monotherapy.",
        "authors": [],
        "journal": "Journal",
        "publication_date": "2026-01-01",
        "publication_types": ["Review"],
    }

    kept = ranking_filter.filter_and_rank(
        [article],
        drug_term_groups=[["Advil", "ibuprofen"], ["warfarin", "blood thinner"]],
        intent="interactions",
        require_all_groups=True,
    )

    assert kept == []


def test_policy_scoring_pool_keeps_candidates_from_multiple_query_strategies() -> None:
    policy = EvidencePolicy(query_budget_max=12, llm_scoring_candidates=3)
    articles = [
        _scored_article(pmid="1", score=0.8, heuristic_score=0.9),
        _scored_article(pmid="2", score=0.7, heuristic_score=0.8),
        _scored_article(pmid="3", score=0.6, heuristic_score=0.7),
    ]

    pool = policy._build_scoring_pool(
        filtered_articles=articles,
        pmid_sources={"1": {"broad"}, "2": {"focused"}, "3": {"focused"}},
        strategy_order=["broad", "focused"],
    )

    assert [item.pmid for item in pool[:2]] == ["1", "2"]


def test_policy_scoring_pool_uses_query_order_not_strategy_label_order() -> None:
    policy = EvidencePolicy(query_budget_max=12, llm_scoring_candidates=2)
    articles = [
        _scored_article(pmid="1", score=0.8, heuristic_score=0.9),
        _scored_article(pmid="2", score=0.7, heuristic_score=0.8),
        _scored_article(pmid="3", score=0.6, heuristic_score=0.7),
    ]

    original_pool = policy._build_scoring_pool(
        filtered_articles=articles,
        pmid_sources={"1": {"focused"}, "2": {"broad"}, "3": {"supplemental"}},
        strategy_order=["focused", "broad", "supplemental"],
    )
    renamed_pool = policy._build_scoring_pool(
        filtered_articles=articles,
        pmid_sources={"1": {"zeta"}, "2": {"alpha"}, "3": {"beta"}},
        strategy_order=["zeta", "alpha", "beta"],
    )
    fewer_candidates_pool = policy._build_scoring_pool(
        filtered_articles=articles[:2],
        pmid_sources={"1": {"zeta"}, "2": {"alpha"}},
        strategy_order=["zeta", "alpha", "beta"],
    )

    assert [item.pmid for item in original_pool] == ["1", "2"]
    assert [item.pmid for item in renamed_pool] == ["1", "2"]
    assert [item.pmid for item in fewer_candidates_pool] == ["1", "2"]


def _scored_article(*, pmid: str, score: float | None, heuristic_score: float = 0.1) -> ArticleRecord:
    return ArticleRecord(
        pmid=pmid,
        title=f"Article {pmid}",
        abstract="Abstract",
        authors=["Author"],
        journal="Journal",
        publication_date="2024-01-01",
        doi=None,
        url=None,
        publication_types=["Review"],
        study_type="review",
        heuristic_score=heuristic_score,
        llm_relevance_score=score,
        llm_reason="reason" if score is not None else None,
    )


@pytest.mark.parametrize(
    ("payload", "expected_generic"),
    [
        (
            ContextObject(
                raw_name="Tylenol",
                canonical_name="acetaminophen",
                match_type="exact",
                rxnorm_cui="161",
                confidence=0.92,
                aliases=["Tylenol", "acetaminophen"],
            ),
            "acetaminophen",
        ),
        (
            {
                "raw_name": "Tylenol",
                "canonical_name": "acetaminophen",
                "match_type": "exact",
                "rxnorm_cui": "161",
                "confidence": 0.92,
                "aliases": ["Tylenol", "acetaminophen"],
            },
            "acetaminophen",
        ),
        (
            NormalizedDrug(raw_name="Tylenol", generic_name="acetaminophen", match_type="exact", rxnorm_cui="161"),
            "acetaminophen",
        ),
    ],
)
@pytest.mark.asyncio
async def test_policy_accepts_multiple_resolver_context_shapes_without_losing_normalization(
    payload: object,
    expected_generic: str,
) -> None:
    strong_article = _scored_article(pmid="1", score=0.9)
    query_builder = CapturingQueryBuilder()
    policy = EvidencePolicy(
        resolver=ContextResolver(payload),
        query_builder=query_builder,
        pubmed_client=StubPubMedClient([strong_article.model_dump()]),
        ranking_filter=PassThroughFilter([strong_article]),
        llm_scorer=StubScorer([strong_article]),
        query_budget_max=12,
        llm_scoring_candidates=15,
    )

    response = await policy.run(EvidenceRequest(question="Is Tylenol safe to take?", intent="safety", drug_names=["Tylenol"]))

    assert response.normalized_drugs == [
        NormalizedDrug(raw_name="Tylenol", generic_name=expected_generic, match_type="exact", rxnorm_cui="161")
    ]
    assert response.meta.normalization_complete is True
    assert query_builder.calls[0]["drug_contexts"] == [
        {
            "raw_name": "Tylenol",
            "canonical_name": expected_generic,
            "match_type": "exact",
            "rxnorm_cui": "161",
            "confidence": pytest.approx(0.92 if not isinstance(payload, NormalizedDrug) else 1.0),
            "aliases": ["Tylenol", expected_generic],
        }
    ]


@pytest.mark.asyncio
async def test_policy_removes_low_semantic_score_articles_from_response() -> None:
    high_score = _scored_article(pmid="1", score=0.82)
    threshold_score = _scored_article(pmid="2", score=0.45)
    low_score = _scored_article(pmid="3", score=0.21)
    policy = EvidencePolicy(
        resolver=StubResolver(),
        query_builder=StubQueryBuilder(),
        pubmed_client=StubPubMedClient([high_score.model_dump(), threshold_score.model_dump(), low_score.model_dump()]),
        ranking_filter=PassThroughFilter([high_score, threshold_score, low_score]),
        llm_scorer=StubScorer([high_score, threshold_score, low_score]),
        query_budget_max=12,
        llm_scoring_candidates=15,
    )

    response = await policy.run(EvidenceRequest(question="Is Tylenol safe to take?", intent="safety", drug_names=["Tylenol"]))

    assert [article.pmid for article in response.articles] == ["1", "2"]
    assert response.meta.evidence_strength_hint == "medium"


@pytest.mark.asyncio
async def test_policy_returns_low_hint_when_no_article_clears_semantic_threshold() -> None:
    low_articles = [_scored_article(pmid="1", score=0.44), _scored_article(pmid="2", score=0.12)]
    policy = EvidencePolicy(
        resolver=StubResolver(),
        query_builder=StubQueryBuilder(),
        pubmed_client=StubPubMedClient([article.model_dump() for article in low_articles]),
        ranking_filter=PassThroughFilter(low_articles),
        llm_scorer=StubScorer(low_articles),
        query_budget_max=12,
        llm_scoring_candidates=15,
    )

    response = await policy.run(EvidenceRequest(question="Is Tylenol safe to take?", intent="safety", drug_names=["Tylenol"]))

    assert response.articles == []
    assert response.meta.evidence_strength_hint == "low"


@pytest.mark.asyncio
async def test_policy_preserves_public_response_shape_when_semantic_gating_applies() -> None:
    strong_article = _scored_article(pmid="1", score=0.9)
    policy = EvidencePolicy(
        resolver=StubResolver(),
        query_builder=StubQueryBuilder(),
        pubmed_client=StubPubMedClient([strong_article.model_dump()]),
        ranking_filter=PassThroughFilter([strong_article]),
        llm_scorer=StubScorer([strong_article]),
        query_budget_max=12,
        llm_scoring_candidates=15,
    )

    response = await policy.run(EvidenceRequest(question="Is Tylenol safe to take?", intent="safety", drug_names=["Tylenol"]))

    assert set(response.model_dump().keys()) == {"normalized_drugs", "queries_executed", "articles", "meta"}
    assert set(response.articles[0].model_dump().keys()) == {
        "pmid",
        "title",
        "abstract",
        "authors",
        "journal",
        "publication_date",
        "doi",
        "url",
        "publication_types",
        "study_type",
        "heuristic_score",
        "llm_relevance_score",
        "llm_reason",
    }


@pytest.mark.asyncio
async def test_policy_relaxes_lexical_gate_when_normalization_confidence_is_low() -> None:
    article = {
        "pmid": "1",
        "title": "Dietary supplementation and lean body mass",
        "abstract": "Randomized trial in resistance training participants.",
        "authors": ["Author"],
        "journal": "Journal",
        "publication_date": "2024-01-01",
        "publication_types": ["Clinical Trial"],
    }

    scorer = EchoScorer()
    policy = EvidencePolicy(
        resolver=StubResolverWithContext(
            normalized_drugs=[
                NormalizedDrug(
                    raw_name="muscle powder",
                    generic_name="creatine monohydrate powder",
                    match_type="approximate",
                    rxnorm_cui=None,
                )
            ],
            contexts=[
                {
                    "raw_name": "muscle powder",
                    "canonical_name": "creatine monohydrate powder",
                    "match_type": "approximate",
                    "confidence": 0.2,
                    "aliases": ["performance supplement", "monohydrate powder blend"],
                }
            ],
        ),
        query_builder=StubQueryBuilder(),
        pubmed_client=StubPubMedClient([article]),
        ranking_filter=RankingFilter(),
        llm_scorer=scorer,
        query_budget_max=12,
        llm_scoring_candidates=15,
    )

    response = await policy.run(
        EvidenceRequest(
            question="Does creatine help with muscle building?",
            intent="benefits",
            drug_names=["muscle powder"],
        )
    )

    assert [item.pmid for item in response.articles] == ["1"]
    assert [item.pmid for item in scorer.calls[0]["articles"]] == ["1"]


@pytest.mark.parametrize("intent", ["interactions", "comparison"])
@pytest.mark.asyncio
async def test_policy_does_not_relax_all_drug_anchor_requirement_for_multi_drug_low_confidence_requests(
    intent: str,
) -> None:
    article = {
        "pmid": "1",
        "title": "Ibuprofen monotherapy in adults",
        "abstract": "Safety profile for ibuprofen without coadministered anticoagulants.",
        "authors": ["Author"],
        "journal": "Journal",
        "publication_date": "2024-01-01",
        "publication_types": ["Clinical Trial"],
    }

    scorer = EchoScorer()
    policy = EvidencePolicy(
        resolver=StubResolverWithContext(
            normalized_drugs=[
                NormalizedDrug(raw_name="pain reliever", generic_name="ibuprofen blend", match_type="approximate", rxnorm_cui=None),
                NormalizedDrug(raw_name="blood thinner", generic_name="warfarin therapy pack", match_type="approximate", rxnorm_cui=None),
            ],
            contexts=[
                {
                    "raw_name": "pain reliever",
                    "canonical_name": "ibuprofen blend",
                    "match_type": "approximate",
                    "confidence": 0.2,
                    "aliases": ["pain relief product"],
                },
                {
                    "raw_name": "blood thinner",
                    "canonical_name": "warfarin therapy pack",
                    "match_type": "approximate",
                    "confidence": 0.2,
                    "aliases": ["anticoagulant therapy"],
                },
            ],
        ),
        query_builder=StubQueryBuilder(),
        pubmed_client=StubPubMedClient([article]),
        ranking_filter=RankingFilter(),
        llm_scorer=scorer,
        query_budget_max=12,
        llm_scoring_candidates=15,
    )

    response = await policy.run(
        EvidenceRequest(
            question="Can I combine a pain reliever with a blood thinner?",
            intent=intent,
            drug_names=["pain reliever", "blood thinner"],
        )
    )

    assert response.articles == []
    assert scorer.calls[0]["articles"] == []
