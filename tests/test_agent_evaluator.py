"""Unit tests for the agent retrieval evaluator."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.agent.config import AgentConfig
from src.agent.evaluator import RetrievalEvaluator
from src.agent.models import QueryIntent
from src.agent.models import QueryPlan
from src.agent.models import RetrievedArticle
from src.agent.models import SubQuestion
from src.agent.retrieval_quality import RetrievalQualityScorer
from src.source_relevance_scorer import SourceRelevanceScore
from src.source_relevance_scorer import SourceRelevanceScorer

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture_articles() -> list[RetrievedArticle]:
    payload = json.loads((FIXTURE_DIR / "agent_pubmed_articles.json").read_text(encoding="utf-8"))
    return [RetrievedArticle.model_validate(item) for item in payload]


def _article(
    *,
    pmid: str,
    title: str,
    abstract: str,
    publication_types: list[str],
) -> RetrievedArticle:
    return RetrievedArticle(
        pmid=pmid,
        title=title,
        abstract=abstract,
        authors=[],
        journal=None,
        publication_types=publication_types,
        source_system="pubmed",
    )


def _plan(sub_questions: list[SubQuestion] | None = None) -> QueryPlan:
    return QueryPlan(
        normalized_question="warfarin question",
        intent=QueryIntent.drug_interaction,
        sub_questions=sub_questions or [SubQuestion(text="bleeding risk monitoring", rationale="core", priority=1)],
        planned_queries=[],
        drugs_identified=["warfarin"],
        reasoning="r",
    )


def _make_evaluator(config: AgentConfig | None = None) -> RetrievalEvaluator:
    cfg = config or AgentConfig(enable_source_relevance_scoring=False)
    return RetrievalEvaluator(config=cfg, retrieval_quality_scorer=RetrievalQualityScorer(config=cfg))


def test_evaluate_sufficient_when_all_criteria_met() -> None:
    evaluator = _make_evaluator(AgentConfig(enable_source_relevance_scoring=False))
    articles = _load_fixture_articles()

    verdict = evaluator.evaluate(_plan(), articles, 1)

    assert verdict.is_sufficient is True
    assert verdict.recommendation == "sufficient"
    assert verdict.failure_modes == []
    assert verdict.retrieval_score > 0
    assert verdict.article_assessments
    assert verdict.subquestion_coverages


def test_evaluate_broaden_when_too_few_relevant() -> None:
    evaluator = _make_evaluator(AgentConfig(enable_source_relevance_scoring=False))
    articles = [
        _article(pmid="1", title="Warfarin review", abstract="monitoring", publication_types=["Review"]),
    ]

    verdict = evaluator.evaluate(_plan(), articles, 1)

    assert verdict.recommendation == "broaden"


def test_evaluate_pivot_when_zero_relevant() -> None:
    evaluator = _make_evaluator(AgentConfig(enable_source_relevance_scoring=False))
    articles = [
        _article(pmid=str(index), title="Aspirin review", abstract="not about the drug", publication_types=["Review"])
        for index in range(10)
    ]

    verdict = evaluator.evaluate(_plan(), articles, 1)

    assert verdict.recommendation == "pivot"
    assert "insufficient_direct_evidence" in verdict.failure_modes


def test_evaluate_narrow_when_too_many_low_signal() -> None:
    evaluator = _make_evaluator(AgentConfig(enable_source_relevance_scoring=False))
    articles = [
        _article(
            pmid=str(index),
            title="Warfarin signal" if index < 2 else "Aspirin review",
            abstract="warfarin monitoring" if index < 2 else "other topic",
            publication_types=["Review"],
        )
        for index in range(60)
    ]

    verdict = evaluator.evaluate(_plan(), articles, 1)

    assert verdict.recommendation == "narrow"


def test_evaluate_quality_gate_passes_when_no_high_quality_types() -> None:
    evaluator = _make_evaluator(AgentConfig(enable_source_relevance_scoring=False))
    articles = [
        _article(pmid="1", title="Warfarin case", abstract="bleeding risk monitoring", publication_types=["Case Report"])
    ]

    verdict = evaluator.evaluate(_plan(), articles, 1)

    assert verdict.recommendation == "broaden"


def test_evaluate_quality_gate_fails_when_rct_exists_but_not_relevant() -> None:
    evaluator = _make_evaluator(AgentConfig(enable_source_relevance_scoring=False))
    articles = [
        _article(pmid="1", title="Aspirin RCT", abstract="other drug", publication_types=["Randomized Controlled Trial"]),
        _article(pmid="2", title="Warfarin case", abstract="bleeding risk monitoring", publication_types=["Case Report"]),
        _article(pmid="3", title="Warfarin case 2", abstract="bleeding risk monitoring", publication_types=["Case Report"]),
        _article(pmid="4", title="Warfarin case 3", abstract="bleeding risk monitoring", publication_types=["Case Report"]),
        _article(pmid="5", title="Warfarin case 4", abstract="bleeding risk monitoring", publication_types=["Case Report"]),
        _article(pmid="6", title="Warfarin case 5", abstract="bleeding risk monitoring", publication_types=["Case Report"]),
    ]

    verdict = evaluator.evaluate(_plan(), articles, 1)

    assert verdict.recommendation == "broaden"
    assert "low_quality_evidence" in verdict.failure_modes


def test_evaluate_coverage_gaps_detected() -> None:
    evaluator = _make_evaluator(AgentConfig(enable_source_relevance_scoring=False))
    articles = [
        _article(pmid="1", title="Warfarin review", abstract="bleeding only", publication_types=["Review"])
    ]
    plan = _plan([SubQuestion(text="renal clearance adjustment", rationale="gap", priority=1)])

    verdict = evaluator.evaluate(plan, articles, 1)

    assert verdict.coverage_gaps
    assert "subquestion_gap" in verdict.failure_modes


def test_evaluate_unmatched_acronym_subquestion_is_coverage_gap() -> None:
    evaluator = _make_evaluator(AgentConfig(enable_source_relevance_scoring=False))
    articles = [
        _article(
            pmid="1",
            title="Warfarin review",
            abstract="bleeding risk monitoring",
            publication_types=["Review"],
        )
    ]
    plan = _plan([SubQuestion(text="AUC", rationale="acronym gap", priority=1)])

    verdict = evaluator.evaluate(plan, articles, 1)

    assert verdict.coverage_gaps == ["AUC"]
    assert "subquestion_gap" in verdict.failure_modes


def test_evaluate_flags_borderline_count() -> None:
    evaluator = _make_evaluator(AgentConfig(sufficient_relevant_articles=5, enable_source_relevance_scoring=False))
    articles = [
        _article(pmid=str(index), title="Warfarin review", abstract="bleeding risk monitoring", publication_types=["Review"])
        for index in range(4)
    ]

    verdict = evaluator.evaluate(_plan(), articles, 1)

    assert "borderline_article_count" in verdict.flags


def test_evaluate_flags_single_study_type() -> None:
    evaluator = _make_evaluator(AgentConfig(enable_source_relevance_scoring=False))
    articles = [
        article.model_copy(update={"publication_types": ["Review"]}) for article in _load_fixture_articles()
    ]

    verdict = evaluator.evaluate(_plan(), articles, 1)

    assert "single_study_type_pool" in verdict.flags


def test_is_subquestion_covered_strips_trailing_punctuation() -> None:
    evaluator = _make_evaluator(AgentConfig(enable_source_relevance_scoring=False))
    articles = [
        _article(pmid="1", title="Warfarin monitoring review", abstract="renal clearance adjustment in patients", publication_types=["Review"])
    ]
    plan = _plan([SubQuestion(text="renal clearance, adjustment?", rationale="punctuation test", priority=1)])

    verdict = evaluator.evaluate(plan, articles, 1)

    assert not verdict.coverage_gaps, "Punctuated sub-question should be covered when keywords match article text"


def test_evaluate_empty_articles_returns_broaden() -> None:
    evaluator = _make_evaluator(AgentConfig(enable_source_relevance_scoring=False))

    verdict = evaluator.evaluate(_plan(), [], 1)

    assert verdict.recommendation == "broaden"
    assert verdict.is_sufficient is False


def test_evaluate_with_empty_drugs_and_no_semantic_signal_has_zero_relevant() -> None:
    """Empty drugs do not create relevance by themselves; semantic relevance still controls supporting evidence."""
    evaluator = _make_evaluator(AgentConfig(enable_source_relevance_scoring=False))
    articles = [
        _article(pmid=str(i), title="Ibuprofen review", abstract="monitoring ibuprofen dosage", publication_types=["Review"])
        for i in range(6)
    ]
    plan = QueryPlan(
        normalized_question="ibuprofen dosage monitoring",
        intent=QueryIntent.drug_interaction,
        sub_questions=[SubQuestion(text="dosage monitoring", rationale="core", priority=1)],
        planned_queries=[],
        drugs_identified=[],
        reasoning="r",
    )

    verdict = evaluator.evaluate(plan, articles, 1)

    assert verdict.relevant_articles == []
    assert verdict.is_sufficient is False
    assert verdict.recommendation in {"broaden", "pivot"}


def test_evaluate_empty_drugs_with_high_semantic_score_has_supporting_relevant_articles() -> None:
    """Semantic-threshold evidence is supporting even when drugs_identified is empty."""
    config = AgentConfig(
        enable_source_relevance_scoring=False,
        min_direct_evidence_articles=1,
    )
    relevance_scorer = Mock(spec=SourceRelevanceScorer)
    relevance_scorer.score_sources.return_value = [
        SourceRelevanceScore(pmid=str(i), title=f"Article {i}", relevance_score=4.5, reasoning="x", is_relevant=True)
        for i in range(5)
    ]
    quality_scorer = RetrievalQualityScorer(config=config, source_relevance_scorer=relevance_scorer)
    evaluator = RetrievalEvaluator(config=config, retrieval_quality_scorer=quality_scorer)
    articles = [
        _article(
            pmid=str(i),
            title=f"Ibuprofen review {i}",
            abstract="ibuprofen dosage monitoring",
            publication_types=["Review"],
        )
        for i in range(5)
    ]
    plan = QueryPlan(
        normalized_question="ibuprofen dosage monitoring",
        intent=QueryIntent.drug_interaction,
        sub_questions=[SubQuestion(text="dosage monitoring", rationale="core", priority=1)],
        planned_queries=[],
        drugs_identified=[],
        reasoning="r",
    )

    verdict = evaluator.evaluate(plan, articles, 1)

    assert [article.pmid for article in verdict.relevant_articles] == ["0", "1", "2", "3", "4"]
    assert {assessment.assessment_label for assessment in verdict.article_assessments} == {"supporting"}
    assert verdict.recommendation != "pivot"
    assert "insufficient_direct_evidence" in verdict.failure_modes


def test_evaluate_with_empty_drugs_low_semantic_pool_remains_non_relevant() -> None:
    """A low semantic pool remains non-relevant when no direct drug evidence is available."""
    evaluator = _make_evaluator(AgentConfig(enable_source_relevance_scoring=False))
    articles = [
        _article(pmid=str(i), title="Aspirin study", abstract="aspirin cardiovascular outcomes", publication_types=["Review"])
        for i in range(10)
    ]
    plan = QueryPlan(
        normalized_question="warfarin bleeding risk",
        intent=QueryIntent.drug_interaction,
        sub_questions=[SubQuestion(text="renal clearance adjustment", rationale="gap", priority=1)],
        planned_queries=[],
        drugs_identified=[],
        reasoning="r",
    )

    verdict = evaluator.evaluate(plan, articles, 1)

    assert verdict.relevant_articles == []
    assert verdict.recommendation in {"broaden", "pivot"}
    assert verdict.is_sufficient is False


def test_evaluate_propagates_retrieval_quality_llm_usage_metadata() -> None:
    relevance_scorer = Mock(spec=SourceRelevanceScorer)
    relevance_scorer.score_sources.return_value = [
        SourceRelevanceScore(
            pmid="1",
            title="Warfarin review",
            relevance_score=4.5,
            reasoning="high match",
            is_relevant=True,
        )
    ]
    relevance_scorer.llm = Mock()
    relevance_scorer.llm.last_usage_tokens = 123
    config = AgentConfig(enable_source_relevance_scoring=True)
    quality_scorer = RetrievalQualityScorer(config=config, source_relevance_scorer=relevance_scorer)
    evaluator = RetrievalEvaluator(config=config, retrieval_quality_scorer=quality_scorer)
    articles = [
        _article(pmid="1", title="Warfarin review", abstract="bleeding risk monitoring", publication_types=["Review"])
    ]

    evaluator.evaluate(_plan(), articles, 1)

    assert evaluator.last_assessment_output_tokens == 123
    assert evaluator.last_assessment_llm_calls == 1


def test_evaluate_reports_zero_llm_usage_without_relevance_scorer() -> None:
    evaluator = _make_evaluator(AgentConfig(enable_source_relevance_scoring=False))
    articles = [
        _article(pmid="1", title="Warfarin review", abstract="bleeding risk monitoring", publication_types=["Review"])
    ]

    evaluator.evaluate(_plan(), articles, 1)

    assert evaluator.last_assessment_output_tokens == 0
    assert evaluator.last_assessment_llm_calls == 0


def test_evaluate_can_reach_sufficient_when_top_k_matches_threshold() -> None:
    config = AgentConfig(
        enable_source_relevance_scoring=False,
        evaluator_top_k_articles=3,
        sufficient_relevant_articles=3,
        min_direct_evidence_articles=2,
    )
    evaluator = _make_evaluator(config)
    articles = [
        _article(pmid=str(index), title="Warfarin review", abstract="bleeding risk monitoring", publication_types=["Review"])
        for index in range(3)
    ]

    verdict = evaluator.evaluate(_plan(), articles, 1)

    assert verdict.is_sufficient is True


def test_evaluate_rejects_invalid_top_k_configuration() -> None:
    with pytest.raises(ValueError, match="evaluator_top_k_articles"):
        AgentConfig(
            enable_source_relevance_scoring=False,
            evaluator_top_k_articles=1,
            sufficient_relevant_articles=2,
            min_direct_evidence_articles=2,
        )
