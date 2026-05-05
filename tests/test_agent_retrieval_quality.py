"""Unit tests for retrieval quality scoring."""
from __future__ import annotations

from unittest.mock import Mock

from src.agent.config import AgentConfig
from src.agent.models import QueryIntent
from src.agent.models import QueryPlan
from src.agent.models import RetrievedArticle
from src.agent.models import SubQuestion


def _plan_no_drugs(sub_questions: list[SubQuestion] | None = None) -> QueryPlan:
    return QueryPlan(
        normalized_question="ibuprofen question",
        intent=QueryIntent.drug_interaction,
        sub_questions=sub_questions or [SubQuestion(text="dosage monitoring", rationale="core", priority=1)],
        planned_queries=[],
        drugs_identified=[],
        reasoning="r",
    )
from src.agent.retrieval_quality import RetrievalQualityScorer
from src.ranking_filter import StudyRankingFilter
from src.source_relevance_scorer import SourceRelevanceScore
from src.source_relevance_scorer import SourceRelevanceScorer


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


def _plan(
    *,
    drugs_identified: list[str] | None = None,
    sub_questions: list[SubQuestion] | None = None,
) -> QueryPlan:
    return QueryPlan(
        normalized_question="warfarin question",
        intent=QueryIntent.drug_interaction,
        sub_questions=sub_questions or [SubQuestion(text="bleeding risk monitoring", rationale="core", priority=1)],
        planned_queries=[],
        drugs_identified=drugs_identified or ["warfarin"],
        reasoning="r",
    )


def test_assess_marks_direct_entity_match() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    article = _article(
        pmid="1",
        title="Warfarin review",
        abstract="Warfarin bleeding risk monitoring in adults",
        publication_types=["Review"],
    )

    result = scorer.assess(_plan(), [article], user_question="warfarin question")

    assert result.article_assessments[0].assessment_label == "direct"
    assert result.subquestion_coverages[0].covered is True


def test_assess_accepts_positional_user_question() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    article = _article(
        pmid="1",
        title="Warfarin review",
        abstract="Warfarin bleeding risk monitoring in adults",
        publication_types=["Review"],
    )

    result = scorer.assess(_plan(), [article], "warfarin question")

    assert result.article_assessments[0].assessment_label == "direct"


def test_assess_labels_supporting_when_only_semantic_signal_strong() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    relevance_scorer = Mock(spec=SourceRelevanceScorer)
    relevance_scorer.score_sources.return_value = [
        SourceRelevanceScore(
            pmid="1",
            title="Coagulation monitoring study",
            relevance_score=4.5,
            reasoning="high match",
            is_relevant=True,
        )
    ]
    scorer = RetrievalQualityScorer(config=config, source_relevance_scorer=relevance_scorer)
    article = _article(
        pmid="1",
        title="Coagulation monitoring study",
        abstract="Bleeding risk monitoring in anticoagulated patients",
        publication_types=["Review"],
    )

    result = scorer.assess(_plan(), [article], user_question="warfarin question")

    assert result.article_assessments[0].assessment_label == "supporting"


def test_assess_labels_tangential_for_mid_semantic() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    article = _article(
        pmid="1",
        title="Warfarin signal",
        abstract="General anticoagulant considerations",
        publication_types=["Review"],
    )

    result = scorer.assess(_plan(drugs_identified=["warfarin", "aspirin"]), [article], user_question="question")

    assert result.article_assessments[0].semantic_relevance_score == 3.0
    assert result.article_assessments[0].assessment_label == "tangential"


def test_assess_labels_irrelevant_when_no_signal() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    article = _article(
        pmid="1",
        title="Metformin study",
        abstract="Glucose control outcomes",
        publication_types=["Review"],
    )

    result = scorer.assess(_plan(), [article], user_question="warfarin question")

    assert result.article_assessments[0].assessment_label == "irrelevant"


def test_assess_does_not_emit_low_quality_evidence_without_qualifying_articles() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    articles = [
        _article(
            pmid="1",
            title="Warfarin signal",
            abstract="general anticoagulant considerations",
            publication_types=["Case Report"],
        ),
        _article(
            pmid="2",
            title="Metformin study",
            abstract="glucose control outcomes",
            publication_types=["Case Report"],
        ),
    ]

    result = scorer.assess(
        _plan(drugs_identified=["warfarin", "aspirin"]),
        articles,
        user_question="warfarin question",
    )

    assert {assessment.assessment_label for assessment in result.article_assessments} == {"tangential", "irrelevant"}
    assert "low_quality_evidence" not in result.failure_modes


def test_assess_emits_insufficient_direct_evidence() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    article = _article(
        pmid="1",
        title="Warfarin review",
        abstract="Warfarin bleeding risk monitoring",
        publication_types=["Review"],
    )

    result = scorer.assess(_plan(), [article], user_question="warfarin question")

    assert "insufficient_direct_evidence" in result.failure_modes


def test_assess_emits_subquestion_gap() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    articles = [
        _article(
            pmid=str(index),
            title=f"Warfarin review {index}",
            abstract="Warfarin bleeding risk monitoring",
            publication_types=["Review"],
        )
        for index in range(1, 4)
    ]
    plan = _plan(sub_questions=[SubQuestion(text="renal clearance adjustment", rationale="gap", priority=1)])

    result = scorer.assess(plan, articles, user_question="warfarin question")

    assert "subquestion_gap" in result.failure_modes
    assert result.subquestion_coverages[0].covered is False


def test_assess_emits_subquestion_gap_for_unmatched_acronym_subquestion() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    articles = [
        _article(
            pmid=str(index),
            title=f"Warfarin review {index}",
            abstract="Warfarin bleeding risk monitoring",
            publication_types=["Review"],
        )
        for index in range(1, 4)
    ]
    plan = _plan(sub_questions=[SubQuestion(text="INR", rationale="acronym gap", priority=1)])

    result = scorer.assess(plan, articles, user_question="warfarin question")

    assert "subquestion_gap" in result.failure_modes
    assert result.subquestion_coverages[0].covered is False


def test_assess_empty_articles_marks_acronym_subquestion_uncovered() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    plan = _plan(sub_questions=[SubQuestion(text="AUC", rationale="acronym gap", priority=1)])

    result = scorer.assess(plan, [], user_question="warfarin question")

    assert result.subquestion_coverages[0].covered is False
    assert result.subquestion_coverages[0].coverage_score == 0.0
    assert "subquestion_gap" in result.failure_modes


def test_assess_emits_low_quality_evidence() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    articles = [
        _article(
            pmid=str(index),
            title=f"Warfarin case {index}",
            abstract="Warfarin bleeding risk monitoring",
            publication_types=["Case Report"],
        )
        for index in range(1, 4)
    ]

    result = scorer.assess(_plan(), articles, user_question="warfarin question")

    assert "low_quality_evidence" in result.failure_modes


def test_assess_short_circuits_empty_articles() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)

    result = scorer.assess(_plan(), [], user_question="warfarin question")

    assert result.article_assessments == []
    assert result.failure_modes == ["insufficient_direct_evidence", "subquestion_gap"]
    assert result.retrieval_score == 0.0


def test_assess_truncates_to_evaluator_top_k() -> None:
    config = AgentConfig(
        enable_source_relevance_scoring=False,
        evaluator_top_k_articles=3,
        sufficient_relevant_articles=3,
        min_direct_evidence_articles=2,
    )
    scorer = RetrievalQualityScorer(config=config)
    articles = [
        _article(pmid="1", title="Warfarin case", abstract="warfarin monitoring", publication_types=["Case Report"]),
        _article(pmid="2", title="Warfarin cohort", abstract="warfarin monitoring", publication_types=["Cohort Study"]),
        _article(
            pmid="3",
            title="Warfarin systematic review",
            abstract="warfarin monitoring",
            publication_types=["Systematic Review"],
        ),
        _article(pmid="4", title="Warfarin review", abstract="warfarin monitoring", publication_types=["Review"]),
        _article(pmid="5", title="Warfarin trial", abstract="warfarin monitoring", publication_types=["Clinical Trial"]),
        _article(pmid="6", title="Warfarin note", abstract="warfarin monitoring", publication_types=["Editorial"]),
        _article(pmid="7", title="Warfarin case 2", abstract="warfarin monitoring", publication_types=["Case Report"]),
        _article(pmid="8", title="Warfarin cohort 2", abstract="warfarin monitoring", publication_types=["Cohort Study"]),
        _article(pmid="9", title="Warfarin review 2", abstract="warfarin monitoring", publication_types=["Review"]),
        _article(pmid="10", title="Warfarin trial 2", abstract="warfarin monitoring", publication_types=["Clinical Trial"]),
        _article(pmid="11", title="Warfarin case 3", abstract="warfarin monitoring", publication_types=["Case Report"]),
        _article(pmid="12", title="Warfarin review 3", abstract="warfarin monitoring", publication_types=["Review"]),
    ]

    result = scorer.assess(_plan(), articles, user_question="warfarin question")
    expected_order = [
        item["pmid"]
        for item in StudyRankingFilter().rank_studies(
            [
                {
                    "pmid": article.pmid,
                    "title": article.title,
                    "abstract": article.abstract,
                    "publication_types": article.publication_types,
                    "study_types": article.publication_types,
                    "publication_date": article.publication_date,
                    "year": None,
                }
                for article in articles
            ],
            query="warfarin question",
        )[:3]
    ]

    assert len(result.article_assessments) == 3
    assert [article.pmid for article in result.ranked_articles] == expected_order


def test_assess_uses_source_relevance_scorer_when_provided() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    relevance_scorer = Mock(spec=SourceRelevanceScorer)
    relevance_scorer.score_sources.return_value = [
        SourceRelevanceScore(
            pmid="1",
            title="Warfarin study",
            relevance_score=4.5,
            reasoning="x",
            is_relevant=True,
        )
    ]
    scorer = RetrievalQualityScorer(config=config, source_relevance_scorer=relevance_scorer)
    article = _article(
        pmid="1",
        title="Warfarin study",
        abstract="Warfarin bleeding risk monitoring",
        publication_types=["Review"],
    )

    result = scorer.assess(_plan(), [article], user_question="warfarin question")

    assert result.article_assessments[0].semantic_relevance_score == 4.5
    assert result.llm_output_tokens == 0
    assert result.llm_calls == 1


def test_assess_captures_llm_usage_when_relevance_scorer_runs() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    relevance_scorer = Mock(spec=SourceRelevanceScorer)
    relevance_scorer.score_sources.return_value = [
        SourceRelevanceScore(
            pmid="1",
            title="Warfarin study",
            relevance_score=4.5,
            reasoning="x",
            is_relevant=True,
        )
    ]
    relevance_scorer.llm = Mock()
    relevance_scorer.llm.last_usage_tokens = 321
    scorer = RetrievalQualityScorer(config=config, source_relevance_scorer=relevance_scorer)
    article = _article(
        pmid="1",
        title="Warfarin study",
        abstract="Warfarin bleeding risk monitoring",
        publication_types=["Review"],
    )

    result = scorer.assess(_plan(), [article], user_question="warfarin question")

    assert result.llm_output_tokens == 321
    assert result.llm_calls == 1


def test_assess_falls_back_when_relevance_scorer_raises() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    relevance_scorer = Mock(spec=SourceRelevanceScorer)
    relevance_scorer.score_sources.side_effect = RuntimeError("boom")
    scorer = RetrievalQualityScorer(config=config, source_relevance_scorer=relevance_scorer)
    article = _article(
        pmid="1",
        title="Warfarin review",
        abstract="Warfarin bleeding risk monitoring",
        publication_types=["Review"],
    )

    result = scorer.assess(_plan(), [article], user_question="warfarin question")

    assert result.article_assessments[0].semantic_relevance_score == 4.5
    assert result.article_assessments[0].assessment_label == "direct"
    assert result.llm_output_tokens == 0
    assert result.llm_calls == 0


def test_assess_does_not_direct_match_entity_substrings() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    article = _article(
        pmid="1",
        title="Warfarin review",
        abstract="Bleeding risk monitoring in adults",
        publication_types=["Review"],
    )
    plan = _plan(drugs_identified=["war"])

    result = scorer.assess(plan, [article], user_question="war question")

    assert result.article_assessments[0].direct_match_score == 0.0
    assert result.article_assessments[0].assessment_label != "direct"


def test_assess_matches_hyphenated_drug_entity_as_single_token() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    articles = [
        _article(
            pmid="1",
            title="TNF-alpha inhibitor review",
            abstract="TNF-alpha response monitoring in adults",
            publication_types=["Review"],
        ),
        _article(
            pmid="2",
            title="TNF alpha biomarker review",
            abstract="TNF alpha response monitoring in adults",
            publication_types=["Review"],
        ),
    ]
    plan = _plan(drugs_identified=["TNF-alpha"])

    result = scorer.assess(plan, articles, user_question="TNF-alpha question")
    assessments = {assessment.pmid: assessment for assessment in result.article_assessments}

    assert assessments["1"].direct_match_score == 1.0
    assert assessments["1"].assessment_label == "direct"
    assert assessments["2"].direct_match_score == 0.0
    assert assessments["2"].assessment_label != "direct"


def test_assess_does_not_cover_subquestion_keyword_substrings() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    article = _article(
        pmid="1",
        title="Adrenal metabolism review",
        abstract="Endocrine outcomes in adults",
        publication_types=["Review"],
    )
    plan = _plan(sub_questions=[SubQuestion(text="renal clearance", rationale="substring", priority=1)])

    result = scorer.assess(plan, [article], user_question="warfarin question")

    assert result.article_assessments[0].covered_subquestions == []
    assert result.subquestion_coverages[0].covered is False


def test_assess_matches_hyphenated_subquestion_keyword_as_single_token() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    articles = [
        _article(
            pmid="1",
            title="TNF-alpha pathway review",
            abstract="Immune signaling outcomes",
            publication_types=["Review"],
        ),
        _article(
            pmid="2",
            title="Alpha pathway review",
            abstract="Immune signaling outcomes",
            publication_types=["Review"],
        ),
    ]
    plan = _plan(sub_questions=[SubQuestion(text="TNF-alpha", rationale="hyphenated term", priority=1)])

    result = scorer.assess(plan, articles, user_question="warfarin question")
    assessments = {assessment.pmid: assessment for assessment in result.article_assessments}

    assert assessments["1"].covered_subquestions == [0]
    assert assessments["2"].covered_subquestions == []
    assert result.subquestion_coverages[0].matched_pmids == ["1"]
    assert result.subquestion_coverages[0].covered is True


def test_assess_empty_articles_no_keyword_subquestion_is_covered() -> None:
    """No-keyword subquestions must be covered=True even in the empty-articles short-circuit path."""
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    plan = _plan(sub_questions=[SubQuestion(text="is ok", rationale="all short words", priority=1)])

    result = scorer.assess(plan, [], user_question="question")

    assert result.subquestion_coverages[0].covered is True
    assert result.subquestion_coverages[0].coverage_score == 1.0
    assert "subquestion_gap" not in result.failure_modes


def test_assess_empty_drugs_with_high_semantic_score_is_supporting() -> None:
    """High semantic relevance produces supporting evidence even when entity extraction found no drugs."""
    config = AgentConfig(enable_source_relevance_scoring=False)
    relevance_scorer = Mock(spec=SourceRelevanceScorer)
    relevance_scorer.score_sources.return_value = [
        SourceRelevanceScore(
            pmid="1",
            title="Ibuprofen review",
            relevance_score=4.5,
            reasoning="high match",
            is_relevant=True,
        )
    ]
    scorer = RetrievalQualityScorer(config=config, source_relevance_scorer=relevance_scorer)
    article = _article(
        pmid="1",
        title="Ibuprofen review",
        abstract="Ibuprofen dosage monitoring",
        publication_types=["Review"],
    )
    plan = _plan_no_drugs()

    result = scorer.assess(plan, [article], user_question="ibuprofen question")

    assert result.article_assessments[0].assessment_label == "supporting"


def test_assess_handles_punctuated_subquestion_text() -> None:
    config = AgentConfig(enable_source_relevance_scoring=False)
    scorer = RetrievalQualityScorer(config=config)
    article = _article(
        pmid="1",
        title="Warfarin monitoring review",
        abstract="renal clearance adjustment in patients",
        publication_types=["Review"],
    )
    plan = _plan(sub_questions=[SubQuestion(text="renal clearance, adjustment?", rationale="punctuation", priority=1)])

    result = scorer.assess(plan, [article], user_question="warfarin question")

    assert result.subquestion_coverages[0].covered is True
