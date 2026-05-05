"""Deterministic retrieval evaluation utilities."""
from __future__ import annotations

from collections import Counter

from .config import AgentConfig
from .models import EvaluationVerdict, QueryPlan, RetrievedArticle
from .retrieval_quality import RetrievalQualityScorer


class RetrievalEvaluator:
    """Evaluate retrieval sufficiency without LLM calls."""

    def __init__(self, config: AgentConfig, retrieval_quality_scorer: RetrievalQualityScorer) -> None:
        self.config = config
        self.retrieval_quality_scorer = retrieval_quality_scorer
        self.last_assessment_output_tokens = 0
        self.last_assessment_llm_calls = 0

    def evaluate(self, plan: QueryPlan, articles: list[RetrievedArticle], turn_number: int) -> EvaluationVerdict:
        del turn_number
        self.last_assessment_output_tokens = 0
        self.last_assessment_llm_calls = 0
        assessment = self.retrieval_quality_scorer.assess(
            plan,
            articles,
            user_question=plan.normalized_question,
            )
        self.last_assessment_output_tokens = assessment.llm_output_tokens
        self.last_assessment_llm_calls = assessment.llm_calls
        relevant_articles = [
            article
            for article, article_assessment in zip(assessment.ranked_articles, assessment.article_assessments)
            if article_assessment.assessment_label in {"direct", "supporting"}
            ]

        study_type_distribution = dict(
            Counter(study_type for article in articles for study_type in article.publication_types)
            )
        coverage_gaps = [
            coverage.sub_question_text
            for coverage in assessment.subquestion_coverages
            if not coverage.covered
            ]

        relevant = len(relevant_articles)
        total = len(articles)
        flags: list[str] = []
        if self.config.sufficient_relevant_articles - 2 <= relevant < self.config.sufficient_relevant_articles:
            flags.append("borderline_article_count")
        if len(study_type_distribution) == 1 and study_type_distribution:
            flags.append("single_study_type_pool")

        if relevant == 0 and total > 0:
            recommendation = "pivot"
        elif total > self.config.too_many_articles_threshold and relevant < self.config.sufficient_relevant_articles:
            recommendation = "narrow"
        elif assessment.failure_modes:
            recommendation = "broaden"
        else:
            recommendation = "sufficient"

        return EvaluationVerdict(
            is_sufficient=(recommendation == "sufficient"),
            relevant_articles=relevant_articles,
            study_type_distribution=study_type_distribution,
            coverage_gaps=coverage_gaps,
            recommendation=recommendation,
            flags=flags,
            article_assessments=assessment.article_assessments,
            subquestion_coverages=assessment.subquestion_coverages,
            failure_modes=assessment.failure_modes,
            retrieval_score=assessment.retrieval_score,
            )
