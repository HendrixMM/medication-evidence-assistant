"""Retrieval quality scoring utilities."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import re

from .config import AgentConfig
from .models import ArticleAssessment, QueryPlan, RetrievedArticle, SubQuestionCoverage
from src.ranking_filter import StudyRankingFilter
from src.source_relevance_scorer import SourceRelevanceScorer

logger = logging.getLogger(__name__)

HIGH_QUALITY_TYPES = {
    "randomized controlled trial",
    "meta-analysis",
    "systematic review",
    "review",
}

SHORT_DOMAIN_KEYWORDS = {
    "alt",
    "ast",
    "auc",
    "cyp",
    "ecg",
    "ekg",
    "inr",
    "pk",
    "pd",
    "pt",
    "qt",
}


@dataclass
class RetrievalAssessmentResult:
    article_assessments: list[ArticleAssessment]
    subquestion_coverages: list[SubQuestionCoverage]
    failure_modes: list[str]
    retrieval_score: float
    ranked_articles: list[RetrievedArticle]
    llm_output_tokens: int = 0
    llm_calls: int = 0


class RetrievalQualityScorer:
    def __init__(
        self,
        config: AgentConfig,
        ranking_filter: StudyRankingFilter | None = None,
        source_relevance_scorer: SourceRelevanceScorer | None = None,
    ) -> None:
        self.config = config
        self.ranking_filter = ranking_filter or StudyRankingFilter()
        self.source_relevance_scorer = source_relevance_scorer

    def assess(
        self,
        plan: QueryPlan,
        articles: list[RetrievedArticle],
        user_question: str,
    ) -> RetrievalAssessmentResult:
        if not articles:
            empty_coverages = self._empty_coverages(plan)
            empty_failure_modes: list[str] = ["insufficient_direct_evidence"]
            empty_covered_ratio = (
                sum(1 for c in empty_coverages if c.covered) / len(empty_coverages)
                if empty_coverages
                else 1.0
            )
            if empty_covered_ratio < self.config.min_subquestion_coverage_ratio:
                empty_failure_modes.append("subquestion_gap")
            return RetrievalAssessmentResult(
                article_assessments=[],
                subquestion_coverages=empty_coverages,
                failure_modes=empty_failure_modes,
                retrieval_score=0.0,
                ranked_articles=[],
                llm_output_tokens=0,
                llm_calls=0,
            )

        article_lookup = {article.pmid: article for article in articles}
        ranked_payloads = self.ranking_filter.rank_studies(
            [self._article_to_dict(article) for article in articles],
            query=user_question,
            entities=plan.drugs_identified,
            intent=plan.intent.value,
        )
        ranked_pmids: list[str] = []
        for payload in ranked_payloads:
            pmid = payload.get("pmid")
            if isinstance(pmid, str) and pmid in article_lookup and pmid not in ranked_pmids:
                ranked_pmids.append(pmid)
        for article in articles:
            if article.pmid not in ranked_pmids:
                ranked_pmids.append(article.pmid)
        ranked_articles = [article_lookup[pmid] for pmid in ranked_pmids][: self.config.evaluator_top_k_articles]
        article_tokens_by_pmid = {
            article.pmid: self._article_tokens(article)
            for article in ranked_articles
        }

        semantic_scores, llm_output_tokens, llm_calls = self._semantic_scores(
            plan,
            ranked_articles,
            user_question,
            article_tokens_by_pmid,
        )

        article_assessments: list[ArticleAssessment] = []
        for article in ranked_articles:
            article_tokens = article_tokens_by_pmid[article.pmid]
            direct_match_score = self._direct_match_score(article_tokens, plan.drugs_identified)
            quality_score = self._quality_score(article.publication_types)
            covered_subquestions = self._matching_subquestion_indices(article_tokens, plan.sub_questions)
            semantic_score = semantic_scores.get(
                article.pmid,
                self._fallback_semantic_score(article_tokens, plan.drugs_identified),
            )
            assessment_label = self._label_article(direct_match_score, semantic_score)
            article_assessments.append(
                ArticleAssessment(
                    pmid=article.pmid,
                    direct_match_score=direct_match_score,
                    semantic_relevance_score=semantic_score,
                    quality_score=quality_score,
                    covered_subquestions=covered_subquestions,
                    assessment_label=assessment_label,
                )
            )

        subquestion_coverages: list[SubQuestionCoverage] = []
        for index, sub_question in enumerate(plan.sub_questions):
            keywords = self._subquestion_keywords(sub_question.text)
            if not keywords:
                subquestion_coverages.append(
                    SubQuestionCoverage(
                        sub_question_index=index,
                        sub_question_text=sub_question.text,
                        coverage_score=1.0,
                        matched_pmids=[],
                        covered=True,
                    )
                )
                continue

            matched_pmids = [
                assessment.pmid
                for assessment in article_assessments
                if index in assessment.covered_subquestions
            ]
            coverage_score = min(1.0, len(matched_pmids) / max(1, self.config.min_direct_evidence_articles))
            subquestion_coverages.append(
                SubQuestionCoverage(
                    sub_question_index=index,
                    sub_question_text=sub_question.text,
                    coverage_score=coverage_score,
                    matched_pmids=matched_pmids,
                    covered=len(matched_pmids) >= 1,
                )
            )

        direct_count = sum(1 for assessment in article_assessments if assessment.assessment_label == "direct")
        covered_ratio = (
            sum(1 for coverage in subquestion_coverages if coverage.covered) / len(subquestion_coverages)
            if subquestion_coverages
            else 1.0
        )
        top_quality_count = sum(
            1
            for assessment in article_assessments
            if assessment.assessment_label in {"direct", "supporting"} and assessment.quality_score >= 0.5
        )
        qualifying_evidence_count = sum(
            1
            for assessment in article_assessments
            if assessment.assessment_label in {"direct", "supporting"}
        )
        failure_modes = self._failure_modes(
            direct_count,
            covered_ratio,
            top_quality_count,
            qualifying_evidence_count,
        )

        direct_ratio = min(1.0, direct_count / max(1, self.config.min_direct_evidence_articles))
        quality_ratio = min(1.0, top_quality_count / max(1, len(article_assessments)))
        retrieval_score = max(0.0, min(1.0, 0.5 * direct_ratio + 0.3 * covered_ratio + 0.2 * quality_ratio))

        return RetrievalAssessmentResult(
            article_assessments=article_assessments,
            subquestion_coverages=subquestion_coverages,
            failure_modes=failure_modes,
            retrieval_score=retrieval_score,
            ranked_articles=ranked_articles,
            llm_output_tokens=llm_output_tokens,
            llm_calls=llm_calls,
        )

    def _semantic_scores(
        self,
        plan: QueryPlan,
        ranked_articles: list[RetrievedArticle],
        user_question: str,
        article_tokens_by_pmid: Mapping[str, Sequence[str]],
    ) -> tuple[dict[str, float], int, int]:
        if self.source_relevance_scorer is not None:
            try:
                scores = self.source_relevance_scorer.score_sources(
                    user_question,
                    [self._article_to_dict(article) for article in ranked_articles],
                )
                return (
                    {score.pmid: float(score.relevance_score) for score in scores},
                    self._read_usage_tokens(getattr(self.source_relevance_scorer, "llm", None)),
                    1,
                )
            except Exception:
                logger.warning("Source relevance scoring failed; falling back to deterministic scoring.", exc_info=True)

        return (
            {
                article.pmid: self._fallback_semantic_score(
                    article_tokens_by_pmid[article.pmid],
                    plan.drugs_identified,
                )
                for article in ranked_articles
            },
            0,
            0,
        )

    @staticmethod
    def _read_usage_tokens(llm_client: object | None) -> int:
        raw_tokens = getattr(llm_client, "last_usage_tokens", 0)
        return raw_tokens if isinstance(raw_tokens, int) and raw_tokens >= 0 else 0

    def _article_to_dict(self, article: RetrievedArticle) -> dict[str, object]:
        publication_year: int | None = None
        if article.publication_date:
            year_match = re.search(r"\b(\d{4})\b", article.publication_date)
            if year_match:
                publication_year = int(year_match.group(1))
        return {
            "pmid": article.pmid,
            "title": article.title or "",
            "abstract": article.abstract or "",
            "publication_types": list(article.publication_types),
            "study_types": list(article.publication_types),
            "publication_date": article.publication_date,
            "publication_year": publication_year,
            "year": publication_year,
        }

    @staticmethod
    def _text_tokens(text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"(?<![a-z0-9-])[a-z0-9]+(?:-[a-z0-9]+)*(?![a-z0-9-])", text.lower()))

    def _article_tokens(self, article: RetrievedArticle) -> tuple[str, ...]:
        return self._text_tokens(f"{article.title or ''} {article.abstract or ''}")

    def _contains_token_phrase(self, haystack_tokens: Sequence[str], phrase: str) -> bool:
        phrase_tokens = self._text_tokens(phrase)
        if not phrase_tokens:
            return False
        if len(phrase_tokens) == 1:
            return phrase_tokens[0] in haystack_tokens

        phrase_length = len(phrase_tokens)
        for start in range(0, len(haystack_tokens) - phrase_length + 1):
            if tuple(haystack_tokens[start : start + phrase_length]) == phrase_tokens:
                return True
        return False

    def _direct_match_score(self, article_tokens: Sequence[str], drugs: list[str]) -> float:
        if not drugs:
            return 0.0
        matched = sum(1 for drug in drugs if self._contains_token_phrase(article_tokens, drug))
        return matched / len(drugs)

    def _fallback_semantic_score(self, article_tokens: Sequence[str], drugs: list[str]) -> float:
        if not drugs:
            return 1.5
        matched = sum(1 for drug in drugs if self._contains_token_phrase(article_tokens, drug))
        if matched == len(drugs):
            return 4.5
        if matched >= 1:
            return 3.0
        return 1.5

    def _quality_score(self, publication_types: list[str]) -> float:
        lowered = {publication_type.lower() for publication_type in publication_types}
        if lowered & HIGH_QUALITY_TYPES:
            return 1.0
        if {"cohort study", "clinical trial"} & lowered:
            return 0.5
        if "case report" in lowered:
            return 0.2
        return 0.3

    def _subquestion_keywords(self, text: str) -> list[str]:
        tokens = self._text_tokens(text)
        return [token for token in tokens if len(token) >= 4 or token in SHORT_DOMAIN_KEYWORDS]

    def _matching_subquestion_indices(
        self,
        article_tokens: Sequence[str],
        sub_questions: list,
    ) -> list[int]:
        article_token_set = set(article_tokens)
        matching_indices: list[int] = []
        for index, sub_question in enumerate(sub_questions):
            keywords = self._subquestion_keywords(sub_question.text)
            if keywords and any(keyword in article_token_set for keyword in keywords):
                matching_indices.append(index)
        return matching_indices

    def _label_article(self, direct_match_score: float, semantic_score: float) -> str:
        if direct_match_score >= 0.8 and semantic_score >= self.config.semantic_relevance_min_score:
            return "direct"
        if semantic_score >= self.config.semantic_relevance_min_score:
            return "supporting"
        if semantic_score >= 2.5:
            return "tangential"
        return "irrelevant"

    def _failure_modes(
        self,
        direct_count: int,
        covered_ratio: float,
        top_quality_count: int,
        qualifying_evidence_count: int,
    ) -> list[str]:
        failure_modes: list[str] = []
        if direct_count < self.config.min_direct_evidence_articles:
            failure_modes.append("insufficient_direct_evidence")
        if covered_ratio < self.config.min_subquestion_coverage_ratio:
            failure_modes.append("subquestion_gap")
        if qualifying_evidence_count > 0 and top_quality_count == 0:
            failure_modes.append("low_quality_evidence")
        return failure_modes

    def _empty_coverages(self, plan: QueryPlan) -> list[SubQuestionCoverage]:
        coverages: list[SubQuestionCoverage] = []
        for index, sub_question in enumerate(plan.sub_questions):
            keywords = self._subquestion_keywords(sub_question.text)
            if not keywords:
                coverages.append(
                    SubQuestionCoverage(
                        sub_question_index=index,
                        sub_question_text=sub_question.text,
                        coverage_score=1.0,
                        matched_pmids=[],
                        covered=True,
                    )
                )
            else:
                coverages.append(
                    SubQuestionCoverage(
                        sub_question_index=index,
                        sub_question_text=sub_question.text,
                        coverage_score=0.0,
                        matched_pmids=[],
                        covered=False,
                    )
                )
        return coverages


__all__ = ["RetrievalAssessmentResult", "RetrievalQualityScorer"]
