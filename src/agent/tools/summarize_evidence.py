"""Evidence summarization adapter for agent workflows."""
from __future__ import annotations

import json
import logging
import re

from src.ddi_pk_processor import DDIPKProcessor
from src.paper_schema import Paper, coerce_paper
from src.ranking_filter import StudyRankingFilter
from src.source_relevance_scorer import SourceRelevanceScorer
from src.synthesis_engine import SynthesisEngine

from ..config import AgentConfig
from ..models import EvidenceLevel, EvidenceSummary, QueryIntent, RetrievedArticle

logger = logging.getLogger(__name__)

_EVIDENCE_LEVEL_MAP = {
    "Level 1": EvidenceLevel.systematic_review,
    "Level 2": EvidenceLevel.rct,
    "Level 3": EvidenceLevel.cohort,
    "Level 4": EvidenceLevel.case_report,
}


class SummarizeEvidenceTool:
    """Rank, synthesize, and optionally score relevance for retrieved evidence."""

    def __init__(
        self,
        ranking_filter: StudyRankingFilter,
        synthesis_engine: SynthesisEngine,
        ddi_pk_processor: DDIPKProcessor,
        relevance_scorer: SourceRelevanceScorer | None,
        config: AgentConfig,
    ) -> None:
        self.ranking_filter = ranking_filter
        self.synthesis_engine = synthesis_engine
        self.ddi_pk_processor = ddi_pk_processor
        self.relevance_scorer = relevance_scorer
        self.config = config

    def execute(
        self,
        articles: list[RetrievedArticle],
        query: str,
        intent: QueryIntent,
        drugs_identified: list[str],
    ) -> EvidenceSummary:
        try:
            papers = self._build_papers(articles)
            ranked_dicts = self.ranking_filter.rank_studies(
                [paper.as_dict() for paper in papers],
                query,
                entities=drugs_identified,
                intent=intent.value,
            )
            ranked_articles = self._ranked_articles_from_dicts(articles, ranked_dicts)
            if not ranked_articles and articles:
                ranked_articles = [
                    article.model_copy(update={"ranking_score": 0.0})
                    for article in articles
                ]

            meta = self.synthesis_engine.generate_meta_summary(ranked_dicts, query)
            key_findings = self._normalize_text_entries(meta.get("key_findings", []) or [])
            comparative_analysis = meta.get("comparative_analysis", {}) or {}
            convergent_findings = self._normalize_text_entries(
                comparative_analysis.get("convergent_findings", []) or []
            )
            divergent_findings = self._normalize_text_entries(
                comparative_analysis.get("divergent_findings", []) or []
            )
            research_gaps = self._normalize_text_entries(meta.get("research_gaps", []) or [])
            evidence_level = _EVIDENCE_LEVEL_MAP.get(
                str((meta.get("evidence_synthesis", {}) or {}).get("highest_level", "")),
                EvidenceLevel.insufficient,
            )

            if intent == QueryIntent.drug_interaction and len(drugs_identified) >= 1:
                primary_drug = drugs_identified[0]
                secondary_drugs = drugs_identified[1:]
                ddi_result = self.ddi_pk_processor.analyze_drug_interactions(
                    [paper.as_dict() for paper in papers],
                    primary_drug,
                    secondary_drugs or None,
                )
                drug_interactions = [ddi_result]
                pk_parameters = dict(ddi_result.get("pk_parameters", {}) or {})
            else:
                drug_interactions = []
                pk_parameters = {}

            if self.config.enable_source_relevance_scoring and self.relevance_scorer is not None:
                sources_for_scorer = [
                    {"pmid": article.pmid, "title": article.title or "", "abstract": article.abstract or ""}
                    for article in ranked_articles
                ]
                scores = self.relevance_scorer.score_sources(query, sources_for_scorer)
                score_map = {score.pmid: float(score.relevance_score) for score in scores}
                ranked_articles = [
                    article.model_copy(update={"relevance_score": score_map.get(article.pmid, article.relevance_score)})
                    for article in ranked_articles
                ]

            return EvidenceSummary(
                ranked_articles=ranked_articles,
                evidence_level=evidence_level,
                key_findings=key_findings,
                convergent_findings=convergent_findings,
                divergent_findings=divergent_findings,
                drug_interactions=drug_interactions,
                pk_parameters=pk_parameters,
                research_gaps=research_gaps,
            )
        except Exception:
            logger.exception("SummarizeEvidenceTool failed")
            return EvidenceSummary(
                ranked_articles=articles,
                evidence_level=EvidenceLevel.insufficient,
                key_findings=[],
                convergent_findings=[],
                divergent_findings=[],
                drug_interactions=[],
                pk_parameters={},
                research_gaps=[],
            )

    @staticmethod
    def _build_papers(articles: list[RetrievedArticle]) -> list[Paper]:
        papers: list[Paper] = []
        for article in articles:
            page_content = article.abstract or article.title or ""
            publication_year = SummarizeEvidenceTool._extract_publication_year(article.publication_date)
            paper_entry = {
                "page_content": page_content,
                "content": page_content,
                "title": article.title,
                "abstract": article.abstract,
                "pmid": article.pmid,
                "publication_year": publication_year,
                "year": publication_year,
                "metadata": {
                    "title": article.title,
                    "abstract": article.abstract,
                    "pmid": article.pmid,
                    "journal": article.journal,
                    "publication_types": article.publication_types,
                    "publication_year": publication_year,
                    "year": publication_year,
                },
                "study_types": list(article.publication_types),
                "tags": list(article.publication_types),
            }
            papers.append(coerce_paper(paper_entry))
        return papers

    @staticmethod
    def _ranked_articles_from_dicts(
        articles: list[RetrievedArticle],
        ranked_dicts: list[dict],
    ) -> list[RetrievedArticle]:
        article_map = {article.pmid: article for article in articles}
        ranked_articles: list[RetrievedArticle] = []
        matched_pmids: set[str] = set()
        for ranked in ranked_dicts:
            pmid = str(ranked.get("pmid") or (ranked.get("metadata", {}) or {}).get("pmid") or "")
            article = article_map.get(pmid)
            if article is None:
                continue
            matched_pmids.add(pmid)
            ranked_articles.append(
                article.model_copy(update={"ranking_score": float(ranked.get("ranking_score", 0.0) or 0.0)})
            )
        for article in articles:
            if article.pmid in matched_pmids:
                continue
            ranked_articles.append(article.model_copy(update={"ranking_score": article.ranking_score}))
        return ranked_articles

    @staticmethod
    def _normalize_findings(findings: list[object]) -> list[str]:
        normalized: list[str] = []
        for finding in findings:
            if isinstance(finding, str):
                normalized.append(finding)
            elif isinstance(finding, dict):
                normalized.append(SummarizeEvidenceTool._stringify_finding_dict(finding))
            else:
                normalized.append(str(finding))
        return normalized

    @staticmethod
    def _normalize_text_entries(value: object) -> list[str]:
        if isinstance(value, (list, tuple)):
            return SummarizeEvidenceTool._normalize_findings(list(value))
        if value in (None, ""):
            return []
        return SummarizeEvidenceTool._normalize_findings([value])

    @staticmethod
    def _stringify_finding_dict(finding: dict[str, object]) -> str:
        for key in ("finding", "summary", "text", "description", "title"):
            value = finding.get(key)
            if value:
                return str(value)

        convergent_finding = finding.get("convergent_finding")
        if convergent_finding:
            drug = finding.get("drug")
            papers_count = finding.get("papers_count")
            text = f"For {drug}: {convergent_finding}" if drug else str(convergent_finding)
            if papers_count:
                text += f" ({papers_count} studies)"
            return text

        entity = finding.get("entity")
        directions = [str(direction) for direction in (finding.get("directions") or []) if direction]
        sample_findings = [str(item) for item in (finding.get("sample_findings") or []) if item]
        papers = finding.get("papers") or []
        if entity or directions or sample_findings:
            label = f"Divergent evidence for {entity or 'target'}"
            direction_text = ", ".join(directions) if directions else "mixed directions"
            summary = f"{label}: {direction_text}"
            if papers:
                summary += f" across {len(papers)} studies"
            if sample_findings:
                summary += f" (e.g., {'; '.join(sample_findings[:2])})"
            return summary

        contradiction_drug = finding.get("drug")
        contradiction_detail = finding.get("details")
        if contradiction_drug and contradiction_detail:
            return f"{contradiction_drug}: {contradiction_detail}"

        return json.dumps(finding, sort_keys=True)

    @staticmethod
    def _extract_publication_year(publication_date: str | None) -> int | None:
        if not publication_date:
            return None
        match = re.search(r"\b(19|20)\d{2}\b", publication_date)
        if match is None:
            return None
        return int(match.group(0))
