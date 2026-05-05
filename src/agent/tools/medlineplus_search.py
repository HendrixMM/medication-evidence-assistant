"""MedlinePlus retrieval adapter for agent workflows."""
from __future__ import annotations

import re
import time

from src.medlineplus_client import MedlinePlusClient, MedlinePlusResult, expand_consumer_query

from ..config import AgentConfig
from ..models import PlannedQuery, RetrievalResult, RetrievedArticle


class MedlinePlusSearchTool:
    """Search MedlinePlus and map patient-education pages into agent models."""

    def __init__(
        self,
        client: MedlinePlusClient,
        max_results: int = AgentConfig.pubmed_max_results_per_query,
    ) -> None:
        self.client = client
        self.max_results = max_results

    def execute(self, query: PlannedQuery) -> RetrievalResult:
        start_time = time.perf_counter()
        query_text = expand_consumer_query(_free_text_query(query.pubmed_query))
        try:
            results = self.client.search(query_text, max_results=max(1, int(self.max_results)))
            return RetrievalResult(
                tool_name="medlineplus_search",
                query_used=query_text,
                articles=[_article_from_result(result) for result in results],
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                error=None,
            )
        except Exception as exc:
            return RetrievalResult(
                tool_name="medlineplus_search",
                query_used=query_text,
                articles=[],
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                error=str(exc),
            )


def _article_from_result(result: MedlinePlusResult) -> RetrievedArticle:
    section_parts: list[str] = []
    for section_name, values in result.sections.items():
        for value in values:
            section_parts.append(f"{section_name}: {value}")
    abstract = "\n".join(part for part in [result.summary, *section_parts] if part)
    return RetrievedArticle(
        pmid=result.identifier,
        title=result.title or None,
        abstract=abstract or None,
        authors=["MedlinePlus"],
        journal="MedlinePlus",
        publication_date=None,
        url=result.url or None,
        publication_types=["MedlinePlus"],
        ranking_score=0.0,
        relevance_score=1.0,
        source_system="medlineplus",
    )


def _free_text_query(query: str) -> str:
    """Convert legacy PubMed-like planned queries into MedlinePlus free text."""

    cleaned = query.strip()
    cleaned = re.sub(r'"([^"]+)"\[[^\]]+\]', r"\1", cleaned)
    cleaned = re.sub(r"\b(?:AND|OR|NOT)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
    cleaned = re.sub(r"\b(?:humans?|english|review|adult|child)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[(){}]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or query.strip()


__all__ = ["MedlinePlusSearchTool"]
