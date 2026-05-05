from __future__ import annotations

from datetime import datetime
from typing import Any

from .schemas import ArticleRecord


class RankingFilter:
    _QUALITY_WEIGHTS = {
        "systematic review": 1.0,
        "meta-analysis": 1.0,
        "randomized controlled trial": 0.9,
        "clinical trial": 0.8,
        "cohort": 0.7,
        "observational study": 0.7,
        "case report": 0.3,
    }

    def filter_and_rank(
        self,
        articles: list[dict[str, Any]],
        *,
        drug_terms: list[str] | None = None,
        drug_term_groups: list[list[str]] | None = None,
        intent: str | None = None,
        require_all_groups: bool = False,
    ) -> list[ArticleRecord]:
        filtered: list[ArticleRecord] = []
        for article in articles:
            if self._is_retracted(article):
                continue
            if self._is_non_human(article):
                continue
            if self._is_non_english(article):
                continue
            if not self._matches_drug_anchor_requirement(
                article,
                drug_terms=drug_terms,
                drug_term_groups=drug_term_groups,
                intent=intent,
                require_all_groups=require_all_groups,
            ):
                continue
            filtered.append(self._to_article_record(article))
        return sorted(filtered, key=lambda item: item.heuristic_score, reverse=True)

    def _to_article_record(self, article: dict[str, Any]) -> ArticleRecord:
        publication_types = list(article.get("publication_types") or [])
        study_type = self._infer_study_type(publication_types)
        heuristic_score = self._compute_heuristic_score(study_type, article.get("publication_date"))
        authors_raw = article.get("authors") or []
        if isinstance(authors_raw, str):
            authors = [item.strip() for item in authors_raw.split(",") if item.strip()]
        else:
            authors = [str(item) for item in authors_raw]
        return ArticleRecord(
            pmid=str(article.get("pmid") or ""),
            title=article.get("title"),
            abstract=article.get("abstract"),
            authors=authors,
            journal=article.get("journal"),
            publication_date=article.get("publication_date"),
            doi=article.get("doi"),
            url=article.get("full_text_url") or article.get("url"),
            publication_types=publication_types,
            study_type=study_type,
            heuristic_score=heuristic_score,
            llm_relevance_score=None,
            llm_reason=None,
        )

    def _infer_study_type(self, publication_types: list[str]) -> str:
        lowered = " ".join(item.lower() for item in publication_types)
        for study_type in self._QUALITY_WEIGHTS:
            if study_type in lowered:
                return study_type
        return publication_types[0].lower() if publication_types else "unknown"

    def _compute_heuristic_score(self, study_type: str, publication_date: str | None) -> float:
        base = self._QUALITY_WEIGHTS.get(study_type, 0.4)
        year = None
        if publication_date:
            try:
                year = int(publication_date[:4])
            except Exception:
                year = None
        if year is None:
            return round(base, 4)
        current_year = datetime.utcnow().year
        recency_bonus = max(0.0, 0.2 - (current_year - year) * 0.01)
        return round(base + recency_bonus, 4)

    def _is_retracted(self, article: dict[str, Any]) -> bool:
        haystack = " ".join(str(item) for item in article.get("publication_types", []))
        return "retracted" in haystack.lower()

    def _is_non_human(self, article: dict[str, Any]) -> bool:
        haystack = " ".join(
            str(article.get(key) or "")
            for key in ("mesh_terms", "species", "title", "abstract")
        ).lower()
        return any(term in haystack for term in ("mice", "mouse", "rat", "rats")) and "human" not in haystack

    def _is_non_english(self, article: dict[str, Any]) -> bool:
        language = str(article.get("language") or "").lower()
        return bool(language and language not in {"english", "eng"})

    def _matches_drug_anchor_requirement(
        self,
        article: dict[str, Any],
        *,
        drug_terms: list[str] | None = None,
        drug_term_groups: list[list[str]] | None = None,
        intent: str | None = None,
        require_all_groups: bool = False,
    ) -> bool:
        groups = [
            [term for term in group if term and term.strip()]
            for group in (drug_term_groups or ([] if not drug_terms else [drug_terms]))
        ]
        if not groups:
            return True
        if require_all_groups:
            return all(self._mentions_drug_anchor(article, group) for group in groups)
        return any(self._mentions_drug_anchor(article, group) for group in groups)

    def _mentions_drug_anchor(self, article: dict[str, Any], drug_terms: list[str]) -> bool:
        haystack = " ".join(str(article.get(key) or "") for key in ("title", "abstract")).lower()
        return any(term.lower() in haystack for term in drug_terms if term and term.strip())
