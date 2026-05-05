from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.medical_retrieval.pubmed_fetcher import (
    CandidateProvenance,
    NormalizedArticle,
    _provenance_to_dict,
)

DEFAULT_POOL_CANDIDATE_CAP = 80


@dataclass
class PooledCandidate:
    pmid: str
    article: NormalizedArticle
    provenance: list[CandidateProvenance] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        payload = self.article.to_tool_payload()
        payload["provenance"] = [_provenance_to_dict(p) for p in self.provenance]
        return payload


class CandidatePool:
    """Accumulates Agent-discovered articles across passes, deduped by PMID."""

    def __init__(self, max_candidates: int = DEFAULT_POOL_CANDIDATE_CAP) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        self.max_candidates = max_candidates
        self._candidates: dict[str, PooledCandidate] = {}
        self._dropped_at_cap = 0

    @property
    def size(self) -> int:
        return len(self._candidates)

    @property
    def dropped_at_cap(self) -> int:
        return self._dropped_at_cap

    def add_articles(self, articles: Iterable[NormalizedArticle]) -> None:
        for article in articles:
            self._add_article(article)

    def _add_article(self, article: NormalizedArticle) -> None:
        existing = self._candidates.get(article.pmid)
        if existing is None:
            if len(self._candidates) >= self.max_candidates:
                self._dropped_at_cap += 1
                return
            self._candidates[article.pmid] = PooledCandidate(
                pmid=article.pmid,
                article=article,
                provenance=list(article.provenance),
            )
            return
        seen = {_provenance_signature(p) for p in existing.provenance}
        for prov in article.provenance:
            if _provenance_signature(prov) not in seen:
                existing.provenance.append(prov)
                seen.add(_provenance_signature(prov))
        if existing.article.pmcid is None and article.pmcid is not None:
            existing.article.pmcid = article.pmcid

    def candidates(self) -> list[PooledCandidate]:
        return list(self._candidates.values())

    def to_payloads(self) -> list[dict[str, Any]]:
        return [c.to_payload() for c in self._candidates.values()]


def _provenance_signature(prov: CandidateProvenance) -> tuple[Any, ...]:
    return (prov.query, prov.strategy_family, prov.source_url, prov.pass_index)
