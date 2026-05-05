from __future__ import annotations

from dataclasses import dataclass, field

from src.medical_retrieval.schemas import (
    EvidenceStrengthHint,
    MedicalRetrievalError,
    MedicalRetrievalMeta,
)


@dataclass
class RetrievalMetrics:
    agent_passes: int = 0
    perplexity_searches: int = 0
    pubmed_fetches: int = 0
    candidate_articles: int = 0
    ranked_articles: int = 0
    pmcid_mapping_success: int = 0
    pmcid_mapping_unmapped: int = 0
    rxnorm_calls: int = 0
    cached: bool = False
    errors: list[MedicalRetrievalError] = field(default_factory=list)

    def add_error(self, layer: str, reason: str) -> None:
        self.errors.append(MedicalRetrievalError(layer=layer, reason=reason))

    def to_meta(self, evidence_strength_hint: EvidenceStrengthHint) -> MedicalRetrievalMeta:
        return MedicalRetrievalMeta(
            agent_passes=self.agent_passes,
            perplexity_searches=self.perplexity_searches,
            pubmed_fetches=self.pubmed_fetches,
            candidate_articles=self.candidate_articles,
            ranked_articles=self.ranked_articles,
            evidence_strength_hint=evidence_strength_hint,
            cached=self.cached,
            errors=list(self.errors),
        )


__all__ = ["RetrievalMetrics"]
