from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.medical_retrieval.claim_models import Claim, ValidatedClaim


EvidenceStrengthHint = Literal["high", "medium", "low", "unknown"]


class ConversationContextMessage(BaseModel):
    role: str
    content: str


class MedicalRetrievalHints(BaseModel):
    preferred_domains: list[str] = Field(default_factory=list)
    max_passes: int | None = None


class MedicalEvidenceRequest(BaseModel):
    question: str
    conversation_context: list[ConversationContextMessage] = Field(default_factory=list)
    hints: MedicalRetrievalHints = Field(default_factory=MedicalRetrievalHints)


class MedicalQuestionEntities(BaseModel):
    drugs: list[str] = Field(default_factory=list)
    supplements: list[str] = Field(default_factory=list)
    normalized_ingredients: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    populations: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)


class MedicalQuestionAnalysis(BaseModel):
    question_type: str
    entities: MedicalQuestionEntities = Field(default_factory=MedicalQuestionEntities)
    optional_drug_normalizations: list[str] = Field(default_factory=list)


class MedicalStrategyExecution(BaseModel):
    strategy_label: str
    query: str
    domains: list[str] = Field(default_factory=list)
    result_count: int = 0


class MedicalArticleRecord(BaseModel):
    pmid: str | None = None
    title: str | None = None
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    publication_date: str | None = None
    doi: str | None = None
    url: str | None = None
    source_domain: str | None = None
    evidence_strength_hint: EvidenceStrengthHint = "unknown"


class MedicalRetrievalError(BaseModel):
    layer: str
    reason: str


class MedicalRetrievalMeta(BaseModel):
    agent_passes: int
    perplexity_searches: int
    pubmed_fetches: int
    candidate_articles: int
    ranked_articles: int
    evidence_strength_hint: EvidenceStrengthHint
    cached: bool
    errors: list[MedicalRetrievalError] = Field(default_factory=list)


class MedicalEvidenceResponse(BaseModel):
    question_analysis: MedicalQuestionAnalysis
    strategies_executed: list[MedicalStrategyExecution] = Field(default_factory=list)
    articles: list[MedicalArticleRecord] = Field(default_factory=list)
    meta: MedicalRetrievalMeta
    claims: list[Claim] = Field(default_factory=list)
    validated_claims: list[ValidatedClaim] = Field(default_factory=list)
    composed_answer: str | None = None
