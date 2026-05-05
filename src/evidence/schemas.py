from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


EvidenceIntent = Literal["side_effects", "benefits", "interactions", "comparison", "dosage", "safety"]
DrugMatchType = Literal["exact", "approximate", "none"]
EvidenceStrengthHint = Literal["high", "medium", "low"]


class EvidenceRequest(BaseModel):
    question: str
    intent: EvidenceIntent
    drug_names: list[str] = Field(default_factory=list)


class NormalizedDrug(BaseModel):
    raw_name: str
    generic_name: str
    match_type: DrugMatchType
    rxnorm_cui: str | None = None


class QueryExecution(BaseModel):
    query: str
    strategy_label: str
    result_count: int = 0


class ArticleRecord(BaseModel):
    pmid: str
    title: str | None
    abstract: str | None
    authors: list[str]
    journal: str | None
    publication_date: str | None
    doi: str | None
    url: str | None
    publication_types: list[str] = Field(default_factory=list)
    study_type: str
    heuristic_score: float = 0.0
    llm_relevance_score: float | None = None
    llm_reason: str | None = None


class ErrorRecord(BaseModel):
    layer: str
    reason: str


class ResponseMeta(BaseModel):
    pubmed_calls: int
    articles_considered: int
    articles_heuristic_filtered: int
    articles_llm_scored: int
    normalization_used: bool
    normalization_complete: bool
    evidence_strength_hint: EvidenceStrengthHint
    cached: bool
    cost_estimate_usd: float
    query_budget_used: int
    query_budget_max: int
    errors: list[ErrorRecord] = Field(default_factory=list)


class EvidenceResponse(BaseModel):
    normalized_drugs: list[NormalizedDrug] = Field(default_factory=list)
    queries_executed: list[QueryExecution] = Field(default_factory=list)
    articles: list[ArticleRecord] = Field(default_factory=list)
    meta: ResponseMeta


class QueryCandidate(BaseModel):
    query: str
    strategy_label: str


class AgentChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AgentChatRequest(BaseModel):
    messages: list[AgentChatMessage]


class AgentChatUsage(BaseModel):
    pubmed_calls: int = 0
    llm_calls: int = 0
    output_tokens: int = 0
    total_latency_ms: float = 0.0


class AgentChatMeta(BaseModel):
    stop_reason: str
    usage: AgentChatUsage = Field(default_factory=AgentChatUsage)
    tool_calls: int = 0
    total_latency_ms: float = 0.0
    errors: list[ErrorRecord] = Field(default_factory=list)


class AgentChatResponse(BaseModel):
    message: AgentChatMessage
    meta: AgentChatMeta
