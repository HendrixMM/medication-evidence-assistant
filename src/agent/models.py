"""Typed models for the agent workflow."""
from __future__ import annotations

import enum
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class QueryIntent(str, enum.Enum):
    side_effects = "side_effects"
    drug_interaction = "drug_interaction"
    dosage = "dosage"
    timing = "timing"
    comparison = "comparison"
    mechanism = "mechanism"
    unknown = "unknown"


class EvidenceLevel(str, enum.Enum):
    systematic_review = "systematic_review"
    meta_analysis = "meta_analysis"
    rct = "rct"
    cohort = "cohort"
    case_report = "case_report"
    insufficient = "insufficient"


class AgentStopReason(str, enum.Enum):
    completed = "completed"
    max_turns = "max_turns"
    max_pubmed_calls = "max_pubmed_calls"
    budget_exhausted = "budget_exhausted"
    error = "error"
    blocked = "blocked"


class Source(BaseModel):
    pmid: str
    title: str
    authors: list[str]
    journal: str | None
    year: int | None
    url: str | None
    relevance_score: float
    study_type: str
    snippet: str | None

    model_config = ConfigDict(frozen=True)


class SubQuestion(BaseModel):
    text: str
    rationale: str
    priority: int

    model_config = ConfigDict(frozen=True)


class PlannedQuery(BaseModel):
    pubmed_query: str
    sub_question_index: int
    strategy_label: str

    model_config = ConfigDict(frozen=True)


class QueryPlan(BaseModel):
    normalized_question: str
    intent: QueryIntent
    sub_questions: list[SubQuestion]
    planned_queries: list[PlannedQuery]
    drugs_identified: list[str]
    reasoning: str

    @field_serializer("intent")
    def serialize_intent(self, value: QueryIntent) -> str:
        return value.value


class RetrievedArticle(BaseModel):
    pmid: str
    title: str | None
    abstract: str | None
    authors: list[str]
    journal: str | None
    publication_date: str | None = None
    url: str | None = None
    publication_types: list[str]
    ranking_score: float = 0.0
    relevance_score: float = 0.0
    source_system: Literal["pubmed", "medlineplus", "local_pdf"] = "pubmed"


class RetrievalResult(BaseModel):
    tool_name: str
    query_used: str
    articles: list[RetrievedArticle]
    execution_time_ms: float
    error: str | None = None


class ArticleAssessment(BaseModel):
    pmid: str
    direct_match_score: float
    semantic_relevance_score: float
    quality_score: float
    covered_subquestions: list[int]
    assessment_label: Literal["direct", "supporting", "tangential", "irrelevant"]

    model_config = ConfigDict(frozen=True)


class SubQuestionCoverage(BaseModel):
    sub_question_index: int
    sub_question_text: str
    coverage_score: float
    matched_pmids: list[str]
    covered: bool

    model_config = ConfigDict(frozen=True)


class EvaluationVerdict(BaseModel):
    is_sufficient: bool
    relevant_articles: list[RetrievedArticle]
    study_type_distribution: dict[str, int]
    coverage_gaps: list[str]
    recommendation: Literal["sufficient", "broaden", "narrow", "pivot"]
    flags: list[str]
    article_assessments: list[ArticleAssessment] = Field(default_factory=list)
    subquestion_coverages: list[SubQuestionCoverage] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    retrieval_score: float = 0.0


class EvidenceSummary(BaseModel):
    ranked_articles: list[RetrievedArticle]
    evidence_level: EvidenceLevel
    key_findings: list[str]
    convergent_findings: list[str]
    divergent_findings: list[str]
    drug_interactions: list[dict[str, Any]]
    pk_parameters: dict[str, Any]
    research_gaps: list[str]

    @field_serializer("evidence_level")
    def serialize_evidence_level(self, value: EvidenceLevel) -> str:
        return value.value


class ConsumerAnswer(BaseModel):
    answer: str
    sources: list[Source]
    evidence_level: EvidenceLevel
    uncertainties: list[str]
    safety_warnings: list[str]
    disclaimer: str
    verdict: str | None = None

    @field_serializer("evidence_level")
    def serialize_evidence_level(self, value: EvidenceLevel) -> str:
        return value.value


class UsageStats(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    pubmed_calls: int = 0
    llm_calls: int = 0
    total_latency_ms: float = 0.0


class TurnRecord(BaseModel):
    turn_number: int
    tool_invocations: list[str]
    evaluation: EvaluationVerdict | None
    tokens_used: int
    decision: str
    num_articles: int
    num_relevant: int
    study_type_counts: dict[str, int]
    latency_breakdown_ms: dict[str, float]


class StreamEvent(BaseModel):
    event_type: str
    data: dict[str, Any]
    timestamp: float = Field(default_factory=time.time)


class SessionState(BaseModel):
    session_id: str
    query: str
    plan: QueryPlan | None = None
    turns: list[TurnRecord] = Field(default_factory=list)
    all_articles: list[RetrievedArticle] = Field(default_factory=list)
    evidence_summary: EvidenceSummary | None = None
    answer: ConsumerAnswer | None = None
    usage: UsageStats = Field(default_factory=UsageStats)
    stop_reason: AgentStopReason | None = None

    @field_serializer("stop_reason")
    def serialize_stop_reason(self, value: AgentStopReason | None) -> str | None:
        return None if value is None else value.value


__all__ = [
    "ArticleAssessment",
    "AgentStopReason",
    "ConsumerAnswer",
    "EvaluationVerdict",
    "EvidenceLevel",
    "EvidenceSummary",
    "PlannedQuery",
    "QueryIntent",
    "QueryPlan",
    "RetrievalResult",
    "RetrievedArticle",
    "SessionState",
    "Source",
    "StreamEvent",
    "SubQuestion",
    "SubQuestionCoverage",
    "TurnRecord",
    "UsageStats",
]
