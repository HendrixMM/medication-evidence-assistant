"""Configuration for the agent workflow."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from src.enhanced_config import _as_bool, _as_float, _as_int, _as_str_list, _get_env


@dataclass
class AgentConfig:
    nvidia_api_key: str | None = None
    openai_api_key: str | None = None
    llm_provider: str = "nvidia"
    planner_model: str = "70b"
    translator_model: str = "nvidia/llama-3.1-nemotron-70b-instruct"
    evaluator_model: str | None = None
    max_agent_turns: int = 3
    max_pubmed_calls: int = 5
    max_total_tokens: int = 15000
    pubmed_email: str | None = None
    pubmed_api_key: str | None = None
    pubmed_max_results_per_query: int = 15
    source_relevance_threshold: float = 3.0
    sufficient_relevant_articles: int = 5
    too_many_articles_threshold: int = 50
    enable_source_relevance_scoring: bool = True
    enable_guardrails: bool = True
    enable_session_expiry: bool = False
    safety_critical_intents: tuple[str, ...] = ("drug_interaction", "dosage")
    session_ttl_hours: int = 24
    session_dir: str = ".sessions"
    session_base_dir: str | None = None
    evaluator_top_k_articles: int = 8
    min_direct_evidence_articles: int = 2
    min_subquestion_coverage_ratio: float = 0.67
    semantic_relevance_min_score: float = 3.5

    def __post_init__(self) -> None:
        if self.max_agent_turns < 1:
            raise ValueError("max_agent_turns must be at least 1")
        if self.max_pubmed_calls < 1:
            raise ValueError("max_pubmed_calls must be at least 1")
        if self.evaluator_top_k_articles < self.sufficient_relevant_articles:
            raise ValueError(
                "evaluator_top_k_articles must be greater than or equal to sufficient_relevant_articles"
            )
        if self.evaluator_top_k_articles < self.min_direct_evidence_articles:
            raise ValueError(
                "evaluator_top_k_articles must be greater than or equal to min_direct_evidence_articles"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AgentConfig:
        values = os.environ if env is None else env
        return cls(
            nvidia_api_key=_get_env(values, "NVIDIA_API_KEY"),
            openai_api_key=_get_env(values, "OPENAI_API_KEY"),
            llm_provider=(_get_env(values, "AGENT_LLM_PROVIDER") or cls.llm_provider).lower(),
            planner_model=_get_env(values, "AGENT_PLANNER_MODEL") or cls.planner_model,
            translator_model=_get_env(values, "AGENT_TRANSLATOR_MODEL") or cls.translator_model,
            evaluator_model=_get_env(values, "AGENT_EVALUATOR_MODEL"),
            max_agent_turns=_as_int(values, "AGENT_MAX_TURNS", cls.max_agent_turns),
            max_pubmed_calls=_as_int(values, "AGENT_MAX_PUBMED_CALLS", cls.max_pubmed_calls),
            max_total_tokens=_as_int(values, "AGENT_MAX_TOTAL_TOKENS", cls.max_total_tokens),
            pubmed_email=_get_env(values, "PUBMED_EMAIL"),
            pubmed_api_key=_get_env(values, "PUBMED_EUTILS_API_KEY") or _get_env(values, "PUBMED_API_KEY"),
            pubmed_max_results_per_query=_as_int(
                values,
                "AGENT_PUBMED_MAX_RESULTS",
                cls.pubmed_max_results_per_query,
            ),
            source_relevance_threshold=_as_float(
                values,
                "AGENT_RELEVANCE_THRESHOLD",
                cls.source_relevance_threshold,
            ),
            sufficient_relevant_articles=_as_int(
                values,
                "AGENT_SUFFICIENT_ARTICLES",
                cls.sufficient_relevant_articles,
            ),
            too_many_articles_threshold=_as_int(
                values,
                "AGENT_TOO_MANY_ARTICLES",
                cls.too_many_articles_threshold,
            ),
            enable_source_relevance_scoring=_as_bool(
                values,
                "AGENT_ENABLE_RELEVANCE_SCORING",
                cls.enable_source_relevance_scoring,
            ),
            enable_guardrails=_as_bool(values, "AGENT_ENABLE_GUARDRAILS", cls.enable_guardrails),
            enable_session_expiry=_as_bool(values, "AGENT_ENABLE_SESSION_EXPIRY", cls.enable_session_expiry),
            safety_critical_intents=tuple(
                _as_str_list(values, "AGENT_SAFETY_CRITICAL_INTENTS", cls.safety_critical_intents)
            ),
            session_ttl_hours=_as_int(values, "AGENT_SESSION_TTL_HOURS", cls.session_ttl_hours),
            session_dir=_get_env(values, "AGENT_SESSION_DIR") or cls.session_dir,
            session_base_dir=_get_env(values, "AGENT_SESSION_BASE_DIR"),
            evaluator_top_k_articles=_as_int(values, "AGENT_EVALUATOR_TOP_K", cls.evaluator_top_k_articles),
            min_direct_evidence_articles=_as_int(
                values,
                "AGENT_MIN_DIRECT_EVIDENCE",
                cls.min_direct_evidence_articles,
            ),
            min_subquestion_coverage_ratio=_as_float(
                values,
                "AGENT_MIN_SUBQUESTION_COVERAGE",
                cls.min_subquestion_coverage_ratio,
            ),
            semantic_relevance_min_score=_as_float(
                values,
                "AGENT_SEMANTIC_RELEVANCE_MIN",
                cls.semantic_relevance_min_score,
            ),
        )
