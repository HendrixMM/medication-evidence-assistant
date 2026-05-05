"""Agent factory — constructs a fully-wired AgentLoop from environment configuration."""
from __future__ import annotations

import os

from src.agent import AgentConfig, AgentGuardrails, AgentLoop, QueryPlanner, RetrievalEvaluator
from src.agent.retrieval_quality import RetrievalQualityScorer
from src.agent.session import SessionManager
from src.agent.tools.medlineplus_search import MedlinePlusSearchTool
from src.agent.tools.summarize_evidence import SummarizeEvidenceTool
from src.agent.tools.translate_for_consumer import TranslateForConsumerTool
from src.ddi_pk_processor import DDIPKProcessor
from src.medical_guardrails import MedicalGuardrails, resolve_medical_guardrails_config_path
from src.clients.openai_llm_client import OpenAILLMClient
from src.medlineplus_client import MedlinePlusClient
from src.nvidia_llm_client import NVIDIALLMClient
from src.ranking_filter import StudyRankingFilter
from src.source_relevance_scorer import SourceRelevanceScorer
from src.synthesis_engine import SynthesisEngine


def _normalize_translator_model(model_name: str | None, provider: str = "nvidia") -> str:
    """Map config model identifiers onto values supported by the configured provider.

    For NVIDIA, accepts short keys ("8b", "70b") or full NVIDIA model IDs, falling
    back to "70b" when unset and mapping ambiguous names via substring match. For
    OpenAI, the model name is passed through unchanged (empty -> None so the
    OpenAILLMClient default applies).
    """
    if provider == "openai":
        return (model_name or "").strip() or None  # type: ignore[return-value]

    if not model_name:
        return "70b"

    normalized = model_name.strip()
    supported = set(NVIDIALLMClient.MODELS) | set(NVIDIALLMClient.MODELS.values())
    if normalized in supported:
        return normalized

    lowered = normalized.lower()
    if "70b" in lowered:
        return "70b"
    if "8b" in lowered:
        return "8b"
    return normalized


def _build_llm_client(config: AgentConfig):
    if (config.llm_provider or "nvidia").lower() == "openai":
        return OpenAILLMClient(api_key=config.openai_api_key)
    return NVIDIALLMClient(api_key=config.nvidia_api_key)


def create_agent_loop() -> AgentLoop:
    config = AgentConfig.from_env()
    llm_client = _build_llm_client(config)
    medlineplus_client = MedlinePlusClient(email=os.getenv("MEDLINEPLUS_EMAIL") or config.pubmed_email)
    medical_guardrails = MedicalGuardrails(
        str(resolve_medical_guardrails_config_path()),
        enabled=config.enable_guardrails,
    )
    planner = QueryPlanner(llm_client=llm_client, config=config)
    pubmed_tool = MedlinePlusSearchTool(client=medlineplus_client, max_results=config.pubmed_max_results_per_query)
    relevance_scorer = (
        SourceRelevanceScorer(llm_client, threshold=config.source_relevance_threshold)
        if config.enable_source_relevance_scoring
        else None
    )
    ranking_filter = StudyRankingFilter()
    retrieval_quality_scorer = RetrievalQualityScorer(
        config=config,
        ranking_filter=ranking_filter,
        source_relevance_scorer=relevance_scorer,
    )
    evaluator = RetrievalEvaluator(config=config, retrieval_quality_scorer=retrieval_quality_scorer)
    summarize_tool = SummarizeEvidenceTool(
        ranking_filter=ranking_filter,
        synthesis_engine=SynthesisEngine(),
        ddi_pk_processor=DDIPKProcessor(),
        relevance_scorer=relevance_scorer,
        config=config,
    )
    translate_tool = TranslateForConsumerTool(
        llm_client=llm_client,
        config=config,
        model=_normalize_translator_model(config.translator_model, config.llm_provider),
    )
    guardrails = AgentGuardrails(guardrails=medical_guardrails, config=config)
    session_manager = SessionManager(config)
    return AgentLoop(
        planner=planner,
        evaluator=evaluator,
        pubmed_tool=pubmed_tool,
        summarize_tool=summarize_tool,
        translate_tool=translate_tool,
        guardrails=guardrails,
        session_manager=session_manager,
        config=config,
    )
