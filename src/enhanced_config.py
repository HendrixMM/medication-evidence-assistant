"""Central configuration management for enhanced RAG and PubMed features."""
# DEPRECATED: replaced by src/agent/config.AgentConfig
# This module is superseded by the agent layer introduced in Phase 4.
# It is retained for backward compatibility with Streamlit apps and existing tests.
# Do not add new functionality here. See src/agent/ for the active implementation.
from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import MutableMapping

_BOOL_TRUE = {"1", "true", "yes", "on", "enabled"}
_BOOL_FALSE = {"0", "false", "no", "off", "disabled"}
_ERROR_HANDLING_MODES = {"graceful", "strict", "silent"}
_DEFAULT_PUBMED_STRATEGY_WEIGHTS = (1.0, 0.8, 0.7, 0.6)


def _get_env(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key)
    return value.strip() if isinstance(value, str) else value


def _as_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = _get_env(env, key)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    raise ValueError(f"Environment variable {key} must be boolean-like, got: {raw}")


def _parse_bool(raw: str | None) -> bool:
    if raw is None:
        raise ValueError("Boolean value cannot be None")
    lowered = raw.strip().lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    raise ValueError(f"Boolean value must be boolean-like, got: {raw}")


def _as_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _get_env(env, key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be an integer, got: {raw}") from exc


def _as_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = _get_env(env, key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be numeric, got: {raw}") from exc


def _as_float_list(env: Mapping[str, str], key: str, default: Iterable[float]) -> list[float]:
    raw = _get_env(env, key)
    if raw is None:
        return [float(value) for value in default]
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return [float(value) for value in default]
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be a comma-separated list of numbers, got: {raw}") from exc


def _as_str_list(env: Mapping[str, str], key: str, default: Iterable[str]) -> list[str]:
    raw = _get_env(env, key)
    if raw is None:
        return [str(value) for value in default]
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parts if parts else [str(value) for value in default]


@dataclass
class EnhancedRAGConfig:
    """Configuration snapshot for hybrid RAG + PubMed functionality."""

    enhanced_features_enabled: bool = False
    enable_pubmed_integration: bool = False
    pubmed_hybrid_mode: bool = False
    pubmed_fallback_enabled: bool = True
    pubmed_cache_integration: bool = True

    max_external_results: int = 10
    relevance_threshold: float = 0.7
    query_timeout_seconds: int = 30
    enable_query_enhancement: bool = True
    enable_query_expansion: bool = True
    enable_direct_answer_format: bool = True
    enable_source_relevance_validation: bool = True
    source_relevance_threshold: float = 0.3
    max_query_refinement_attempts: int = 2
    refinement_sanity_window: int = 10
    max_refinement_time_ms: float = 10000.0
    pubmed_multi_strategy_enabled: bool = False
    pubmed_strategy_weights: list[float] = field(default_factory=lambda: list(_DEFAULT_PUBMED_STRATEGY_WEIGHTS))
    pubmed_strategy_max_results_per: int = 5
    pubmed_strategy_parallel_timeout: int = 30

    # LLM-guided query generation configuration
    enable_llm_query_generation: bool = False
    llm_query_generation_model: str = "gpt-4o-mini"
    llm_query_temperature: float = 0.2
    llm_query_max_tokens: int = 350
    llm_query_timeout_seconds: int = 10
    llm_query_success_gate_enabled: bool = True
    llm_query_success_gate_path: str = "evaluation/results/llm_query_comparison_llm.json"
    llm_query_min_source_relevance: float = 4.5
    llm_query_min_pass_rate: float = 0.97

    # LLM post-retrieval source filtering
    enable_llm_source_filtering: bool = False
    llm_source_relevance_threshold: float = 3.0
    llm_source_relevance_max_sources: int = 10

    enable_status_reporting: bool = True
    enable_metrics_collection: bool = True
    enable_performance_logging: bool = False
    enable_dashboard_integration: bool = True

    gradual_rollout_enabled: bool = False
    rollout_percentage: float = 0.0
    beta_mode: bool = False

    safe_mode: bool = True
    force_70b_for_safety_critical: bool = True
    safety_critical_intents: list[str] = field(default_factory=lambda: ["drug_interaction", "dosage"])
    error_handling_strategy: str = "graceful"
    fallback_to_local_on_error: bool = True

    enable_enhanced_pubmed_scraper: bool = False
    enable_advanced_caching: bool = True
    use_normalized_cache_keys: bool = False
    mirror_legacy_cache: bool = False
    prune_legacy_after_migration: bool = False

    enable_rate_limiting: bool = True
    rate_limit_window_seconds: int = 1
    rate_limit_max_requests: int = 3

    cache_prefetch_on_startup: bool = False
    cache_cleanup_on_startup: bool = False

    # NIM-specific configuration
    enable_nim_health_gate: bool = True
    health_check_timeout_seconds: int = 10
    max_health_latency_ms: float = 5000.0
    health_check_retry_attempts: int = 2

    # Model overrides
    embedding_model_override: str | None = None
    reranking_model_override: str | None = None
    extraction_model_override: str | None = None

    # Endpoint overrides
    custom_embedding_endpoint: str | None = None
    custom_reranking_endpoint: str | None = None
    custom_extraction_endpoint: str | None = None

    # NVIDIA Build platform integration
    enable_nvidia_build_fallback: bool = False
    nvidia_build_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_build_embedding_model: str | None = None
    nvidia_build_llm_model: str | None = None
    nvidia_build_embedding_input_type: str | None = None
    enable_nvidia_build_embedding_input_type: bool = True

    # Ollama local inference
    enable_ollama: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str | None = None
    ollama_embed_model: str | None = None
    ollama_timeout_seconds: int = 60

    # Fallback order configuration
    fallback_order: str = "nvidia_build,nemo,ollama"

    # Cloud-first rerank configuration
    enable_cloud_first_rerank: bool = True
    rerank_retry_backoff_base: float = 0.5
    rerank_retry_max_attempts: int = 3
    rerank_retry_jitter: bool = True

    # Pharmaceutical research feature flags
    enable_drug_interaction_analysis: bool = True
    enable_clinical_trial_processing: bool = True
    enable_pharmacokinetics_optimization: bool = True

    # Pharmaceutical cost optimization settings
    research_project_budgeting: bool = True
    research_project_budget_limit_usd: float = 0.0
    cost_per_query_tracking: bool = True
    pharma_project_id: str | None = None

    # Pharmaceutical model preferences (chat)
    pharma_model_chat_drug_interaction: str | None = None
    pharma_model_chat_pharmacokinetics: str | None = None
    pharma_model_chat_clinical_trial: str | None = None

    # Pharmaceutical performance + batch processing
    pharma_batch_max_size: int = 16
    pharma_batch_max_latency_ms: int = 500

    # NVIDIA Build API compatibility
    prefer_responses_api: bool = True

    # Pharmaceutical research compliance & QA
    pharma_compliance_mode: bool = True
    pharma_require_disclaimer: bool = True
    pharma_region: str = "US"
    pharma_quality_assurance_enabled: bool = True
    pharma_medical_terminology_validation: bool = True
    pharma_workflow_templates_enabled: bool = True
    pharma_specialized_metrics_enabled: bool = True

    # Global pharmaceutical research mode (optional, env-driven)
    # Allows toggling pharma-focused behavior across components when supported
    pharmaceutical_research_mode: bool = True

    config_loaded_at: float = field(default_factory=time.time)
    source_env: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> EnhancedRAGConfig:
        """Build a configuration snapshot from environment variables."""
        env_map: Mapping[str, str] = env or os.environ

        try:
            enhanced_features_enabled = _as_bool(env_map, "RAG_ENHANCED_FEATURES_ENABLED", False)
            enable_pubmed_integration = _as_bool(env_map, "ENABLE_RAG_PUBMED_INTEGRATION", False)
            pubmed_hybrid_mode = _as_bool(env_map, "RAG_PUBMED_HYBRID_MODE", False)
            pubmed_fallback_enabled = _as_bool(env_map, "RAG_PUBMED_FALLBACK_ENABLED", True)
            pubmed_cache_integration = _as_bool(env_map, "RAG_PUBMED_CACHE_INTEGRATION", True)

            max_external_results = _as_int(env_map, "RAG_PUBMED_MAX_EXTERNAL_RESULTS", 10)
            relevance_threshold = _as_float(env_map, "RAG_PUBMED_RELEVANCE_THRESHOLD", 0.7)
            query_timeout_seconds = _as_int(env_map, "RAG_PUBMED_QUERY_TIMEOUT_SECONDS", 30)
            enable_query_enhancement = _as_bool(env_map, "RAG_PUBMED_ENABLE_QUERY_ENHANCEMENT", True)
            enable_query_expansion = _parse_bool(env_map.get("ENABLE_QUERY_EXPANSION", "true"))
            enable_direct_answer_format = _parse_bool(env_map.get("ENABLE_DIRECT_ANSWER_FORMAT", "true"))
            enable_source_relevance_validation = _parse_bool(env_map.get("ENABLE_SOURCE_RELEVANCE_VALIDATION", "true"))
            source_relevance_threshold = _as_float(env_map, "SOURCE_RELEVANCE_THRESHOLD", 0.3)
            max_query_refinement_attempts = _as_int(env_map, "MAX_QUERY_REFINEMENT_ATTEMPTS", 2)
            max_refinement_time_ms = _as_float(env_map, "MAX_REFINEMENT_TIME_MS", 10000.0)
            refinement_sanity_window = _as_int(env_map, "REFINEMENT_SANITY_WINDOW", 10)
            pubmed_multi_strategy_enabled = _as_bool(env_map, "PUBMED_MULTI_STRATEGY_ENABLED", False)
            pubmed_strategy_weights = _as_float_list(
                env_map,
                "PUBMED_STRATEGY_WEIGHTS",
                _DEFAULT_PUBMED_STRATEGY_WEIGHTS,
            )
            pubmed_strategy_max_results_per = _as_int(env_map, "PUBMED_STRATEGY_MAX_RESULTS_PER", 5)
            pubmed_strategy_parallel_timeout = _as_int(env_map, "PUBMED_STRATEGY_PARALLEL_TIMEOUT", 30)

            # LLM query generation configuration
            enable_llm_query_generation = _as_bool(env_map, "ENABLE_LLM_QUERY_GENERATION", False)
            llm_query_generation_model = _get_env(env_map, "LLM_QUERY_GENERATION_MODEL") or "gpt-4o-mini"
            llm_query_temperature = _as_float(env_map, "LLM_QUERY_TEMPERATURE", 0.2)
            llm_query_max_tokens = _as_int(env_map, "LLM_QUERY_MAX_TOKENS", 350)
            llm_query_timeout_seconds = _as_int(env_map, "LLM_QUERY_TIMEOUT_SECONDS", 10)
            llm_query_success_gate_enabled = _as_bool(env_map, "LLM_QUERY_SUCCESS_GATE_ENABLED", True)
            llm_query_success_gate_path = (
                _get_env(env_map, "LLM_QUERY_SUCCESS_GATE_PATH") or "evaluation/results/llm_query_comparison_llm.json"
            )
            llm_query_min_source_relevance = _as_float(env_map, "LLM_QUERY_MIN_SOURCE_RELEVANCE", 4.5)
            llm_query_min_pass_rate = _as_float(env_map, "LLM_QUERY_MIN_PASS_RATE", 0.97)

            enable_llm_source_filtering = _as_bool(env_map, "ENABLE_LLM_SOURCE_FILTERING", False)
            llm_source_relevance_threshold = _as_float(env_map, "LLM_SOURCE_RELEVANCE_THRESHOLD", 3.0)
            llm_source_relevance_max_sources = _as_int(env_map, "LLM_SOURCE_RELEVANCE_MAX_SOURCES", 10)

            enable_status_reporting = _as_bool(env_map, "ENABLE_RAG_PUBMED_STATUS_REPORTING", True)
            enable_metrics_collection = _as_bool(env_map, "RAG_PUBMED_METRICS_COLLECTION", True)
            enable_performance_logging = _as_bool(env_map, "RAG_PUBMED_PERFORMANCE_LOGGING", False)
            enable_dashboard_integration = _as_bool(env_map, "RAG_PUBMED_DASHBOARD_INTEGRATION", True)

            gradual_rollout_enabled = _as_bool(env_map, "RAG_PUBMED_GRADUAL_ROLLOUT", False)
            rollout_percentage = _as_float(env_map, "RAG_PUBMED_ROLLOUT_PERCENTAGE", 0.0)
            beta_mode = _as_bool(env_map, "RAG_PUBMED_BETA_MODE", False)

            safe_mode = _as_bool(env_map, "RAG_PUBMED_SAFE_MODE", True)
            force_70b_for_safety_critical = _parse_bool(env_map.get("FORCE_70B_FOR_SAFETY_CRITICAL", "true"))
            safety_critical_intents = _as_str_list(
                env_map,
                "SAFETY_CRITICAL_INTENTS",
                ["drug_interaction", "dosage"],
            )
            error_handling_strategy = _get_env(env_map, "RAG_PUBMED_ERROR_HANDLING") or "graceful"
            fallback_to_local_on_error = _as_bool(env_map, "RAG_PUBMED_FALLBACK_TO_LOCAL", True)

            enable_enhanced_pubmed_scraper = _as_bool(env_map, "ENABLE_ENHANCED_PUBMED_SCRAPER", False)
            enable_advanced_caching = _as_bool(env_map, "ENABLE_ADVANCED_CACHING", True)
            use_normalized_cache_keys = _as_bool(env_map, "USE_NORMALIZED_CACHE_KEYS", False)
            mirror_legacy_cache = _as_bool(env_map, "MIRROR_TO_LEGACY_ROOT", False)
            prune_legacy_after_migration = _as_bool(env_map, "PRUNE_LEGACY_AFTER_MIGRATION", False)

            # Support both ENABLE_PUBMED_RATE_LIMITING and ENABLE_RATE_LIMITING for compatibility
            enable_rate_limiting = _as_bool(
                env_map, "ENABLE_RATE_LIMITING", _as_bool(env_map, "ENABLE_PUBMED_RATE_LIMITING", True)
            )
            rate_limit_window_seconds = _as_int(env_map, "PUBMED_RATE_LIMIT_WINDOW_SECONDS", 1)
            # Support both PUBMED_RATE_LIMIT_MAX_REQUESTS and MAX_REQUESTS_PER_SECOND for compatibility
            rate_limit_max_requests = _as_int(
                env_map, "MAX_REQUESTS_PER_SECOND", _as_int(env_map, "PUBMED_RATE_LIMIT_MAX_REQUESTS", 3)
            )

            cache_prefetch_on_startup = _as_bool(env_map, "RAG_PUBMED_CACHE_PREFETCH", False)
            cache_cleanup_on_startup = _as_bool(env_map, "RAG_PUBMED_CACHE_CLEANUP", False)

            # NIM / NEMO specific
            enable_nim_health_gate = _as_bool(env_map, "ENABLE_NIM_HEALTH_GATE", True)
            health_check_timeout_seconds = _as_int(env_map, "HEALTH_CHECK_TIMEOUT_SECONDS", 10)
            # Support both HEALTH_LATENCY_MS and HEALTH_LATENCY_MS (legacy naming)
            max_health_latency_ms = _as_float(env_map, "HEALTH_LATENCY_MS", 5000.0)
            health_check_retry_attempts = _as_int(env_map, "HEALTH_CHECK_RETRY_ATTEMPTS", 2)

            embedding_model_override = _get_env(env_map, "EMBEDDING_MODEL")
            reranking_model_override = _get_env(env_map, "RERANK_MODEL")
            extraction_model_override = _get_env(env_map, "EXTRACTION_MODEL")

            custom_embedding_endpoint = _get_env(env_map, "NEMO_EMBEDDING_ENDPOINT")
            custom_reranking_endpoint = _get_env(env_map, "NEMO_RERANKING_ENDPOINT")
            custom_extraction_endpoint = _get_env(env_map, "NEMO_EXTRACTION_ENDPOINT")

            # NVIDIA Build platform integration
            enable_nvidia_build_fallback = _as_bool(env_map, "ENABLE_NVIDIA_BUILD_FALLBACK", False)
            nvidia_build_base_url = _get_env(env_map, "NVIDIA_BUILD_BASE_URL") or "https://integrate.api.nvidia.com/v1"
            nvidia_build_embedding_model = _get_env(env_map, "NVIDIA_BUILD_EMBEDDING_MODEL")
            nvidia_build_llm_model = _get_env(env_map, "NVIDIA_BUILD_LLM_MODEL")
            nvidia_build_embedding_input_type = _get_env(env_map, "NVIDIA_BUILD_EMBEDDING_INPUT_TYPE") or "query"
            enable_nvidia_build_embedding_input_type = _as_bool(
                env_map, "ENABLE_NVIDIA_BUILD_EMBEDDING_INPUT_TYPE", True
            )

            # Ollama local inference
            enable_ollama = _as_bool(env_map, "OLLAMA_ENABLED", False)
            ollama_base_url = _get_env(env_map, "OLLAMA_BASE_URL") or "http://localhost:11434"
            ollama_chat_model = _get_env(env_map, "OLLAMA_CHAT_MODEL")
            ollama_embed_model = _get_env(env_map, "OLLAMA_EMBED_MODEL")
            ollama_timeout_seconds = _as_int(env_map, "OLLAMA_TIMEOUT_SECONDS", 60)

            # Fallback order configuration
            fallback_order = _get_env(env_map, "FALLBACK_ORDER") or "nvidia_build,nemo,ollama"

            # Cloud-first rerank configuration
            enable_cloud_first_rerank = _as_bool(env_map, "ENABLE_CLOUD_FIRST_RERANK", True)
            rerank_retry_backoff_base = _as_float(env_map, "RERANK_RETRY_BACKOFF_BASE", 0.5)
            rerank_retry_max_attempts = _as_int(env_map, "RERANK_RETRY_MAX_ATTEMPTS", 3)
            rerank_retry_jitter = _as_bool(env_map, "RERANK_RETRY_JITTER", True)

            # NVIDIA Build API compatibility
            prefer_responses_api = _as_bool(env_map, "PREFER_RESPONSES_API", True)

            # Pharmaceutical research feature flags
            enable_drug_interaction_analysis = _as_bool(
                env_map, "PHARMACEUTICAL_FEATURE_DRUG_INTERACTION_ANALYSIS", True
            )
            enable_clinical_trial_processing = _as_bool(
                env_map, "PHARMACEUTICAL_FEATURE_CLINICAL_TRIAL_PROCESSING", True
            )
            enable_pharmacokinetics_optimization = _as_bool(
                env_map, "PHARMACEUTICAL_FEATURE_PHARMACOKINETICS_OPTIMIZATION", True
            )

            # Pharmaceutical cost optimization
            research_project_budgeting = _as_bool(env_map, "PHARMA_RESEARCH_PROJECT_BUDGETING", True)
            research_project_budget_limit_usd = _as_float(env_map, "PHARMA_BUDGET_LIMIT_USD", 0.0)
            cost_per_query_tracking = _as_bool(env_map, "PHARMA_COST_PER_QUERY_TRACKING", True)
            pharma_project_id = _get_env(env_map, "PHARMA_PROJECT_ID")

            # Pharmaceutical model preferences (chat)
            pharma_model_chat_drug_interaction = _get_env(env_map, "PHARMA_MODEL_CHAT_DRUG_INTERACTION")
            pharma_model_chat_pharmacokinetics = _get_env(env_map, "PHARMA_MODEL_CHAT_PHARMACOKINETICS")
            pharma_model_chat_clinical_trial = _get_env(env_map, "PHARMA_MODEL_CHAT_CLINICAL_TRIAL")

            # Pharmaceutical performance + batch
            pharma_batch_max_size = _as_int(env_map, "PHARMA_BATCH_MAX_SIZE", 16)
            pharma_batch_max_latency_ms = _as_int(env_map, "PHARMA_BATCH_MAX_LATENCY_MS", 500)

            # Pharmaceutical compliance & QA
            pharma_compliance_mode = _as_bool(env_map, "PHARMA_COMPLIANCE_MODE", True)
            pharma_require_disclaimer = _as_bool(env_map, "PHARMA_REQUIRE_DISCLAIMER", True)
            pharma_region = _get_env(env_map, "PHARMA_REGION") or "US"
            pharma_quality_assurance_enabled = _as_bool(env_map, "PHARMA_QUALITY_ASSURANCE_ENABLED", True)
            pharma_medical_terminology_validation = _as_bool(env_map, "PHARMA_MEDICAL_TERMINOLOGY_VALIDATION", True)
            pharma_workflow_templates_enabled = _as_bool(env_map, "PHARMA_WORKFLOW_TEMPLATES_ENABLED", True)
            pharma_specialized_metrics_enabled = _as_bool(env_map, "PHARMA_SPECIALIZED_METRICS_ENABLED", True)
            # Optional global pharma research mode toggle
            pharmaceutical_research_mode = _as_bool(env_map, "PHARMACEUTICAL_RESEARCH_MODE", True)
        except ValueError as exc:
            config = cls()
            config.errors.append(str(exc))
            return config

        config = cls(
            enhanced_features_enabled=enhanced_features_enabled,
            enable_pubmed_integration=enable_pubmed_integration,
            pubmed_hybrid_mode=pubmed_hybrid_mode,
            pubmed_fallback_enabled=pubmed_fallback_enabled,
            pubmed_cache_integration=pubmed_cache_integration,
            max_external_results=max_external_results,
            relevance_threshold=relevance_threshold,
            query_timeout_seconds=query_timeout_seconds,
            enable_query_enhancement=enable_query_enhancement,
            enable_query_expansion=enable_query_expansion,
            enable_direct_answer_format=enable_direct_answer_format,
            enable_source_relevance_validation=enable_source_relevance_validation,
            source_relevance_threshold=source_relevance_threshold,
            max_query_refinement_attempts=max_query_refinement_attempts,
            max_refinement_time_ms=max_refinement_time_ms,
            refinement_sanity_window=refinement_sanity_window,
            pubmed_multi_strategy_enabled=pubmed_multi_strategy_enabled,
            pubmed_strategy_weights=pubmed_strategy_weights,
            pubmed_strategy_max_results_per=pubmed_strategy_max_results_per,
            pubmed_strategy_parallel_timeout=pubmed_strategy_parallel_timeout,
            enable_llm_query_generation=enable_llm_query_generation,
            llm_query_generation_model=llm_query_generation_model,
            llm_query_temperature=llm_query_temperature,
            llm_query_max_tokens=llm_query_max_tokens,
            llm_query_timeout_seconds=llm_query_timeout_seconds,
            llm_query_success_gate_enabled=llm_query_success_gate_enabled,
            llm_query_success_gate_path=llm_query_success_gate_path,
            llm_query_min_source_relevance=llm_query_min_source_relevance,
            llm_query_min_pass_rate=llm_query_min_pass_rate,
            enable_llm_source_filtering=enable_llm_source_filtering,
            llm_source_relevance_threshold=llm_source_relevance_threshold,
            llm_source_relevance_max_sources=llm_source_relevance_max_sources,
            enable_status_reporting=enable_status_reporting,
            enable_metrics_collection=enable_metrics_collection,
            enable_performance_logging=enable_performance_logging,
            enable_dashboard_integration=enable_dashboard_integration,
            gradual_rollout_enabled=gradual_rollout_enabled,
            rollout_percentage=rollout_percentage,
            beta_mode=beta_mode,
            safe_mode=safe_mode,
            force_70b_for_safety_critical=force_70b_for_safety_critical,
            safety_critical_intents=safety_critical_intents,
            error_handling_strategy=error_handling_strategy,
            fallback_to_local_on_error=fallback_to_local_on_error,
            enable_enhanced_pubmed_scraper=enable_enhanced_pubmed_scraper,
            enable_advanced_caching=enable_advanced_caching,
            use_normalized_cache_keys=use_normalized_cache_keys,
            mirror_legacy_cache=mirror_legacy_cache,
            prune_legacy_after_migration=prune_legacy_after_migration,
            enable_rate_limiting=enable_rate_limiting,
            rate_limit_window_seconds=rate_limit_window_seconds,
            rate_limit_max_requests=rate_limit_max_requests,
            cache_prefetch_on_startup=cache_prefetch_on_startup,
            cache_cleanup_on_startup=cache_cleanup_on_startup,
            source_env=dict(env_map),
            enable_nim_health_gate=enable_nim_health_gate,
            health_check_timeout_seconds=health_check_timeout_seconds,
            max_health_latency_ms=max_health_latency_ms,
            health_check_retry_attempts=health_check_retry_attempts,
            embedding_model_override=embedding_model_override,
            reranking_model_override=reranking_model_override,
            extraction_model_override=extraction_model_override,
            custom_embedding_endpoint=custom_embedding_endpoint,
            custom_reranking_endpoint=custom_reranking_endpoint,
            custom_extraction_endpoint=custom_extraction_endpoint,
            enable_nvidia_build_fallback=enable_nvidia_build_fallback,
            nvidia_build_base_url=nvidia_build_base_url,
            nvidia_build_embedding_model=nvidia_build_embedding_model,
            nvidia_build_llm_model=nvidia_build_llm_model,
            nvidia_build_embedding_input_type=nvidia_build_embedding_input_type,
            enable_nvidia_build_embedding_input_type=enable_nvidia_build_embedding_input_type,
            enable_ollama=enable_ollama,
            ollama_base_url=ollama_base_url,
            ollama_chat_model=ollama_chat_model,
            ollama_embed_model=ollama_embed_model,
            ollama_timeout_seconds=ollama_timeout_seconds,
            fallback_order=fallback_order,
            enable_cloud_first_rerank=enable_cloud_first_rerank,
            rerank_retry_backoff_base=rerank_retry_backoff_base,
            rerank_retry_max_attempts=rerank_retry_max_attempts,
            rerank_retry_jitter=rerank_retry_jitter,
            prefer_responses_api=prefer_responses_api,
            enable_drug_interaction_analysis=enable_drug_interaction_analysis,
            enable_clinical_trial_processing=enable_clinical_trial_processing,
            enable_pharmacokinetics_optimization=enable_pharmacokinetics_optimization,
            research_project_budgeting=research_project_budgeting,
            research_project_budget_limit_usd=research_project_budget_limit_usd,
            cost_per_query_tracking=cost_per_query_tracking,
            pharma_project_id=pharma_project_id,
            pharma_model_chat_drug_interaction=pharma_model_chat_drug_interaction,
            pharma_model_chat_pharmacokinetics=pharma_model_chat_pharmacokinetics,
            pharma_model_chat_clinical_trial=pharma_model_chat_clinical_trial,
            pharma_batch_max_size=pharma_batch_max_size,
            pharma_batch_max_latency_ms=pharma_batch_max_latency_ms,
            pharma_compliance_mode=pharma_compliance_mode,
            pharma_require_disclaimer=pharma_require_disclaimer,
            pharma_region=pharma_region,
            pharma_quality_assurance_enabled=pharma_quality_assurance_enabled,
            pharma_medical_terminology_validation=pharma_medical_terminology_validation,
            pharma_workflow_templates_enabled=pharma_workflow_templates_enabled,
            pharma_specialized_metrics_enabled=pharma_specialized_metrics_enabled,
            pharmaceutical_research_mode=pharmaceutical_research_mode,
        )
        # Ensure latest environment snapshot is available for feature flags
        try:
            config.source_env = dict(env_map)
        except Exception:
            pass
        config._validate()
        return config

    # ------------------------------------------------------------------
    # Validation and mutation helpers
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        self.warnings.clear()
        self.errors.clear()

        if not 0.0 <= self.relevance_threshold <= 1.0:
            self.warnings.append("Relevance threshold must be between 0 and 1; clamping to valid range.")
            self.relevance_threshold = max(0.0, min(1.0, self.relevance_threshold))

        if self.max_external_results < 0:
            self.warnings.append("Max external results cannot be negative; using 0.")
            self.max_external_results = 0

        if self.rollout_percentage < 0:
            self.warnings.append("Rollout percentage cannot be negative; using 0.")
            self.rollout_percentage = 0.0
        elif self.rollout_percentage > 100.0:
            self.warnings.append("Rollout percentage cannot exceed 100; capping to 100.")
            self.rollout_percentage = 100.0

        if self.query_timeout_seconds <= 0:
            self.warnings.append("Query timeout must be positive; defaulting to 30 seconds.")
            self.query_timeout_seconds = 30
        if self.pubmed_strategy_max_results_per <= 0:
            self.warnings.append("PubMed strategy max results must be positive; defaulting to 5.")
            self.pubmed_strategy_max_results_per = 5
        if self.pubmed_strategy_parallel_timeout <= 0:
            self.warnings.append("PubMed strategy timeout must be positive; defaulting to 30 seconds.")
            self.pubmed_strategy_parallel_timeout = 30
        if len(self.pubmed_strategy_weights) not in [3, 4] or any(
            weight <= 0 for weight in self.pubmed_strategy_weights
        ):
            self.warnings.append("PubMed strategy weights must be 3 or 4 positive numbers; resetting to defaults.")
            self.pubmed_strategy_weights = list(_DEFAULT_PUBMED_STRATEGY_WEIGHTS)
        if self.llm_query_temperature < 0.0 or self.llm_query_temperature > 1.0:
            self.warnings.append("LLM query temperature must be 0.0-1.0; clamping to valid range.")
            self.llm_query_temperature = max(0.0, min(1.0, self.llm_query_temperature))
        if self.llm_query_max_tokens <= 0:
            self.warnings.append("LLM query max tokens must be positive; defaulting to 350.")
            self.llm_query_max_tokens = 350
        if self.llm_query_timeout_seconds <= 0:
            self.warnings.append("LLM query timeout must be positive; defaulting to 10 seconds.")
            self.llm_query_timeout_seconds = 10

        if self.rate_limit_window_seconds <= 0:
            self.warnings.append("Rate limit window must be positive; defaulting to 1 second.")
            self.rate_limit_window_seconds = 1

        if self.rate_limit_max_requests <= 0:
            self.warnings.append("Rate limit max requests must be positive; defaulting to 1.")
            self.rate_limit_max_requests = 1

        if self.error_handling_strategy not in _ERROR_HANDLING_MODES:
            self.warnings.append(
                "Unknown error handling mode '%s'; defaulting to 'graceful'." % self.error_handling_strategy
            )
            self.error_handling_strategy = "graceful"

        if self.force_70b_for_safety_critical:
            if not self.safety_critical_intents:
                self.warnings.append(
                    "Safety-critical intents list is empty while force_70b_for_safety_critical is enabled; "
                    "defaulting to drug_interaction,dosage."
                )
                self.safety_critical_intents = ["drug_interaction", "dosage"]
        else:
            env_label = (
                (self.source_env.get("ENVIRONMENT") or self.source_env.get("APP_ENV") or self.source_env.get("RAG_ENV"))
                or ""
            ).lower()
            if env_label in {"prod", "production"}:
                self.warnings.append(
                    "force_70b_for_safety_critical is disabled in production; consider enabling for safety."
                )

        # Basic sanity for health check settings
        if self.health_check_timeout_seconds <= 0:
            self.warnings.append("Health check timeout must be positive; defaulting to 10 seconds.")
            self.health_check_timeout_seconds = 10
        if self.max_health_latency_ms < 0:
            self.warnings.append("Max health latency must be non-negative; defaulting to 5000ms.")
            self.max_health_latency_ms = 5000.0
        if self.health_check_retry_attempts < 0:
            self.warnings.append("Health check retries cannot be negative; using 0.")
            self.health_check_retry_attempts = 0

        if not self.enable_pubmed_integration:
            if self.pubmed_hybrid_mode:
                self.warnings.append("Hybrid mode requested without PubMed integration; disabling hybrid mode.")
                self.pubmed_hybrid_mode = False
            if self.pubmed_cache_integration:
                self.warnings.append(
                    "Cache integration requested without PubMed integration; disabling cache integration."
                )
                self.pubmed_cache_integration = False

        if not self.enable_enhanced_pubmed_scraper and self.enable_advanced_caching:
            self.warnings.append(
                "Advanced caching requested but enhanced scraper disabled; disabling advanced caching."
            )
            self.enable_advanced_caching = False
            self.use_normalized_cache_keys = False
            self.mirror_legacy_cache = False
            self.prune_legacy_after_migration = False

        if self.enable_advanced_caching and not self.pubmed_cache_integration:
            self.warnings.append(
                "Advanced caching enabled without RAG cache integration; consider enabling integration for best results."
            )

        if not self.enhanced_features_enabled and (
            self.enable_pubmed_integration or self.pubmed_hybrid_mode or self.beta_mode
        ):
            self.warnings.append(
                "Specific PubMed features requested while master switch is off; features remain disabled."
            )

        self.validate_refinement_config()

    def validate_refinement_config(self) -> None:
        """Validate source relevance validation configuration."""
        import logging

        logger = logging.getLogger(__name__)
        if self.enable_source_relevance_validation:
            if not 0.0 <= self.source_relevance_threshold <= 1.0:
                raise ValueError(
                    "SOURCE_RELEVANCE_THRESHOLD must be between 0.0 and 1.0, got " f"{self.source_relevance_threshold}"
                )
        if self.source_relevance_threshold < 0.1:
            logger.warning(
                "SOURCE_RELEVANCE_THRESHOLD is very low (%.2f). This may trigger excessive refinements.",
                self.source_relevance_threshold,
            )
        if self.source_relevance_threshold > 0.7:
            logger.warning(
                "SOURCE_RELEVANCE_THRESHOLD is very high (%.2f). Refinement may rarely trigger.",
                self.source_relevance_threshold,
            )
        if not 1.0 <= self.llm_source_relevance_threshold <= 5.0:
            raise ValueError(
                "LLM_SOURCE_RELEVANCE_THRESHOLD must be between 1.0 and 5.0, got "
                f"{self.llm_source_relevance_threshold}"
            )
        if self.llm_source_relevance_max_sources <= 0:
            logger.warning("LLM_SOURCE_RELEVANCE_MAX_SOURCES must be positive; defaulting to 10.")
            self.llm_source_relevance_max_sources = 10
        max_attempts = getattr(self, "max_query_refinement_attempts", 2)
        if not isinstance(max_attempts, int):
            try:
                max_attempts = int(max_attempts)
            except (TypeError, ValueError):
                logger.warning("max_query_refinement_attempts must be an int; defaulting to 2.")
                max_attempts = 2
            self.max_query_refinement_attempts = max_attempts
        if max_attempts < 0 or max_attempts > 5:
            logger.warning(
                "max_query_refinement_attempts should be 0-5, got %d. Clamping to valid range.",
                max_attempts,
            )
            self.max_query_refinement_attempts = max(0, min(5, max_attempts))
        max_time_ms = getattr(self, "max_refinement_time_ms", 10000.0)
        if not isinstance(max_time_ms, (int, float)):
            try:
                max_time_ms = float(max_time_ms)
            except (TypeError, ValueError):
                logger.warning("max_refinement_time_ms must be numeric; defaulting to 10000.0ms.")
                max_time_ms = 10000.0
            self.max_refinement_time_ms = max_time_ms
        if max_time_ms <= 0:
            logger.warning("max_refinement_time_ms must be positive; defaulting to 10000.0ms.")
            self.max_refinement_time_ms = 10000.0
        sanity_window = getattr(self, "refinement_sanity_window", 0)
        if not isinstance(sanity_window, int):
            try:
                sanity_window = int(sanity_window)
            except (TypeError, ValueError):
                logger.warning("refinement_sanity_window must be an int; defaulting to 0.")
                sanity_window = 0
            self.refinement_sanity_window = sanity_window
        if sanity_window < 0 or sanity_window > 1000:
            logger.warning(
                "refinement_sanity_window should be 0-1000, got %d. Clamping to valid range.",
                sanity_window,
            )
            self.refinement_sanity_window = max(0, min(1000, sanity_window))

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------
    def to_dict(self, include_env: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_env:
            data.pop("source_env", None)
        return data

    def export_public_view(self) -> dict[str, Any]:
        data = self.to_dict(include_env=False)
        data.pop("warnings", None)
        data.pop("errors", None)
        data.pop("_lock", None)
        return data

    def should_enable_pubmed(self) -> bool:
        return bool(self.enhanced_features_enabled and self.enable_pubmed_integration and not self.errors)

    def should_use_hybrid_mode(self) -> bool:
        return bool(self.should_enable_pubmed() and self.pubmed_hybrid_mode)

    def is_rollout_active(self) -> bool:
        return bool(self.gradual_rollout_enabled and self.rollout_percentage > 0)

    def reload(self, env: Mapping[str, str] | None = None) -> None:
        new_config = self.from_env(env)
        with self._lock:
            for key, value in asdict(new_config).items():
                if key == "_lock":
                    continue
                setattr(self, key, value)
            try:
                self.source_env = dict(env or os.environ)
            except Exception:
                pass
            self._validate()
            self.config_loaded_at = time.time()

    def apply_overrides(self, overrides: MutableMapping[str, Any], env: Mapping[str, str] | None = None) -> None:
        """
        Apply configuration overrides.

        Args:
            overrides: Dictionary of configuration field overrides
            env: Optional environment mapping to refresh source_env for env-driven flags.
                If not provided, source_env is not updated. Use reload() for full env refresh.

        Note:
            When overrides depend on environment-driven feature flags, consider providing
            the env parameter or use reload() instead for complete environment refresh.
        """
        with self._lock:
            for key, value in overrides.items():
                if not hasattr(self, key):
                    raise AttributeError(f"Unknown configuration field: {key}")
                setattr(self, key, value)

            # Optionally refresh source_env for env-driven flags
            if env is not None:
                try:
                    self.source_env = dict(env)
                except Exception:
                    pass

            self._validate()
            self.config_loaded_at = time.time()

    def summarize_flags(self) -> dict[str, Any]:
        return {
            "enabled": self.should_enable_pubmed(),
            "hybrid": self.should_use_hybrid_mode(),
            "gradual_rollout": self.is_rollout_active(),
            "beta_mode": self.beta_mode,
            "safe_mode": self.safe_mode,
            "status_reporting": self.enable_status_reporting,
        }

    # ------------------------------------------------------------------
    # NIM-native helpers
    # ------------------------------------------------------------------
    def validate_nim_configuration(self) -> list[str]:
        errs: list[str] = []
        try:
            from src.nemo_retriever_client import NeMoRetrieverClient
        except Exception:
            # If client import fails, skip strict model validation here
            return errs

        # Guard check: ensure the validate_model_availability method exists
        if hasattr(NeMoRetrieverClient, "validate_model_availability"):
            if self.embedding_model_override and not NeMoRetrieverClient.validate_model_availability(
                "embedding", self.embedding_model_override
            ):
                errs.append(f"Invalid embedding model override: {self.embedding_model_override}")
            if self.reranking_model_override and not NeMoRetrieverClient.validate_model_availability(
                "reranking", self.reranking_model_override
            ):
                errs.append(f"Invalid reranking model override: {self.reranking_model_override}")
        else:
            # Fallback: skip validation if method doesn't exist (backward compatibility)
            import logging

            logging.getLogger(__name__).warning(
                "NeMoRetrieverClient.validate_model_availability method not found. " "Skipping model validation checks."
            )

        return errs

    def get_effective_models(self) -> dict[str, str]:
        # Defaults aligned with client
        eff = {
            "embedding": self.embedding_model_override or "nvidia/nv-embedqa-e5-v5",
            "reranking": self.reranking_model_override or "llama-3_2-nemoretriever-500m-rerank-v2",
            "extraction": self.extraction_model_override or "nvidia/nv-ingest",
        }
        return eff

    def get_effective_endpoints(self) -> dict[str, str]:
        try:
            from src.nemo_retriever_client import NeMoRetrieverClient

            defaults = NeMoRetrieverClient.DEFAULT_ENDPOINTS
        except Exception:
            defaults = {
                "embedding": os.getenv(
                    "NEMO_EMBEDDING_ENDPOINT", "https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings"
                ),
                "reranking": os.getenv(
                    "NEMO_RERANKING_ENDPOINT", "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
                ),
                "extraction": os.getenv(
                    "NEMO_EXTRACTION_ENDPOINT", "https://ai.api.nvidia.com/v1/retrieval/nvidia/extraction"
                ),
            }
        return {
            "embedding": self.custom_embedding_endpoint or defaults.get("embedding", ""),
            "reranking": self.custom_reranking_endpoint or defaults.get("reranking", ""),
            "extraction": self.custom_extraction_endpoint or defaults.get("extraction", ""),
        }

    def get_health_check_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enable_nim_health_gate,
            "timeout_seconds": self.health_check_timeout_seconds,
            "max_latency_ms": self.max_health_latency_ms,
            "retries": self.health_check_retry_attempts,
        }

    def get_nim_overrides(self) -> dict[str, Any]:
        return {
            "models": {
                "embedding": self.embedding_model_override,
                "reranking": self.reranking_model_override,
                "extraction": self.extraction_model_override,
            },
            "endpoints": {
                "embedding": self.custom_embedding_endpoint,
                "reranking": self.custom_reranking_endpoint,
                "extraction": self.custom_extraction_endpoint,
            },
        }

    def get_nvidia_build_config(self) -> dict[str, Any]:
        """Get NVIDIA Build platform configuration."""
        return {
            "enabled": self.enable_nvidia_build_fallback,
            "base_url": self.nvidia_build_base_url,
            "embedding_model": self.nvidia_build_embedding_model or "nvidia/nv-embed-v1",
            "llm_model": self.nvidia_build_llm_model or "meta/llama-3.1-8b-instruct",
            "endpoints": {
                "embeddings": f"{self.nvidia_build_base_url}/embeddings",
                "chat_completions": f"{self.nvidia_build_base_url}/chat/completions",
            },
        }

    def get_embedding_input_type_config(self) -> tuple[bool, str | None]:
        """Return embedding input_type toggle and value for asymmetric models.

        Returns a tuple (enabled, value). When enabled is False, callers should
        avoid setting input_type. When True and value is provided, set it unless
        the caller already supplies one explicitly in their payload.
        """
        try:
            enabled = bool(self.enable_nvidia_build_embedding_input_type)
        except Exception:
            enabled = True
        value = self.nvidia_build_embedding_input_type
        return enabled, value

    def get_effective_input_type(self) -> str | None:
        """Helper method to return the effective embedding input_type based on config hierarchy.

        Returns the input_type value if config is enabled and value is set, otherwise None.
        This centralizes the logic for determining when input_type should be used.
        """
        if self.enable_nvidia_build_embedding_input_type and self.nvidia_build_embedding_input_type:
            return self.nvidia_build_embedding_input_type
        return None

    # ------------------------- Embedding/Rerank Helpers -------------------------
    def effective_embedding_dim(self) -> int | None:
        """Return the effective embedding dimension for the configured model, if known."""
        model = self.get_effective_models().get("embedding")
        if not model:
            return None
        try:
            # Prefer using NeMoRetrieverClient's model info if available
            from src.nemo_retriever_client import NeMoRetrieverClient  # type: ignore

            key = model.split("/")[-1]
            info = NeMoRetrieverClient.EMBEDDING_MODELS.get(key)
            if info and isinstance(info.get("dimensions"), int):
                return int(info["dimensions"])  # type: ignore[arg-type]
        except Exception:
            pass
        # Fallback mapping for common models
        fallback_dims = {
            "nvidia/nv-embedqa-e5-v5": 1024,
            "nvidia/nv-embed-v1": 4096,
            "Snowflake/snowflake-arctic-embed-l": 1024,
            "nvidia/nv-embedqa-mistral7b-v2": 4096,
        }
        return fallback_dims.get(model)

    def _normalize_model_simple(self, name: str | None) -> str | None:
        """Normalize model naming quirks such as underscores and missing namespaces."""
        if not name:
            return name
        n = name.strip()
        # Replace llama-3_2 with llama-3.2
        n = n.replace("llama-3_2", "llama-3.2")
        # Common rerank short names
        if n == "llama-3_2-nemoretriever-500m-rerank-v2":
            n = "meta/llama-3_2-nemoretriever-500m-rerank-v2"
        # Ensure namespace for known NV models
        if n.startswith("nv-rerank") and not n.startswith("nvidia/"):
            n = f"nvidia/{n}"
        if n.startswith("nv-embed") and not n.startswith("nvidia/"):
            n = f"nvidia/{n}"
        return n

    def get_effective_rerank_model(self) -> str | None:
        """Return normalized effective rerank model name."""
        model = self.get_effective_models().get("reranking")
        return self._normalize_model_simple(model)

    # ------------------------ Request Policy Controls ------------------------
    # Exposed knobs for timeout/retries/backoff used by OpenAIWrapper
    request_timeout_seconds: int = 60
    request_max_retries: int = 3
    request_backoff_base: float = 0.5
    request_backoff_jitter: float = 0.1

    def should_use_nvidia_build_fallback(self) -> bool:
        """Check if NVIDIA Build fallback should be used."""
        return self.enable_nvidia_build_fallback

    def get_cloud_first_strategy(self) -> dict[str, Any]:
        """Get cloud-first configuration strategy with decision logging.

        Note: When `enable_nvidia_build_fallback` is true, NVIDIA Build is used as the
        PRIMARY (cloud-first) endpoint and self-hosted endpoints serve as fallback.
        """
        import logging

        logger = logging.getLogger(__name__)

        strategy = {
            "cloud_first_enabled": self.enable_nvidia_build_fallback,
            "primary_endpoint": "cloud" if self.enable_nvidia_build_fallback else "self_hosted",
            "fallback_endpoint": "self_hosted" if self.enable_nvidia_build_fallback else "none",
            "decision_factors": [],
            "pharmaceutical_optimized": True,  # Always true for this domain
        }

        # Log decision factors
        if self.enable_nvidia_build_fallback:
            strategy["decision_factors"].append("NVIDIA Build cloud-first enabled")
            logger.info("Cloud-first strategy: Using NVIDIA Build as primary endpoint")

            # Check for pharmaceutical optimization preferences
            if self.nvidia_build_embedding_model:
                strategy["decision_factors"].append(f"Custom embedding model: {self.nvidia_build_embedding_model}")

            if self.nvidia_build_llm_model:
                strategy["decision_factors"].append(f"Custom LLM model: {self.nvidia_build_llm_model}")

        else:
            strategy["decision_factors"].append("Cloud-first disabled, using self-hosted NeMo")
            logger.info("Cloud-first strategy: Using self-hosted NeMo as primary")

        # Add cost considerations
        strategy["cost_optimization"] = {
            "free_tier_available": self.enable_nvidia_build_fallback,
            "estimated_monthly_limit": 10000 if self.enable_nvidia_build_fallback else 0,
            "infrastructure_costs": not self.enable_nvidia_build_fallback,
        }

        return strategy

    @property
    def cloud_first_enabled(self) -> bool:
        """Alias for `enable_nvidia_build_fallback` for clarity."""
        return self.enable_nvidia_build_fallback

    def validate_openai_sdk_compatibility(self) -> dict[str, Any]:
        """Validate OpenAI SDK configuration compatibility.

        Returns a dictionary with keys:
        - compatible: bool
        - issues: list[str]
        - base_url: str
        - models: dict[str, Optional[str]]
        """
        issues: list[str] = []

        # Base URL checks
        if not self.nvidia_build_base_url.startswith("https://"):
            issues.append("NVIDIA Build base URL must use HTTPS")
        if not self.nvidia_build_base_url.endswith("/v1"):
            issues.append("NVIDIA Build base URL should end with /v1 for OpenAI compatibility")

        # Model checks when cloud-first is enabled
        if self.enable_nvidia_build_fallback:
            if not self.nvidia_build_embedding_model:
                issues.append("Cloud-first enabled but no embedding model specified")
            if not self.nvidia_build_llm_model:
                issues.append("Cloud-first enabled but no LLM model specified")

        return {
            "compatible": len(issues) == 0,
            "issues": issues,
            "base_url": self.nvidia_build_base_url,
            "models": {
                "embedding": self.nvidia_build_embedding_model,
                "llm": self.nvidia_build_llm_model,
            },
        }

    def get_feature_flags(self) -> dict[str, bool]:
        """Get environment-driven feature flags for modular control."""
        return {
            # Core features
            "cloud_first_enabled": self.enable_nvidia_build_fallback,
            "cloud_first_rerank_enabled": self.enable_cloud_first_rerank,
            "pharmaceutical_optimized": True,  # Always enabled for this domain
            "pharmaceutical_research_mode": bool(getattr(self, "pharmaceutical_research_mode", True)),
            "cost_monitoring_enabled": True,  # Always enabled for free tier
            "ollama_enabled": self.enable_ollama,
            # Pharmaceutical research feature flags
            "drug_interaction_analysis": self.enable_drug_interaction_analysis,
            "clinical_trial_processing": self.enable_clinical_trial_processing,
            "pharmacokinetics_optimization": self.enable_pharmacokinetics_optimization,
            "research_project_budgeting": self.research_project_budgeting,
            "cost_per_query_tracking": self.cost_per_query_tracking,
            # Advanced features (environment-driven)
            "daily_alerts_enabled": _as_bool(self.source_env, "ENABLE_DAILY_CREDIT_ALERTS", True),
            "batch_optimization_enabled": _as_bool(self.source_env, "ENABLE_BATCH_OPTIMIZATION", True),
            "fallback_health_checks_enabled": _as_bool(self.source_env, "ENABLE_FALLBACK_HEALTH_CHECKS", True),
            "pharmaceutical_benchmarks_enabled": _as_bool(self.source_env, "ENABLE_PHARMA_BENCHMARKS", False),
            # Development features
            "debug_logging_enabled": _as_bool(self.source_env, "ENABLE_DEBUG_LOGGING", False),
            "performance_profiling_enabled": _as_bool(self.source_env, "ENABLE_PERFORMANCE_PROFILING", False),
            # Future-proofing features
            "ngc_deprecation_warnings_enabled": _as_bool(self.source_env, "ENABLE_NGC_DEPRECATION_WARNINGS", True),
            "migration_assistance_enabled": _as_bool(self.source_env, "ENABLE_MIGRATION_ASSISTANCE", True),
        }

    def log_configuration_decisions(self) -> None:
        """Log configuration decisions for debugging and monitoring."""
        import logging

        logger = logging.getLogger(__name__)

        strategy = self.get_cloud_first_strategy()
        feature_flags = self.get_feature_flags()
        validation = self.validate_openai_sdk_compatibility()

        logger.info("=== NVIDIA Build Configuration Decisions ===")
        logger.info(f"Primary Strategy: {strategy['primary_endpoint']}")
        logger.info(f"Fallback Strategy: {strategy['fallback_endpoint']}")

        for factor in strategy["decision_factors"]:
            logger.info(f"Decision Factor: {factor}")

        if validation.get("issues"):
            logger.warning("Configuration Validation Issues:")
            for issue in validation.get("issues", []):
                logger.warning(f"  - {issue}")

        # Log key feature flags
        enabled_flags = [flag for flag, enabled in feature_flags.items() if enabled]
        logger.info(f"Enabled Features: {', '.join(enabled_flags)}")

        # Cost optimization logging
        cost_info = strategy["cost_optimization"]
        if cost_info["free_tier_available"]:
            logger.info(
                f"Cost Optimization: Free tier available ({cost_info['estimated_monthly_limit']} requests/month)"
            )
        else:
            logger.info("Cost Optimization: Infrastructure costs apply (self-hosted)")

        # Pharmaceutical research context
        logger.info("--- Pharmaceutical Settings ---")
        try:
            logger.info(
                "Pharma Research Mode | %s",
                bool(getattr(self, "pharmaceutical_research_mode", True)),
            )
        except Exception:
            # Do not break logging if attribute is missing in older configs
            pass
        logger.info(
            "Pharma Flags | DI=%s CT=%s PK=%s QA=%s Compliance=%s Region=%s",
            self.enable_drug_interaction_analysis,
            self.enable_clinical_trial_processing,
            self.enable_pharmacokinetics_optimization,
            self.pharma_quality_assurance_enabled,
            self.pharma_compliance_mode,
            self.pharma_region,
        )
        logger.info(
            "Pharma Models | DI=%s PK=%s CT=%s",
            self.pharma_model_chat_drug_interaction or "default",
            self.pharma_model_chat_pharmacokinetics or "default",
            self.pharma_model_chat_clinical_trial or "default",
        )
        logger.info(
            "Pharma Cost | budgeting=%s limit_usd=%s tracking=%s project=%s",
            self.research_project_budgeting,
            self.research_project_budget_limit_usd,
            self.cost_per_query_tracking,
            self.pharma_project_id or "unset",
        )

        logger.info(
            "Safety Routing | force_70b_for_safety_critical=%s intents=%s",
            self.force_70b_for_safety_critical,
            ",".join(self.safety_critical_intents) if self.safety_critical_intents else "none",
        )

        # Compliance validation warnings
        compliance_issues = self.validate_pharma_environment()
        if compliance_issues:
            logger.warning("Pharmaceutical Compliance Warnings:")
            for issue in compliance_issues:
                logger.warning(f"  - {issue}")

        logger.info("=== End Configuration Decisions ===")

    def get_endpoint_priority_order(self) -> List[dict[str, str]]:
        """Get ordered list of endpoints by priority for fallback logic."""
        endpoints = []

        if self.enable_nvidia_build_fallback:
            # Cloud-first: NVIDIA Build primary, NeMo self-hosted fallback
            endpoints.append(
                {
                    "type": "cloud",
                    "name": "NVIDIA Build",
                    "base_url": self.nvidia_build_base_url,
                    "priority": "primary",
                    "cost_tier": "free_tier",
                }
            )

            # Add Ollama local (if enabled) as middle-tier fallback
            if self.enable_ollama:
                endpoints.append(
                    {
                        "type": "self_hosted",
                        "name": "Ollama Local",
                        "base_url": self.ollama_base_url,
                        "priority": "fallback",
                        "cost_tier": "infrastructure",
                    }
                )

            # Add NeMo self-hosted as fallback
            nemo_endpoints = self.get_effective_endpoints()
            if nemo_endpoints.get("embedding"):
                endpoints.append(
                    {
                        "type": "self_hosted",
                        "name": "NeMo Retriever",
                        "base_url": nemo_endpoints["embedding"],
                        "priority": "fallback",
                        "cost_tier": "infrastructure",
                    }
                )

        else:
            # Traditional: NeMo self-hosted primary
            nemo_endpoints = self.get_effective_endpoints()
            for service, url in nemo_endpoints.items():
                endpoints.append(
                    {
                        "type": "self_hosted",
                        "name": f"NeMo {service.title()}",
                        "base_url": url,
                        "priority": "primary",
                        "cost_tier": "infrastructure",
                    }
                )

        return endpoints

    # Ollama helpers
    def get_ollama_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enable_ollama,
            "base_url": self.ollama_base_url,
            "chat_model": self.ollama_chat_model or "llama3.1:8b",
            "embed_model": self.ollama_embed_model or "nomic-embed-text",
            "timeout_seconds": self.ollama_timeout_seconds,
        }

    def get_fallback_order(self) -> list[str]:
        """Parse and validate fallback order configuration into ordered list with robust normalization.

        Normalization pipeline:
        1. Split on commas and strip whitespace
        2. Convert to lowercase for case-insensitive matching
        3. Remove duplicates while preserving order
        4. Filter against allowed values: nvidia_build, nemo, ollama
        5. Log warnings for unknown/invalid tokens

        Only allowed endpoints are: nvidia_build, nemo, ollama. Invalid tokens
        are ignored with a warning. When all tokens are invalid or list is empty,
        a safe default order is returned.
        """
        allowed = {"nvidia_build", "nemo", "ollama"}
        try:
            import logging

            log = logging.getLogger(__name__)
        except Exception:
            log = None  # pragma: no cover

        # Step 1: Split on commas and strip whitespace
        raw_tokens = [s.strip() for s in (self.fallback_order or "").split(",") if s.strip()]

        # Step 2: Convert to lowercase for case-insensitive matching
        normalized_tokens = [t.lower() for t in raw_tokens]

        # Step 3: Remove duplicates while preserving order
        seen = set()
        deduplicated_tokens = []
        for token in normalized_tokens:
            if token not in seen:
                seen.add(token)
                deduplicated_tokens.append(token)

        # Step 4: Filter against allowed values and log warnings
        validated: list[str] = []
        for original_token, normalized_token in zip(raw_tokens, normalized_tokens):
            if normalized_token in allowed:
                if normalized_token not in validated:  # Additional dedup check
                    validated.append(normalized_token)
            else:
                if log and normalized_token:  # Don't log empty strings
                    log.warning(
                        "Ignoring unknown fallback endpoint: '%s' (normalized: '%s'). Allowed: %s",
                        original_token,
                        normalized_token,
                        sorted(allowed),
                    )

        # Step 5: Return safe default if no valid tokens
        if not validated:
            if log:
                log.info("No valid fallback endpoints found, using default order: nvidia_build,nemo,ollama")
            return ["nvidia_build", "nemo", "ollama"]

        return validated

    def has_nvidia_build_credentials(self) -> bool:
        """Check if NVIDIA Build credentials are available via environment.

        Currently checks NVIDIA_API_KEY presence. Returns False for empty/whitespace values.
        """
        key = (
            self.source_env.get("NVIDIA_API_KEY")
            if isinstance(getattr(self, "source_env", None), dict)
            else os.getenv("NVIDIA_API_KEY")
        )
        return bool(key and str(key).strip())

    def get_rerank_strategy(self) -> dict[str, Any]:
        """Get rerank service prioritization strategy.

        Returns:
            Dictionary with rerank strategy configuration including:
            - cloud_first_enabled: bool - Whether to prioritize cloud rerank services
            - primary_service: str - Primary service name
            - fallback_services: list[str] - Ordered list of fallback services
            - decision_factors: list[str] - Human-readable decision factors
        """
        strategy = {
            "cloud_first_enabled": self.enable_cloud_first_rerank and self.enable_nvidia_build_fallback,
            "primary_service": "nvidia_build"
            if (self.enable_cloud_first_rerank and self.enable_nvidia_build_fallback)
            else "nemo",
            "fallback_services": [],
            "decision_factors": [],
        }

        if strategy["cloud_first_enabled"]:
            strategy["fallback_services"] = ["nemo", "ollama"] if self.enable_ollama else ["nemo"]
            strategy["decision_factors"].append("Cloud-first rerank enabled with NVIDIA Build primary")
            if self.enable_ollama:
                strategy["decision_factors"].append("Ollama fallback enabled")
        else:
            strategy["fallback_services"] = (
                ["nvidia_build", "ollama"]
                if (self.enable_nvidia_build_fallback and self.enable_ollama)
                else (
                    ["nvidia_build"]
                    if self.enable_nvidia_build_fallback
                    else (["ollama"] if self.enable_ollama else [])
                )
            )
            if not self.enable_cloud_first_rerank:
                strategy["decision_factors"].append("Cloud-first rerank disabled, using NeMo primary")
            if not self.enable_nvidia_build_fallback:
                strategy["decision_factors"].append("NVIDIA Build not available, using NeMo primary")

        return strategy

    # ------------------------- Pharma Helpers -------------------------
    def get_pharma_settings(self) -> dict[str, Any]:
        return {
            "flags": {
                "drug_interaction_analysis": self.enable_drug_interaction_analysis,
                "clinical_trial_processing": self.enable_clinical_trial_processing,
                "pharmacokinetics_optimization": self.enable_pharmacokinetics_optimization,
                "qa_enabled": self.pharma_quality_assurance_enabled,
                "compliance_mode": self.pharma_compliance_mode,
            },
            "models": {
                "drug_interaction": self.pharma_model_chat_drug_interaction
                or self.nvidia_build_llm_model
                or "meta/llama-3.1-8b-instruct",
                "pharmacokinetics": self.pharma_model_chat_pharmacokinetics
                or self.nvidia_build_llm_model
                or "meta/llama-3.1-8b-instruct",
                "clinical_trial": self.pharma_model_chat_clinical_trial
                or self.nvidia_build_llm_model
                or "meta/llama-3.1-8b-instruct",
            },
            "batch": {
                "max_size": self.pharma_batch_max_size,
                "max_latency_ms": self.pharma_batch_max_latency_ms,
            },
            "cost": self.get_cost_monitoring_config(),
        }

    def get_pharma_model_for_query_type(self, query_type: str) -> str:
        prefs = self.get_pharma_settings()["models"]
        return prefs.get(query_type, self.nvidia_build_llm_model or "meta/llama-3.1-8b-instruct")

    def get_cost_monitoring_config(self) -> dict[str, Any]:
        return {
            "project_budgeting": self.research_project_budgeting,
            "budget_limit_usd": self.research_project_budget_limit_usd,
            "per_query_tracking": self.cost_per_query_tracking,
            "project_id": self.pharma_project_id,
        }

    def get_batch_config(self) -> dict[str, Any]:
        return {
            "batch_max_size": self.pharma_batch_max_size,
            "batch_max_latency_ms": self.pharma_batch_max_latency_ms,
        }

    def validate_pharma_environment(self) -> list[str]:
        issues: list[str] = []
        if self.pharma_compliance_mode and not self.pharma_require_disclaimer:
            issues.append("Compliance mode enabled but disclaimer not required. Set PHARMA_REQUIRE_DISCLAIMER=true")
        if self.research_project_budgeting and self.research_project_budget_limit_usd <= 0:
            issues.append("Project budgeting enabled but PHARMA_BUDGET_LIMIT_USD not set (>0)")
        if self.enable_drug_interaction_analysis and not (
            self.nvidia_build_llm_model or self.pharma_model_chat_drug_interaction
        ):
            issues.append("Drug interaction analysis enabled but no LLM model preference configured")
        return issues


__all__ = ["EnhancedRAGConfig"]
