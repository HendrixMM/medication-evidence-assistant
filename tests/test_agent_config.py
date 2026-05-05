"""Unit tests for agent configuration."""
import pytest

from src.agent.config import AgentConfig


def test_agent_config_defaults() -> None:
    config = AgentConfig.from_env({})

    assert config.max_agent_turns == 3
    assert config.max_pubmed_calls == 5
    assert config.max_total_tokens == 15000
    assert config.sufficient_relevant_articles == 5
    assert config.too_many_articles_threshold == 50
    assert config.enable_source_relevance_scoring is True
    assert config.enable_guardrails is True
    assert config.session_dir == ".sessions"
    assert config.session_base_dir is None


def test_agent_config_overrides_from_env() -> None:
    config = AgentConfig.from_env(
        {
            "AGENT_MAX_TURNS": "5",
            "AGENT_ENABLE_GUARDRAILS": "false",
            "AGENT_SESSION_BASE_DIR": "/tmp/agent-sessions",
        }
    )

    assert config.max_agent_turns == 5
    assert config.enable_guardrails is False
    assert config.session_base_dir == "/tmp/agent-sessions"


def test_agent_config_boolean_parsing() -> None:
    true_values = ("yes", "1", "on")
    false_values = ("no", "0", "off")

    for value in true_values:
        assert AgentConfig.from_env({"AGENT_ENABLE_GUARDRAILS": value}).enable_guardrails is True

    for value in false_values:
        assert AgentConfig.from_env({"AGENT_ENABLE_GUARDRAILS": value}).enable_guardrails is False

    with pytest.raises(ValueError):
        AgentConfig.from_env({"AGENT_ENABLE_GUARDRAILS": "maybe"})


def test_agent_config_integer_parsing_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        AgentConfig.from_env({"AGENT_MAX_TURNS": "three"})


def test_agent_config_safety_critical_intents_parses_comma_separated_values() -> None:
    config = AgentConfig.from_env(
        {"AGENT_SAFETY_CRITICAL_INTENTS": "drug_interaction,dosage,timing"}
    )

    assert config.safety_critical_intents == ("drug_interaction", "dosage", "timing")


def test_agent_config_post_init_validates_limits() -> None:
    with pytest.raises(ValueError):
        AgentConfig(max_agent_turns=0)


def test_agent_config_rejects_top_k_below_sufficient_relevant_articles() -> None:
    with pytest.raises(ValueError, match="evaluator_top_k_articles"):
        AgentConfig(
            evaluator_top_k_articles=4,
            sufficient_relevant_articles=5,
            min_direct_evidence_articles=2,
        )


def test_agent_config_rejects_top_k_below_min_direct_evidence_articles() -> None:
    with pytest.raises(ValueError, match="evaluator_top_k_articles"):
        AgentConfig(
            evaluator_top_k_articles=1,
            sufficient_relevant_articles=1,
            min_direct_evidence_articles=2,
        )


def test_agent_config_retrieval_quality_defaults() -> None:
    config = AgentConfig.from_env({})

    assert config.evaluator_top_k_articles == 8
    assert config.min_direct_evidence_articles == 2
    assert config.min_subquestion_coverage_ratio == 0.67
    assert config.semantic_relevance_min_score == 3.5


def test_agent_config_reads_retrieval_quality_thresholds_from_env() -> None:
    config = AgentConfig.from_env(
        {
            "AGENT_MIN_DIRECT_EVIDENCE": "4",
            "AGENT_MIN_SUBQUESTION_COVERAGE": "0.75",
            "AGENT_EVALUATOR_TOP_K": "11",
            "AGENT_SEMANTIC_RELEVANCE_MIN": "4.0",
        }
    )

    assert config.evaluator_top_k_articles == 11
    assert config.min_direct_evidence_articles == 4
    assert config.min_subquestion_coverage_ratio == 0.75
    assert config.semantic_relevance_min_score == 4.0


def test_agent_config_retrieval_quality_int_parsing_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        AgentConfig.from_env({"AGENT_EVALUATOR_TOP_K": "abc"})


def test_agent_config_retrieval_quality_float_parsing_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        AgentConfig.from_env({"AGENT_MIN_SUBQUESTION_COVERAGE": "not_a_number"})


def test_agent_config_reads_nvidia_api_key_from_env() -> None:
    config = AgentConfig.from_env({"NVIDIA_API_KEY": "nvapi-test"})

    assert config.nvidia_api_key == "nvapi-test"


def test_agent_config_reads_pubmed_eutils_api_key_from_env() -> None:
    config = AgentConfig.from_env({"PUBMED_EUTILS_API_KEY": "eutils-key"})

    assert config.pubmed_api_key == "eutils-key"


def test_agent_config_falls_back_to_pubmed_api_key_env() -> None:
    config = AgentConfig.from_env({"PUBMED_API_KEY": "legacy-key"})

    assert config.pubmed_api_key == "legacy-key"


def test_agent_config_prefers_pubmed_eutils_api_key_when_both_are_set() -> None:
    config = AgentConfig.from_env(
        {
            "PUBMED_EUTILS_API_KEY": "preferred-key",
            "PUBMED_API_KEY": "legacy-key",
        }
    )

    assert config.pubmed_api_key == "preferred-key"
