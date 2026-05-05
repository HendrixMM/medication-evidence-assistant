"""Tests for src/agent/factory — model normalization and create_agent_loop wiring."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.config import AgentConfig
from src.agent.factory import _normalize_translator_model, create_agent_loop
from src.medical_guardrails import resolve_medical_guardrails_config_path
from src.nvidia_llm_client import NVIDIALLMClient


# ---------------------------------------------------------------------------
# _normalize_translator_model unit tests
# ---------------------------------------------------------------------------


class TestNormalizeTranslatorModel:
    def test_none_maps_to_70b(self) -> None:
        assert _normalize_translator_model(None) == "70b"

    def test_empty_string_maps_to_70b(self) -> None:
        assert _normalize_translator_model("") == "70b"

    def test_short_key_70b_passthrough(self) -> None:
        assert _normalize_translator_model("70b") == "70b"

    def test_short_key_8b_passthrough(self) -> None:
        assert _normalize_translator_model("8b") == "8b"

    def test_full_model_id_70b_passthrough(self) -> None:
        full_id = NVIDIALLMClient.MODELS["70b"]
        assert _normalize_translator_model(full_id) == full_id

    def test_full_model_id_8b_passthrough(self) -> None:
        full_id = NVIDIALLMClient.MODELS["8b"]
        assert _normalize_translator_model(full_id) == full_id

    def test_default_config_nemotron_70b_maps_to_70b(self) -> None:
        # AgentConfig default translator_model is "nvidia/llama-3.1-nemotron-70b-instruct"
        assert _normalize_translator_model("nvidia/llama-3.1-nemotron-70b-instruct") == "70b"

    def test_unknown_model_with_8b_substring_maps_to_8b(self) -> None:
        assert _normalize_translator_model("some/custom-8b-instruct") == "8b"

    def test_unknown_model_with_no_size_hint_returned_as_is(self) -> None:
        # Returned verbatim so NVIDIALLMClient can raise its own error, not silently fall back.
        result = _normalize_translator_model("nvidia/nemotron-4-340b")
        assert result == "nvidia/nemotron-4-340b"

    def test_whitespace_stripped(self) -> None:
        assert _normalize_translator_model("  70b  ") == "70b"


# ---------------------------------------------------------------------------
# create_agent_loop — translator model normalization wiring
# ---------------------------------------------------------------------------

_FACTORY_MODULE = "src.agent.factory"


def _make_config(**kwargs) -> AgentConfig:
    base = dict(
        nvidia_api_key="test-key",
        enable_source_relevance_scoring=False,
        enable_guardrails=False,
    )
    base.update(kwargs)
    return AgentConfig(**base)


def _run_factory_with_config(config: AgentConfig) -> MagicMock:
    """Patch all heavy deps, run create_agent_loop, return the TranslateForConsumerTool mock."""
    mock_translate = MagicMock()
    with patch.multiple(
        _FACTORY_MODULE,
        NVIDIALLMClient=MagicMock(),
        MedlinePlusClient=MagicMock(),
        MedicalGuardrails=MagicMock(),
        QueryPlanner=MagicMock(),
        RetrievalQualityScorer=MagicMock(),
        RetrievalEvaluator=MagicMock(),
        MedlinePlusSearchTool=MagicMock(),
        StudyRankingFilter=MagicMock(),
        SynthesisEngine=MagicMock(),
        DDIPKProcessor=MagicMock(),
        SummarizeEvidenceTool=MagicMock(),
        TranslateForConsumerTool=mock_translate,
        AgentGuardrails=MagicMock(),
        SessionManager=MagicMock(),
        AgentLoop=MagicMock(),
        AgentConfig=MagicMock(from_env=MagicMock(return_value=config)),
    ):
        create_agent_loop()
    return mock_translate


def test_create_agent_loop_normalizes_default_nemotron_model() -> None:
    """Factory must map the default nemotron translator_model to a supported key."""
    config = _make_config(translator_model="nvidia/llama-3.1-nemotron-70b-instruct")
    mock_translate = _run_factory_with_config(config)

    model_used = mock_translate.call_args.kwargs.get("model")

    supported = set(NVIDIALLMClient.MODELS) | set(NVIDIALLMClient.MODELS.values())
    assert model_used in supported, (
        f"TranslateForConsumerTool received unsupported model {model_used!r}; "
        f"expected one of {supported}"
    )
    assert model_used == "70b"


def test_create_agent_loop_passes_through_supported_key() -> None:
    """Factory must not alter model IDs that are already supported."""
    config = _make_config(translator_model="8b")
    mock_translate = _run_factory_with_config(config)

    assert mock_translate.call_args.kwargs.get("model") == "8b"


def test_resolve_medical_guardrails_config_path_uses_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override_path = Path("/tmp/custom-medical-guardrails.json")
    monkeypatch.setenv("MEDICAL_GUARDRAILS_CONFIG_PATH", str(override_path))

    assert resolve_medical_guardrails_config_path() == override_path.resolve()


def test_resolve_medical_guardrails_config_path_defaults_to_repo_relative_absolute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MEDICAL_GUARDRAILS_CONFIG_PATH", raising=False)

    resolved = resolve_medical_guardrails_config_path()

    assert resolved.is_absolute()
    assert resolved == Path(__file__).resolve().parent.parent / "config" / "medical_guardrails.json"


def test_create_agent_loop_uses_resolved_medical_guardrails_path() -> None:
    config = _make_config(translator_model="8b")
    expected_path = "/tmp/resolved-medical-guardrails.json"
    medical_guardrails = MagicMock()

    with patch.multiple(
        _FACTORY_MODULE,
        NVIDIALLMClient=MagicMock(),
        MedlinePlusClient=MagicMock(),
        MedicalGuardrails=medical_guardrails,
        QueryPlanner=MagicMock(),
        RetrievalQualityScorer=MagicMock(),
        RetrievalEvaluator=MagicMock(),
        MedlinePlusSearchTool=MagicMock(),
        StudyRankingFilter=MagicMock(),
        SynthesisEngine=MagicMock(),
        DDIPKProcessor=MagicMock(),
        SummarizeEvidenceTool=MagicMock(),
        TranslateForConsumerTool=MagicMock(),
        AgentGuardrails=MagicMock(),
        SessionManager=MagicMock(),
        AgentLoop=MagicMock(),
        AgentConfig=MagicMock(from_env=MagicMock(return_value=config)),
        resolve_medical_guardrails_config_path=MagicMock(return_value=Path(expected_path)),
    ):
        create_agent_loop()

    medical_guardrails.assert_called_once_with(expected_path, enabled=config.enable_guardrails)


def test_create_agent_loop_passes_retrieval_quality_scorer_to_evaluator() -> None:
    config = _make_config(translator_model="8b")
    evaluator = MagicMock()
    retrieval_quality_scorer = MagicMock()

    with patch.multiple(
        _FACTORY_MODULE,
        NVIDIALLMClient=MagicMock(),
        MedlinePlusClient=MagicMock(),
        MedicalGuardrails=MagicMock(),
        QueryPlanner=MagicMock(),
        RetrievalQualityScorer=MagicMock(return_value=retrieval_quality_scorer),
        RetrievalEvaluator=evaluator,
        MedlinePlusSearchTool=MagicMock(),
        StudyRankingFilter=MagicMock(),
        SynthesisEngine=MagicMock(),
        DDIPKProcessor=MagicMock(),
        SummarizeEvidenceTool=MagicMock(),
        TranslateForConsumerTool=MagicMock(),
        AgentGuardrails=MagicMock(),
        SessionManager=MagicMock(),
        AgentLoop=MagicMock(),
        AgentConfig=MagicMock(from_env=MagicMock(return_value=config)),
    ):
        create_agent_loop()

    assert evaluator.call_args.kwargs["config"] is config
    assert evaluator.call_args.kwargs["retrieval_quality_scorer"] is retrieval_quality_scorer
