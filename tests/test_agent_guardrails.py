"""Unit tests for AgentGuardrails input validation."""
from __future__ import annotations

from unittest.mock import Mock

from src.agent.config import AgentConfig
from src.agent.guardrails import AgentGuardrails


def _guardrails(*, enabled: bool = True) -> AgentGuardrails:
    medical_guardrails = Mock()
    medical_guardrails.validate_medical_query.return_value = {"is_valid": True}
    return AgentGuardrails(guardrails=medical_guardrails, config=AgentConfig(enable_guardrails=enabled))


# --- emergency detection: should block ---


def test_validate_input_blocks_overdose_query() -> None:
    ok, message = _guardrails().validate_input("I think I took an overdose of warfarin")
    assert ok is False
    assert message is not None


def test_validate_input_blocks_suicidal_query() -> None:
    ok, message = _guardrails().validate_input("I am feeling suicidal and took my medication")
    assert ok is False
    assert message is not None


def test_validate_input_blocks_suicide_query() -> None:
    ok, message = _guardrails().validate_input("I am considering suicide")
    assert ok is False
    assert message is not None


def test_validate_input_blocks_call_911_query() -> None:
    ok, message = _guardrails().validate_input("should I call 911 after taking too many pills?")
    assert ok is False
    assert message is not None


def test_validate_input_blocks_medical_emergency_phrase() -> None:
    ok, message = _guardrails().validate_input("this is a medical emergency please help")
    assert ok is False
    assert message is not None


# --- emergency detection: should not block ---


def test_validate_input_passes_emergency_contraception_query() -> None:
    ok, _ = _guardrails().validate_input("Is emergency contraception safe with warfarin?")
    assert ok is True


def test_validate_input_passes_emergency_medicine_query() -> None:
    ok, _ = _guardrails().validate_input("What drugs are used in emergency medicine protocols?")
    assert ok is True


def test_validate_input_passes_standalone_emergency_word() -> None:
    ok, _ = _guardrails().validate_input("emergency department drug interaction guidelines")
    assert ok is True


# --- guardrails disabled ---


def test_validate_input_passes_all_queries_when_guardrails_disabled() -> None:
    guardrails = _guardrails(enabled=False)
    ok, message = guardrails.validate_input("I took an overdose")
    assert ok is True
    assert message is None
