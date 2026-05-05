"""Shared pytest configuration for the product-focused test suite."""
from __future__ import annotations

import os

import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("NVIDIA_API_KEY", "test-nvidia-key")
os.environ.setdefault("AGENT_MODEL", "gpt-5.4-mini")
os.environ.setdefault("AGENT_MODE_ENABLED", "true")
os.environ.setdefault("ENABLE_MEDICAL_GUARDRAILS", "false")


@pytest.fixture(autouse=True, scope="session")
def load_dotenv_if_available() -> None:
    """Load local environment variables when python-dotenv is installed."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except Exception:
        return

    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)
