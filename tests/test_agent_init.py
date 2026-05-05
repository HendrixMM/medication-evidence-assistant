"""Unit tests for agent package exports."""
from __future__ import annotations

import importlib
import sys


def test_agent_mode_enabled_export_does_not_import_loop(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_MODE_ENABLED", "true")
    sys.modules.pop("src.agent", None)
    sys.modules.pop("src.agent.loop", None)

    module = importlib.import_module("src.agent")

    assert module.AGENT_MODE_ENABLED is True
    assert "src.agent.loop" not in sys.modules


def test_agent_tools_package_does_not_eagerly_import_tool_modules() -> None:
    sys.modules.pop("src.agent.tools", None)
    sys.modules.pop("src.agent.tools.medlineplus_search", None)
    sys.modules.pop("src.agent.tools.pubmed_search", None)

    importlib.import_module("src.agent.tools")

    assert "src.agent.tools.medlineplus_search" not in sys.modules
    assert "src.agent.tools.pubmed_search" not in sys.modules
