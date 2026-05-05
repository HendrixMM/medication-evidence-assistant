"""Agent package public exports."""
from __future__ import annotations

from importlib import import_module
import os
from typing import Any

from . import models
from .config import AgentConfig
from .session import SessionManager, TranscriptStore

__version__ = "0.1.0"
AGENT_MODE_ENABLED: bool = os.getenv("AGENT_MODE_ENABLED", "true").lower() in ("1", "true", "yes", "on")

_LAZY_EXPORTS = {
    "AgentGuardrails": (".guardrails", "AgentGuardrails"),
    "AgentLoop": (".loop", "AgentLoop"),
    "QueryPlanner": (".planner", "QueryPlanner"),
    "RetrievalEvaluator": (".evaluator", "RetrievalEvaluator"),
    "StreamingAdapter": (".streaming", "StreamingAdapter"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "AGENT_MODE_ENABLED",
    "AgentConfig",
    "AgentGuardrails",
    "AgentLoop",
    "QueryPlanner",
    "RetrievalEvaluator",
    "SessionManager",
    "StreamingAdapter",
    "TranscriptStore",
    "models",
]
