"""Public tool exports for the agent workflow."""
from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "MedlinePlusSearchTool": (".medlineplus_search", "MedlinePlusSearchTool"),
    "PubMedSearchTool": (".pubmed_search", "PubMedSearchTool"),
    "SummarizeEvidenceTool": (".summarize_evidence", "SummarizeEvidenceTool"),
    "TranslateForConsumerTool": (".translate_for_consumer", "TranslateForConsumerTool"),
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
    "MedlinePlusSearchTool",
    "PubMedSearchTool",
    "SummarizeEvidenceTool",
    "TranslateForConsumerTool",
]
