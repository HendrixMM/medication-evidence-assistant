from __future__ import annotations

from src.medical_retrieval.perplexity_agent_client import PerplexityAgentError
from src.medical_retrieval.pubmed_fetcher import PubmedFetcherError
from src.medical_retrieval.question_analysis import QuestionAnalysisError
from src.medical_retrieval.ranker import RerankerError


LAYER_QUESTION_ANALYSIS = "question_analysis"
LAYER_AGENT_DISCOVERY = "agent_discovery"
LAYER_PUBMED_FETCH = "pubmed_fetch"
LAYER_RANKING = "ranking"
LAYER_CACHE = "cache"
LAYER_ORCHESTRATION = "orchestration"


class MedicalRetrievalServiceError(RuntimeError):
    """Top-level orchestration failure that could not be attributed to one layer."""


def layer_for_exception(exc: BaseException) -> str:
    if isinstance(exc, QuestionAnalysisError):
        return LAYER_QUESTION_ANALYSIS
    if isinstance(exc, PerplexityAgentError):
        return LAYER_AGENT_DISCOVERY
    if isinstance(exc, PubmedFetcherError):
        return LAYER_PUBMED_FETCH
    if isinstance(exc, RerankerError):
        return LAYER_RANKING
    return LAYER_ORCHESTRATION


__all__ = [
    "LAYER_AGENT_DISCOVERY",
    "LAYER_CACHE",
    "LAYER_ORCHESTRATION",
    "LAYER_PUBMED_FETCH",
    "LAYER_QUESTION_ANALYSIS",
    "LAYER_RANKING",
    "MedicalRetrievalServiceError",
    "PerplexityAgentError",
    "PubmedFetcherError",
    "QuestionAnalysisError",
    "RerankerError",
    "layer_for_exception",
]
