"""Medication evidence chat package."""

__version__ = "2.1.0"
__author__ = "Hendrix Moreau"
__description__ = "Medication evidence chat with retrieval-backed answers"

from .ddi_pk_processor import DDIPKProcessor, DrugInteraction, PKParameter
from .medical_guardrails import MedicalGuardrails
from .paper_schema import (
    DOI_PATTERN,
    PMID_PATTERN,
    PMID_PATTERN_EXTRACT,
    Paper,
    clean_identifier,
    coerce_paper,
    coerce_papers,
    extract_doi,
    extract_pmid,
    normalize_doi,
    normalize_identifier,
    normalize_pmid,
    validate_doi,
    validate_pmid,
)
from .ranking_filter import StudyRankingFilter
from .synthesis_engine import KeyFinding, SynthesisEngine

__all__ = [
    "__version__",
    "DDIPKProcessor",
    "DrugInteraction",
    "PKParameter",
    "MedicalGuardrails",
    "StudyRankingFilter",
    "SynthesisEngine",
    "KeyFinding",
    "Paper",
    "coerce_paper",
    "coerce_papers",
    "DOI_PATTERN",
    "PMID_PATTERN",
    "PMID_PATTERN_EXTRACT",
    "clean_identifier",
    "normalize_doi",
    "normalize_pmid",
    "extract_doi",
    "extract_pmid",
    "normalize_identifier",
    "validate_doi",
    "validate_pmid",
]
