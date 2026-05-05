"""Shared pharmaceutical query utilities."""
from __future__ import annotations

import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)


def classify_query(text: str) -> str:
    """Classify a pharma query into one of known categories.

    Returns: 'drug_interaction' | 'pharmacokinetics' | 'clinical_trial' | 'general'
    """
    if not text:
        return "general"
    t = text.lower()
    if re.search(r"\b(interaction|interacts?|contraindicat|co-?administer|ddi)\b", t):
        return "drug_interaction"
    if re.search(r"\b(pk|pharmacokinetic|half-?life|clearance|cmax|tmax|auc)\b", t):
        return "pharmacokinetics"
    if re.search(r"\b(trial|phase\s*[1i]{1,3}|randomi[sz]ed|placebo|cohort|arm|endpoint)\b", t):
        return "clinical_trial"
    return "general"


class PharmaceuticalQueryClassifier:
    """
    Centralized pharmaceutical query classifier with thread-safe error handling.

    Provides a single source of truth for pharmaceutical query classification
    across all clients, eliminating code duplication and maintenance drift.
    """

    _instance: PharmaceuticalQueryClassifier | None = None
    _lock = threading.RLock()

    def __init__(self):
        self._import_warned = False
        self._classification_cache: dict[str, str] = {}
        self._cache_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> PharmaceuticalQueryClassifier:
        """Get singleton instance of pharmaceutical query classifier."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def classify_query_safe(self, text: str) -> str:
        """
        Thread-safe classification with centralized error handling.

        Args:
            text: Query text to classify

        Returns:
            Classification type: 'drug_interaction' | 'pharmacokinetics' | 'clinical_trial' | 'general'
        """
        if not text:
            return "general"

        # Check cache first (for performance)
        with self._cache_lock:
            if text in self._classification_cache:
                return self._classification_cache[text]

        try:
            classification = classify_query(text)

            # Cache the result
            with self._cache_lock:
                self._classification_cache[text] = classification

            return classification

        except Exception as e:
            if not self._import_warned:
                logger.warning("Pharmaceutical query classification failed: %s, defaulting to 'general'", str(e))
                self._import_warned = True
            return "general"

    def classify_with_context(self, text: str, medical_context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Enhanced classification with pharmaceutical domain context.

        Args:
            text: Query text to classify
            medical_context: Optional medical context for enhanced classification

        Returns:
            Dictionary with classification details including safety flags
        """
        base_type = self.classify_query_safe(text)

        result = {
            "query_type": base_type,
            "confidence": self._calculate_confidence(text, base_type),
            "safety_critical": base_type in {"drug_interaction", "pharmacokinetics"},
            "requires_disclaimer": True,  # Conservative default for pharmaceutical queries
            "compliance_flags": self._get_compliance_flags(base_type),
        }

        # Add medical context if provided
        if medical_context:
            result["medical_context"] = medical_context

        return result

    def _calculate_confidence(self, text: str, classification: str) -> float:
        """Calculate confidence score for classification."""
        if not text:
            return 0.0

        t = text.lower()

        # Count relevant keywords for the classification
        keyword_counts = {
            "drug_interaction": len(re.findall(r"\b(interaction|interacts?|contraindicat|co-?administer|ddi)\b", t)),
            "pharmacokinetics": len(re.findall(r"\b(pk|pharmacokinetic|half-?life|clearance|cmax|tmax|auc)\b", t)),
            "clinical_trial": len(
                re.findall(r"\b(trial|phase\s*[1i]{1,3}|randomi[sz]ed|placebo|cohort|arm|endpoint)\b", t)
            ),
        }

        if classification == "general":
            # Low confidence for general classification
            return 0.5

        # Calculate confidence based on keyword density
        word_count = len(t.split())
        relevant_keywords = keyword_counts.get(classification, 0)

        if word_count == 0:
            return 0.0

        # Confidence based on keyword density (max 1.0)
        confidence = min(1.0, (relevant_keywords / word_count) * 10)
        return max(0.6, confidence)  # Minimum confidence for specific classifications

    def _get_compliance_flags(self, classification: str) -> dict[str, bool]:
        """Get compliance flags for pharmaceutical classification."""
        return {
            "requires_medical_disclaimer": True,
            "requires_professional_consultation": classification in {"drug_interaction", "pharmacokinetics"},
            "requires_clinical_validation": classification == "clinical_trial",
            "high_risk_content": classification in {"drug_interaction", "pharmacokinetics"},
        }

    def clear_cache(self) -> None:
        """Clear classification cache (for testing or memory management)."""
        with self._cache_lock:
            self._classification_cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get classifier statistics for monitoring."""
        with self._cache_lock:
            cache_size = len(self._classification_cache)

        return {
            "cache_size": cache_size,
            "import_warned": self._import_warned,
            "supported_types": ["drug_interaction", "pharmacokinetics", "clinical_trial", "general"],
        }


# Convenience function for backward compatibility
def classify_pharma_query_safe(text: str) -> str:
    """
    Convenience function for safe pharmaceutical query classification.

    This function provides the same interface as the old duplicate logic
    while using the centralized classifier.
    """
    return PharmaceuticalQueryClassifier.get_instance().classify_query_safe(text)
