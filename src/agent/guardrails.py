"""Agent guardrail adapter layer."""
from __future__ import annotations

import logging
import re

from guardrails.constants import MEDICAL_DISCLAIMER
from src.medical_guardrails import MedicalGuardrails

from .config import AgentConfig
from .models import ConsumerAnswer

logger = logging.getLogger(__name__)

_EMERGENCY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\boverdose\b"),
    re.compile(r"\bsuicid(?:e|al|ality)\b"),
    re.compile(r"\bcall\s+911\b"),
    re.compile(r"\bmedical\s+emergency\b"),
)


class AgentGuardrails:
    """Adapter for input and output medical guardrails."""

    def __init__(self, guardrails: MedicalGuardrails, config: AgentConfig) -> None:
        self.guardrails = guardrails
        self.config = config

    def validate_input(self, question: str) -> tuple[bool, str | None]:
        if not self.config.enable_guardrails:
            return True, None
        normalized = question.lower()
        if any(pattern.search(normalized) for pattern in _EMERGENCY_PATTERNS):
            return False, "If this is a medical emergency, please call 911 or your local emergency number immediately."
        try:
            result = self.guardrails.validate_medical_query(question)
            if result["is_valid"] is False:
                issues = [str(issue) for issue in result.get("issues", [])]
                if any("PII" in issue or "PHI" in issue for issue in issues):
                    return False, "Please remove personal information from your question before continuing."
                if any("jailbreak" in issue.lower() for issue in issues):
                    return False, "This query cannot be processed."
                return False, "Your question could not be processed. Please rephrase and try again."
            return True, None
        except Exception:
            logger.exception("Guardrails input validation failed")
            return True, None

    def validate_output(self, answer: ConsumerAnswer) -> ConsumerAnswer:
        if not self.config.enable_guardrails:
            return answer
        updated_answer = answer
        if MEDICAL_DISCLAIMER not in answer.disclaimer:
            updated_answer = answer.model_copy(update={"disclaimer": MEDICAL_DISCLAIMER})
        try:
            sources_as_dicts = [source.model_dump() for source in updated_answer.sources]
            self.guardrails.validate_medical_response(updated_answer.answer, sources_as_dicts)

            source_pmids = {source.pmid for source in updated_answer.sources}
            for pmid in re.findall(r"\bPMID\s*:?\s*(\d+)\b", updated_answer.answer):
                if pmid not in source_pmids:
                    logger.warning("Answer referenced PMID not present in sources: %s", pmid)
            return updated_answer
        except Exception:
            logger.exception("Guardrails output validation failed")
            return updated_answer
