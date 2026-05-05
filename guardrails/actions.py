"""
Comprehensive medical safety actions for NVIDIA NeMo Guardrails integration.
"""
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set

# Optional Presidio imports for advanced PII/PHI detection
PRESIDIO_AVAILABLE = False
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    PRESIDIO_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Presidio detected - will use for advanced PII/PHI detection when enabled")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.info("Presidio not available - using regex-based PII/PHI detection")


# Import modular components for better maintainability
from .modules.source_metadata_utils import (
    _get_source_value,
    append_source_warning,
    set_source_flag,
    update_source_metadata,
)

# Provide safe defaults for optional Presidio interfaces
AnalyzerEngine = globals().get("AnalyzerEngine", None)
AnonymizerEngine = globals().get("AnonymizerEngine", None)
NlpEngineProvider = globals().get("NlpEngineProvider", None)


class MedicalSafetyActions:
    """
    Medical safety actions for NVIDIA NeMo Guardrails integration.
    Provides comprehensive safety validation for pharmaceutical research AI.
    """

    def __init__(self):
        """Initialize medical safety actions."""
        self.medical_patterns = {
            "mrn": r"\b(?:MRN|medical record|patient id)[:\s]*(\d{6,12})\b",
            "patient_name": r"\bpatient\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
            "dob": r"\b(?:DOB|date of birth)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
            "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        }

        self.jailbreak_patterns = [
            r"ignore.*medical.*guidelines",
            r"bypass.*safety.*protocols",
            r"pretend.*not.*medical.*ai",
            r"act.*as.*doctor",
            r"provide.*medical.*advice",
            r"diagnose.*patient",
            r"recommend.*treatment.*for",
            r"prescribe.*medication",
        ]

        self.disclaimer_templates = {
            "drug_information": "This information is for research purposes only. Consult healthcare professionals for medical advice.",
            "drug_interactions": "Drug interaction information provided is for educational purposes. Always consult a healthcare provider before making medication changes.",
            "clinical_research": "This analysis is based on published research and is not intended as medical advice. Clinical decisions should involve qualified healthcare professionals.",
            "general_medical": "This content is for informational purposes only and does not constitute medical advice, diagnosis, or treatment recommendations.",
        }

    async def check_medical_context(self, query: str) -> Dict[str, Any]:
        """
        Validate medical queries and determine disclaimer requirements.

        Args:
            query: Input query to validate

        Returns:
            Dictionary containing validation results
        """
        try:
            query_lower = query.lower()

            # Valid pharmaceutical research contexts
            valid_contexts = [
                "drug interaction",
                "pharmacokinetics",
                "clinical trial",
                "mechanism of action",
                "adverse effects",
                "efficacy",
                "safety profile",
                "literature review",
                "meta-analysis",
                "systematic review",
                "research",
                "study",
                "pubmed",
            ]

            # Invalid medical advice contexts
            invalid_contexts = [
                "diagnose me",
                "what medication should i take",
                "medical advice",
                "am i having",
                "should i stop taking",
                "prescribe",
                "treat my",
                "cure for",
                "medical emergency",
                "symptoms mean",
            ]

            # Check for valid pharmaceutical research context
            has_valid_context = any(context in query_lower for context in valid_contexts)

            # Check for invalid medical advice requests
            has_invalid_context = any(context in query_lower for context in invalid_contexts)

            # Determine overall validity
            valid = has_valid_context and not has_invalid_context

            # Special handling for edge cases
            if not has_valid_context and not has_invalid_context:
                # Neutral queries - allow with disclaimer
                valid = True

            context_type = self._classify_query_context(query)

            return {
                "valid": valid,
                "has_valid_context": has_valid_context,
                "has_invalid_context": has_invalid_context,
                "context_type": context_type,
                "disclaimer_required": not has_valid_context or has_invalid_context,
            }

        except Exception as e:
            logger.error(f"Error in medical context check: {str(e)}")
            return {"valid": False, "error": str(e)}

    async def detect_medical_jailbreak(self, query: str) -> bool:
        """
        Detect medical jailbreak attempts using regex patterns.

        Args:
            query: Query to analyze for jailbreak attempts

        Returns:
            Boolean indicating if jailbreak attempt detected
        """
        try:
            query_lower = query.lower()

            # Check against jailbreak patterns
            for pattern in self.jailbreak_patterns:
                if re.search(pattern, query_lower):
                    logger.warning(f"Medical jailbreak detected: {pattern}")
                    return True

            # Additional heuristics for medical jailbreaks
            jailbreak_indicators = [
                ("ignore", "medical"),
                ("pretend", "doctor"),
                ("act as", "physician"),
                ("roleplay", "medical"),
                ("bypass", "safety"),
                ("override", "guidelines"),
            ]

            for word1, word2 in jailbreak_indicators:
                if word1 in query_lower and word2 in query_lower:
                    logger.warning(f"Medical jailbreak indicator: {word1} + {word2}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error in jailbreak detection: {str(e)}")
            return True  # Fail safe - assume jailbreak if error

    async def scan_medical_pii(self, text: str) -> Dict[str, Any]:
        """
        Scan for medical PII/PHI using healthcare identifier patterns.

        Uses Presidio when available for advanced detection, otherwise falls back to regex patterns.

        Args:
            text: Text to scan for PII/PHI

        Returns:
            Dictionary containing detection results
        """
        # Check if Presidio is available
        if PRESIDIO_AVAILABLE:
            try:
                return await self._scan_medical_pii_with_presidio(text)
            except Exception as e:
                logger.warning(f"Presidio PII/PHI scanning failed, falling back to regex: {e}")
                # Fall back to regex-based scanning

        # Regex-based scanning (fallback or when Presidio not available)
        try:
            detected_types = []
            detections = []

            for pii_type, pattern in self.medical_patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    detected_types.append(pii_type)
                    detections.append(
                        {"type": pii_type, "text": match.group(), "start": match.start(), "end": match.end()}
                    )

            return {
                "detected": len(detected_types) > 0,
                "types": list(set(detected_types)),
                "count": len(detections),
                "detections": detections,
            }

        except Exception as e:
            logger.error(f"Error scanning for medical PII: {str(e)}")
            return {"detected": False, "error": str(e)}

    async def _scan_medical_pii_with_presidio(self, text: str) -> Dict[str, Any]:
        """Scan for PII/PHI using Presidio analyzer."""
        try:
            # Initialize Presidio engines if not already done
            if not hasattr(self, "_analyzer"):
                # Create NLP engine provider
                nlp_engine_provider = NlpEngineProvider()
                nlp_engine = nlp_engine_provider.create_engine()

                # Initialize analyzer with the NLP engine
                self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

            # Analyze the text for PII/PHI entities
            analyzer_results = self._analyzer.analyze(text=text, language="en")

            # Extract detected entities
            detected_types = []
            detections = []
            for result in analyzer_results:
                # Map Presidio entity types to our internal types
                entity_type_map = {
                    "PERSON": "patient_name",
                    "PHONE_NUMBER": "phone",
                    "EMAIL_ADDRESS": "email",
                    "US_SSN": "ssn",
                    "DATE_TIME": "dob",
                    "LOCATION": "location",
                    "US_BANK_NUMBER": "bank_account",
                    "CREDIT_CARD": "credit_card",
                    "US_DRIVER_LICENSE": "driver_license",
                    "US_PASSPORT": "passport",
                }

                internal_type = entity_type_map.get(result.entity_type, result.entity_type.lower())
                detected_types.append(internal_type)

                detections.append(
                    {
                        "type": internal_type,
                        "text": text[result.start : result.end],
                        "start": result.start,
                        "end": result.end,
                        "confidence": result.score,
                    }
                )

            return {
                "detected": len(detected_types) > 0,
                "types": list(set(detected_types)),
                "count": len(detections),
                "detections": detections,
            }

        except Exception as e:
            logger.error(f"Error in Presidio PII/PHI scanning: {str(e)}")
            raise

    async def mask_medical_pii(self, text: str, detected_pii: List[str]) -> str:
        """
        Mask medical PII/PHI with appropriate placeholders.

        Uses Presidio when available for advanced anonymization, otherwise falls back to regex patterns.

        Args:
            text: Text containing PII to mask
            detected_pii: List of detected PII types

        Returns:
            Text with PII masked
        """
        # Check if Presidio is available
        if PRESIDIO_AVAILABLE:
            try:
                return await self._mask_medical_pii_with_presidio(text, detected_pii)
            except Exception as e:
                logger.warning(f"Presidio PII/PHI masking failed, falling back to regex: {e}")
                # Fall back to regex-based masking

        # Regex-based masking (fallback or when Presidio not available)
        try:
            masked_text = text

            for pii_type, pattern in self.medical_patterns.items():
                if pii_type in detected_pii:
                    if pii_type == "mrn":
                        masked_text = re.sub(pattern, "[MEDICAL_RECORD_NUMBER]", masked_text, flags=re.IGNORECASE)
                    elif pii_type == "patient_name":
                        masked_text = re.sub(pattern, "[PATIENT_NAME]", masked_text, flags=re.IGNORECASE)
                    elif pii_type == "dob":
                        masked_text = re.sub(pattern, "[DATE_OF_BIRTH]", masked_text, flags=re.IGNORECASE)
                    elif pii_type == "ssn":
                        masked_text = re.sub(pattern, "[SSN]", masked_text, flags=re.IGNORECASE)
                    elif pii_type == "phone":
                        masked_text = re.sub(pattern, "[PHONE_NUMBER]", masked_text, flags=re.IGNORECASE)
                    elif pii_type == "email":
                        masked_text = re.sub(pattern, "[EMAIL_ADDRESS]", masked_text, flags=re.IGNORECASE)

            return masked_text

        except Exception as e:
            logger.error(f"Error masking medical PII: {str(e)}")
            return text  # Return original text if masking fails

    async def _mask_medical_pii_with_presidio(self, text: str, detected_pii: List[str]) -> str:
        """Mask PII/PHI using Presidio anonymizer."""
        try:
            # Initialize Presidio engines if not already done
            if not hasattr(self, "_analyzer") or not hasattr(self, "_anonymizer"):
                # Create NLP engine provider
                nlp_engine_provider = NlpEngineProvider()
                nlp_engine = nlp_engine_provider.create_engine()

                # Initialize analyzer with the NLP engine
                self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

                # Initialize anonymizer
                self._anonymizer = AnonymizerEngine()

            # Analyze the text for PII/PHI entities
            analyzer_results = self._analyzer.analyze(text=text, language="en")

            # Anonymize the text
            if analyzer_results:
                anonymized_result = self._anonymizer.anonymize(text=text, analyzer_results=analyzer_results)
                return anonymized_result.text
            else:
                return text

        except Exception as e:
            logger.error(f"Error in Presidio PII/PHI masking: {str(e)}")
            raise

    async def get_medical_disclaimer(
        self, response_type: str = "general_medical", content_preview: str | None = None
    ) -> str:
        """
        Get context-appropriate medical disclaimers.

        Args:
            response_type: Type of medical response
            content_preview: Optional preview of content for context

        Returns:
            Appropriate disclaimer string
        """
        try:
            key = (response_type or "general_medical").strip().lower()
            mapping = {
                "drug_information": "drug_information",
                "drug_interactions": "drug_interactions",
                "clinical_research": "clinical_research",
                "pharmacokinetics": "pharmacokinetics",
                "safety_information": "safety_information",
                "general": "general_medical",
            }
            template_key = mapping.get(key, "general_medical")
            return self.disclaimer_templates.get(template_key, self.disclaimer_templates.get("general_medical", ""))

        except Exception as e:
            logger.error(f"Error getting medical disclaimer: {str(e)}")
            return self.disclaimer_templates.get("general_medical", "")

    async def medical_hallucination_check(self, response: str, sources: List[Dict]) -> Dict[str, Any]:
        """
        Check for medical hallucinations against provided sources.

        Args:
            response: Generated response to check
            sources: Source documents for validation

        Returns:
            Dictionary containing hallucination detection results
        """
        try:
            # Extract specific claims (numbers, percentages, study names)
            number_pattern = r"\b\d+\.?\d*%?\b"
            study_pattern = r"\b[A-Z][a-z]+\s+et\s+al\.\s+\(\d{4}\)"
            drug_pattern = r"\b[A-Z][a-z]+(?:in|ol|am|ex)\b"

            response_numbers = set(re.findall(number_pattern, response))
            response_studies = set(re.findall(study_pattern, response))
            response_drugs = set(re.findall(drug_pattern, response))

            # Extract same patterns from sources
            source_text = " ".join(
                [
                    source.get("page_content", "")
                    + " "
                    + source.get("metadata", {}).get("abstract", "")
                    + " "
                    + source.get("metadata", {}).get("title", "")
                    for source in sources
                ]
            )

            source_numbers = set(re.findall(number_pattern, source_text))
            source_studies = set(re.findall(study_pattern, source_text))
            source_drugs = set(re.findall(drug_pattern, source_text))

            # Check for unsupported claims
            unsupported_numbers = response_numbers - source_numbers
            unsupported_studies = response_studies - source_studies
            unsupported_drugs = response_drugs - source_drugs

            # Determine hallucination severity
            if len(unsupported_studies) > 0:
                severity = "high"  # Study citations should match exactly
            elif len(unsupported_numbers) > 3:
                severity = "high"  # Many unsupported numbers
            elif len(unsupported_drugs) > 5:
                severity = "moderate"  # Some unsupported drug names
            elif len(unsupported_numbers) > 1:
                severity = "moderate"  # Few unsupported numbers
            else:
                severity = "low"

            hallucination_detected = severity in ["high", "moderate"]

            return {
                "detected": hallucination_detected,
                "severity": severity,
                "details": {
                    "unsupported_numbers": list(unsupported_numbers),
                    "unsupported_studies": list(unsupported_studies),
                    "unsupported_drugs": list(unsupported_drugs),
                },
                "confidence": 0.8 if hallucination_detected else 0.9,
            }

        except Exception as e:
            logger.error(f"Error in hallucination detection: {str(e)}")
            return {"detected": False, "error": str(e)}

    def _classify_query_context(self, query: str) -> str:
        """Classify the type of medical query."""
        query_lower = query.lower()

        if any(term in query_lower for term in ["interaction", "drug-drug", "combination"]):
            return "drug_interactions"
        elif any(term in query_lower for term in ["pharmacokinetics", "pk", "absorption", "metabolism"]):
            return "pharmacokinetics"
        elif any(term in query_lower for term in ["clinical trial", "study", "research"]):
            return "clinical_research"
        elif any(term in query_lower for term in ["side effects", "adverse", "safety"]):
            return "safety_information"
        else:
            return "general_medical"


# ---------------------------------------------------------------------------
# Medical Content Evaluation Functions (using imported utilities)
# ---------------------------------------------------------------------------


async def ensure_disclaimer(response: str, response_type: str) -> str:
    """Ensure the response contains a single appropriate disclaimer and risk notices."""
    try:
        base_response = response or ""
        actions = MedicalSafetyActions()
        disclaimer_text = await actions.get_medical_disclaimer(response_type=response_type)

        normalized_response = base_response.strip()
        disclaimer_marker = "**Medical Disclaimer:**"
        if disclaimer_text and not re.search(r"(?i)\*\*medical disclaimer:\*\*", normalized_response):
            normalized_response = normalized_response + "\n\n" + f"{disclaimer_marker} {disclaimer_text}"

        lower_text = normalized_response.lower()

        if "drug interaction" in lower_text and "always consult a healthcare provider" not in lower_text:
            warning = (
                "⚠️ **Important:** Drug interaction information is for educational purposes. "
                "Always consult a healthcare provider before making medication changes."
            )
            if warning.lower() not in lower_text:
                normalized_response = normalized_response + "\n\n" + warning

        dosage_markers = ["dosage", " dose ", " titration", "dose adjustments"]
        if any(marker in lower_text for marker in dosage_markers) and "never adjust medications" not in lower_text:
            dosage_warning = (
                "⚠️ **Important:** Dosage information is for research purposes only. "
                "Never adjust medications without medical supervision."
            )
            if dosage_warning.lower() not in lower_text:
                normalized_response = normalized_response + "\n\n" + dosage_warning

        return normalized_response
    except Exception as exc:
        logger.error(f"Error ensuring disclaimer: {exc}")
        return response


async def evaluate_hallucination_risk(response: str, sources: List[Dict]) -> Dict[str, Any]:
    """Evaluate hallucination risk and return directives for response handling."""
    try:
        actions = MedicalSafetyActions()
        check = await actions.medical_hallucination_check(response=response, sources=sources)
        if not check:
            return {"block": False, "message": response, "notice": None, "metadata": {}}

        severity = str(check.get("severity", "")).lower()
        if check.get("detected") and severity == "high":
            return {"block": True, "message": response, "notice": "high", "metadata": check}

        message = response
        notice_text = None
        if check.get("detected"):
            notice_text = (
                "⚠️ **Verification Note:** Some claims in this response may require additional verification. "
                "Please cross-reference with original sources."
            )
            message = f"{response}\n\n{notice_text}"

        return {"block": False, "message": message, "notice": notice_text, "metadata": check}
    except Exception as exc:
        logger.error(f"Error evaluating hallucination risk: {exc}")
        return {"block": False, "message": response, "notice": None, "metadata": {"error": str(exc)}}


async def evaluate_fact_check_result(response: str, sources: List[Dict]) -> Dict[str, Any]:
    """Assess fact-check support and construct updated messaging when needed."""
    try:
        fact_check = await validate_against_pubmed_sources(claims=response, sources=sources)
        notice_parts: List[str] = []
        message = response

        if fact_check.get("support_ratio", 1.0) < 0.5:
            notice_parts.append(
                "⚠️ **Source Verification:** Claims may not be fully supported by the provided sources. Please verify with additional literature."
            )

        if fact_check.get("sources_count", len(sources)) < 2:
            notice_parts.append(
                "📝 **Limited Sources:** This analysis is based on limited sources. Consider reviewing additional literature."
            )

        if notice_parts:
            message = f"{response}\n\n" + "\n\n".join(notice_parts)

        return {"message": message, "notice": bool(notice_parts), "metadata": fact_check}
    except Exception as exc:
        logger.error(f"Error evaluating fact check result: {exc}")
        return {"message": response, "notice": False, "metadata": {"error": str(exc)}}


async def evaluate_regulatory_compliance_flow(response: str) -> Dict[str, Any]:
    """Evaluate regulatory compliance and generate remediation guidance."""
    try:
        compliance = await assess_regulatory_compliance(response=response)
        if not compliance:
            return {"block": False, "message": response, "metadata": {}}

        lower_response = response.lower()
        if any(term in lower_response for term in ["guaranteed cure", "miracle drug"]):
            return {"block": True, "message": response, "metadata": compliance}

        message = response
        violations = compliance.get("violations", [])
        if violations:
            regulatory_notice = "⚠️ **Regulatory Note:** This information is for research purposes and does not constitute medical advice."
            if regulatory_notice.lower() not in message.lower():
                message = f"{message}\n\n{regulatory_notice}"

        if any(term in lower_response for term in ["efficacy", "effectiveness"]):
            disclaimer = "📋 **Regulatory Note:** Efficacy claims are based on published research and may not reflect FDA/EMA approved indications."
            if disclaimer.lower() not in message.lower():
                message = f"{message}\n\n{disclaimer}"

        return {"block": False, "message": message, "metadata": compliance}
    except Exception as exc:
        logger.error(f"Error evaluating regulatory compliance: {exc}")
        return {"block": False, "message": response, "metadata": {"error": str(exc)}}


async def filter_sensitive_response_content(response: str) -> Dict[str, Any]:
    """Filter sensitive medical content and detect disallowed advice patterns."""
    try:
        filtered = await filter_sensitive_medical_info(response=response)
        text = filtered or ""

        risk_patterns = [
            r"(?i)\byou\b.*(should take|must take|need to take|stop taking|increase your dose|decrease your dose)",
            r"(?i)\b(i|we)\s+(recommend|advise)\s+(you|that you|taking|stopping)",
            r"(?i)\byou\b.*(have|are suffering|diagnosis is|are diagnosed with)",
            r"(?i)(your diagnosis|you likely have|you probably have|this indicates you)",
        ]

        for pattern in risk_patterns:
            if re.search(pattern, text):
                return {"block": True, "message": text, "reason": pattern}

        return {"block": False, "message": text, "reason": None}
    except Exception as exc:
        logger.error(f"Error filtering sensitive response content: {exc}")
        return {"block": False, "message": response, "reason": str(exc)}


async def append_pharmaceutical_warnings_response(response: str) -> str:
    """Append contextual pharmaceutical safety warnings when keywords are present."""
    try:
        additions: List[str] = []
        lowered = response.lower()
        if re.search(r"interaction|contraindicated|avoid.*combination", lowered):
            additions.append(
                "🚨 **Drug Interaction Warning:** This information is for research purposes. Clinical drug interaction decisions should involve healthcare professionals."
            )
        if re.search(r"adverse|side effect|toxicity|contraindication", lowered):
            additions.append(
                "⚠️ **Safety Information:** Adverse effect insights are based on clinical studies. Individual responses may vary; consult healthcare providers for concerns."
            )
        if re.search(r"metabolism|clearance|half-life|bioavailability", lowered):
            additions.append(
                "🧬 **Pharmacokinetic Note:** PK parameters may vary with genetics, age, disease state, and concomitant medications."
            )
        if not additions:
            return response
        return response + "\n\n" + "\n\n".join(additions)
    except Exception as exc:
        logger.error(f"Error appending pharmaceutical warnings: {exc}")
        return response


async def append_evidence_quality_summary(response: str, sources: List[Dict]) -> str:
    """Append evidence quality summary based on assessed source levels."""
    try:
        levels = await assess_evidence_levels(sources=sources)
        if not levels:
            return response
        parts: List[str] = ["📊 **Evidence Quality:**"]
        if levels.get("high_quality", 0):
            parts.append(f"High quality studies: {levels['high_quality']}")
        if levels.get("moderate_quality", 0):
            parts.append(f"Moderate quality studies: {levels['moderate_quality']}")
        if levels.get("low_quality", 0):
            parts.append(f"Lower quality studies: {levels['low_quality']}")
        if len(parts) == 1:
            return response
        summary = " | ".join(parts)
        return f"{response}\n\n{summary}"
    except Exception as exc:
        logger.error(f"Error appending evidence quality summary: {exc}")
        return response


async def ensure_citation_block(response: str, sources: List[Dict]) -> Dict[str, Any]:
    """Ensure the response contains citations referencing provided sources."""
    try:
        citations_block_added = False
        if not re.search(r"PMID|PubMed|doi|et al", response, re.IGNORECASE):
            citation_list = await format_source_citations(sources=sources)
            if citation_list:
                response = f"{response}\n\n📚 **Sources:**\n{citation_list}"
                citations_block_added = True

        citation_validation = await validate_citations(response=response, sources=sources)
        return {
            "message": response,
            "invalid_citations": citation_validation.get("invalid_citations", []),
            "citations_added": citations_block_added,
            "metadata": citation_validation,
        }
    except Exception as exc:
        logger.error(f"Error ensuring citation block: {exc}")
        return {"message": response, "invalid_citations": [], "citations_added": False, "metadata": {"error": str(exc)}}


async def enhance_response_quality_format(response: str) -> str:
    """Apply structured formatting helpers to improve medical response clarity."""
    try:
        enhanced = response
        if re.search(r"(?i)mechanism of action|\bMOA\b", enhanced):
            enhanced = await structure_moa_information(response=enhanced)
        if re.search(r"(?i)(drug\s+interaction|\bDDI\b)", enhanced):
            enhanced = await structure_interaction_information(response=enhanced)
        if re.search(r"(?i)pharmacokinetic|clearance|half-life|bioavailability|cmax|auc", enhanced):
            enhanced = await structure_pk_information(response=enhanced)
        return enhanced
    except Exception as exc:
        logger.error(f"Error enhancing response quality: {exc}")
        return response


# Helper functions for medical query classification
async def classify_medical_query_type(query: str) -> str:
    """
    Classify the type of medical query for appropriate handling.

    Args:
        query: Query to classify

    Returns:
        String indicating query type
    """
    try:
        query_lower = query.lower()

        if "drug interaction" in query_lower or "ddi" in query_lower:
            return "drug_interactions"
        elif "pharmacokinetics" in query_lower or "pk" in query_lower or "adme" in query_lower:
            return "pharmacokinetics"
        elif "clinical trial" in query_lower or "study" in query_lower or "research" in query_lower:
            return "clinical_research"
        elif "side effects" in query_lower or "adverse" in query_lower or "safety" in query_lower:
            return "safety_information"
        elif "mechanism" in query_lower or "moa" in query_lower or "target" in query_lower:
            return "mechanism_of_action"
        else:
            return "general_pharmaceutical"

    except Exception as e:
        logger.error(f"Error classifying query type: {str(e)}")
        return "general_pharmaceutical"


async def validate_pharmaceutical_context(query: str) -> Dict[str, Any]:
    """
    Validate that queries are within pharmaceutical research scope.

    Args:
        query: Query to validate

    Returns:
        Dictionary containing validation results
    """
    try:
        query_lower = query.lower()

        # Research-focused indicators
        research_indicators = [
            "literature review",
            "clinical trial",
            "pharmacokinetics",
            "drug interaction",
            "mechanism of action",
            "adverse effects",
            "systematic review",
            "meta-analysis",
            "pubmed",
            "research",
            "study",
            "efficacy",
            "safety profile",
        ]

        # Medical advice indicators (to reject)
        advice_indicators = [
            "diagnose me",
            "what should I take",
            "medical advice",
            "treat my",
            "cure for",
            "am I having",
            "symptoms mean",
            "should I stop",
            "prescribe me",
            "medical emergency",
        ]

        has_research_context = any(indicator in query_lower for indicator in research_indicators)
        seeks_medical_advice = any(indicator in query_lower for indicator in advice_indicators)

        return {
            "valid": has_research_context and not seeks_medical_advice,
            "research_focused": has_research_context,
            "seeks_advice": seeks_medical_advice,
            "context_type": await classify_medical_query_type(query),
        }

    except Exception as e:
        logger.error(f"Error validating pharmaceutical context: {str(e)}")
        return {"valid": False, "error": str(e)}


async def assess_medical_toxicity(text: str) -> float:
    """
    Assess toxicity level of medical text content.

    Args:
        text: Text to assess

    Returns:
        Float toxicity score (0.0 to 1.0)
    """
    try:
        text_lower = text.lower()

        # High toxicity indicators
        high_toxicity_terms = [
            "self-harm",
            "suicide",
            "overdose",
            "kill myself",
            "end my life",
            "dangerous drug",
            "illegal substance",
            "poison",
            "toxic dose",
        ]

        # Moderate toxicity indicators
        moderate_toxicity_terms = [
            "recreational drug",
            "get high",
            "abuse",
            "addiction",
            "street drug",
            "controlled substance",
        ]

        toxicity_score = 0.0

        # Check high toxicity terms
        for term in high_toxicity_terms:
            if term in text_lower:
                toxicity_score = max(toxicity_score, 0.9)

        # Check moderate toxicity terms
        for term in moderate_toxicity_terms:
            if term in text_lower:
                toxicity_score = max(toxicity_score, 0.6)

        return toxicity_score

    except Exception as e:
        logger.error(f"Error assessing medical toxicity: {str(e)}")
        return 0.5  # Default moderate score on error


async def validate_against_pubmed_sources(claims: str, sources: List[Dict]) -> Dict[str, Any]:
    """
    Validate medical claims against PubMed sources.

    Args:
        claims: Text containing claims to validate
        sources: List of source documents

    Returns:
        Dictionary containing validation results
    """
    try:
        # Extract factual claims from response
        claim_patterns = [
            r"studies show[s]?\s+([^.]+)",
            r"research indicates?\s+([^.]+)",
            r"evidence suggests?\s+([^.]+)",
            r"trials demonstrated?\s+([^.]+)",
            r"analysis found\s+([^.]+)",
        ]

        extracted_claims = []
        for pattern in claim_patterns:
            matches = re.findall(pattern, claims, re.IGNORECASE)
            extracted_claims.extend(matches)

        # Validate claims against sources
        source_abstracts = " ".join(
            [source.get("metadata", {}).get("abstract", "") + source.get("page_content", "") for source in sources]
        ).lower()

        supported_claims = []
        unsupported_claims = []

        for claim in extracted_claims:
            claim_lower = claim.lower()
            # Simple keyword matching for claim validation
            claim_words = set(claim_lower.split())

            # Calculate overlap with source content
            source_words = set(source_abstracts.split())
            overlap = len(claim_words.intersection(source_words))

            support_ratio = overlap / len(claim_words) if claim_words else 0

            if support_ratio > 0.3:  # 30% keyword overlap threshold
                supported_claims.append(claim)
            else:
                unsupported_claims.append(claim)

        return {
            "total_claims": len(extracted_claims),
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "support_ratio": len(supported_claims) / len(extracted_claims) if extracted_claims else 1.0,
            "sources_count": len(sources),
        }

    except Exception as e:
        logger.error(f"Error validating against PubMed sources: {str(e)}")
        return {"error": str(e)}


async def assess_regulatory_compliance(response: str) -> Dict[str, Any]:
    """
    Assess response for regulatory compliance.

    Args:
        response: Response text to check

    Returns:
        Dictionary containing compliance assessment
    """
    try:
        violations = []
        warnings = []

        response_lower = response.lower()

        # FDA compliance checks
        fda_violations = [
            "guaranteed cure",
            "miracle drug",
            "completely safe",
            "no side effects",
            "fda approved for all",
            "always effective",
        ]

        for violation in fda_violations:
            if violation in response_lower:
                violations.append(f"FDA compliance violation: {violation}")

        # Medical advice warnings
        advice_patterns = [
            "you should take",
            "i recommend",
            "stop taking",
            "increase dose",
            "this will cure",
            "definitely effective",
        ]

        for pattern in advice_patterns:
            if pattern in response_lower:
                warnings.append(f"Potential medical advice: {pattern}")

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "compliance_score": 1.0 - (len(violations) * 0.3) - (len(warnings) * 0.1),
        }

    except Exception as e:
        logger.error(f"Error assessing regulatory compliance: {str(e)}")
        return {"compliant": False, "error": str(e)}


# Additional missing actions for rails flows


async def classify_medical_response_type(response: str) -> str:
    """Classify medical response type for appropriate disclaimer."""
    try:
        response_lower = response.lower()
        if "drug interaction" in response_lower or "interaction" in response_lower:
            return "drug_interactions"
        elif "clinical trial" in response_lower or "study" in response_lower:
            return "clinical_research"
        elif "efficacy" in response_lower or "effectiveness" in response_lower:
            return "drug_information"
        else:
            return "general_medical"
    except Exception as e:
        logger.error(f"Error classifying response type: {str(e)}")
        return "general_medical"


async def filter_sensitive_medical_info(response: str) -> str:
    """Filter sensitive medical information from responses."""
    try:
        # Remove any remaining PII patterns
        filtered = response
        pii_patterns = {
            r"\b(?:MRN|medical record)[:\s]*\d{6,12}\b": "[MEDICAL_RECORD]",
            r"\bpatient\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b": "[PATIENT_NAME]",
            r"\b\d{3}-?\d{2}-?\d{4}\b": "[SSN]",
        }

        for pattern, replacement in pii_patterns.items():
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)

        return filtered
    except Exception as e:
        logger.error(f"Error filtering sensitive info: {str(e)}")
        return response


async def assess_evidence_levels(sources: List[Dict]) -> Dict[str, Any]:
    """Assess evidence levels of provided sources with granular detail.

    Returns:
        Dict with consistent structure containing:
        - quality counts (int): high_quality, moderate_quality, low_quality, unclassified
        - high_quality_ratio (float): Ratio of high quality sources
        - details (List[Dict]): Detailed information for each source
        - notes (List[str]): Any notes or warnings
    """
    # Default return structure with all expected keys
    default_return = {
        "high_quality": 0,
        "moderate_quality": 0,
        "low_quality": 0,
        "unclassified": 0,
        "high_quality_ratio": 0.0,
        "details": [],
        "notes": [],
    }

    try:
        if not sources:
            return {**default_return, "notes": ["No sources provided"]}

        quality_mapping = {
            "systematic_review": ("high", "Level 1"),
            "meta_analysis": ("high", "Level 1"),
            "randomized_controlled_trial": ("high", "Level 2"),
            "clinical_trial": ("high", "Level 2"),
            "cohort_study": ("moderate", "Level 3"),
            "case_control_study": ("moderate", "Level 3"),
            "observational_study": ("moderate", "Level 4"),
            "case_report": ("low", "Level 5"),
            "case_series": ("low", "Level 5"),
            "in_vitro": ("low", "Level 6"),
        }

        quality_counts = {"high": 0, "moderate": 0, "low": 0, "unclassified": 0}
        details: List[Dict[str, Any]] = []

        for source in sources:
            metadata = source.get("metadata", {})
            content = (metadata.get("abstract", "") + " " + source.get("page_content", "")).lower()
            study_type = metadata.get("study_type")

            if not study_type:
                study_type = await classify_study_type(content)

            quality_label, evidence_level = quality_mapping.get(study_type, ("unclassified", "Level 6"))
            quality_counts[quality_label] = quality_counts.get(quality_label, 0) + 1

            journal = metadata.get("journal", "")
            journal_quality = {}
            if journal:
                journal_quality = await assess_journal_quality(journal=journal)

            details.append(
                {
                    "title": metadata.get("title", "Unknown title"),
                    "study_type": study_type,
                    "evidence_level": evidence_level,
                    "quality_label": quality_label,
                    "journal_quality": journal_quality,
                    "pmid": metadata.get("pmid"),
                }
            )

        total_sources = len(sources)
        high_quality_ratio = quality_counts["high"] / total_sources if total_sources else 0.0

        notes = []
        if high_quality_ratio < 0.5:
            notes.append("Less than half of the sources are high-evidence studies")
        if quality_counts["unclassified"]:
            notes.append("Some sources could not be classified based on available metadata")

        return {
            "high_quality": quality_counts["high"],
            "moderate_quality": quality_counts["moderate"],
            "low_quality": quality_counts["low"],
            "unclassified": quality_counts["unclassified"],
            "high_quality_ratio": round(high_quality_ratio, 2),
            "details": details,
            "notes": notes,
        }
    except Exception as e:
        logger.error(f"Error assessing evidence levels: {str(e)}")
        return {**default_return, "notes": [f"Assessment failed: {str(e)}"]}


async def format_source_citations(sources: List[Dict]) -> str:
    """Format source citations for responses.

    Returns:
        str: Formatted citations as a string, one per line, or error message
    """
    try:
        if not sources:
            return "No supporting sources supplied."

        citations = []
        for i, source in enumerate(sources[:5], 1):  # Limit to 5 citations
            metadata = source.get("metadata", {})
            title = metadata.get("title", "Unknown title").strip()
            journal = metadata.get("journal", "").strip()
            year = metadata.get("year") or metadata.get("publication_year") or ""
            pmid = metadata.get("pmid")
            doi = metadata.get("doi")
            authors = metadata.get("authors")

            if isinstance(authors, list):
                if not authors:
                    author_str = "Anonymous"
                elif len(authors) == 1:
                    author_str = authors[0]
                elif len(authors) == 2:
                    author_str = " & ".join(authors)
                else:
                    author_str = f"{authors[0]} et al."
            elif isinstance(authors, str) and authors.strip():
                author_str = authors.strip()
            else:
                author_str = "Anonymous"

            fragments = [f"{i}. {author_str}"]
            if year:
                fragments[-1] = f"{fragments[-1]} ({year})"
            fragments.append(title[:140] + ("..." if len(title) > 140 else ""))
            if journal:
                fragments.append(journal)
            if pmid:
                fragments.append(f"PMID: {pmid}")
            if doi:
                fragments.append(f"DOI: {doi}")

            # Filter out empty fragments and join
            filtered_fragments = [fragment for fragment in fragments if fragment]
            if filtered_fragments:
                citations.append(" - ".join(filtered_fragments))

        if not citations:
            return "No valid citations could be formatted."

        return "\n".join(citations)
    except Exception as e:
        logger.error(f"Error formatting citations: {str(e)}")
        return "Citations unavailable due to formatting error."


async def validate_citations(response: str, sources: List[Dict]) -> Dict[str, Any]:
    """Validate citations in response against provided sources.

    Returns:
        Dict with consistent structure containing:
        - invalid_citations (List[str]): List of invalid PMIDs found in response
        - valid_citations (List[str]): List of valid PMIDs found in response
        - all_valid (bool): Whether all citations are valid
    """
    # Default return structure with all expected keys
    default_return = {"invalid_citations": [], "valid_citations": [], "all_valid": True}

    try:
        # Extract PMIDs from response
        pmid_pattern = r"PMID:?\s*(\d{7,8})"
        response_pmids = set(re.findall(pmid_pattern, response, re.IGNORECASE))

        # Extract PMIDs from sources
        source_pmids = set()
        for source in sources:
            pmid = source.get("metadata", {}).get("pmid", "")
            if pmid:
                source_pmids.add(str(pmid))

        invalid_citations = list(response_pmids - source_pmids)
        valid_citations = list(response_pmids & source_pmids)

        return {
            "invalid_citations": invalid_citations,
            "valid_citations": valid_citations,
            "all_valid": len(invalid_citations) == 0,
        }
    except Exception as e:
        logger.error(f"Error validating citations: {str(e)}")
        return {**default_return, "all_valid": False, "invalid_citations": [f"Validation error: {str(e)}"]}


async def comprehensive_safety_check(response: str) -> Dict[str, Any]:
    """Comprehensive safety check for responses."""
    try:
        warnings = []
        safety_score = 1.0

        response_lower = response.lower()

        # Check for medical advice language
        advice_patterns = ["you should", "i recommend", "take this drug", "stop taking"]
        for pattern in advice_patterns:
            if pattern in response_lower:
                warnings.append(f"Contains potential medical advice: '{pattern}'")
                safety_score -= 0.2

        # Check for diagnostic language
        diagnostic_patterns = ["you have", "diagnosis is", "you are suffering from"]
        for pattern in diagnostic_patterns:
            if pattern in response_lower:
                warnings.append(f"Contains diagnostic language: '{pattern}'")
                safety_score -= 0.3

        # Check for guarantee claims
        guarantee_patterns = ["guaranteed cure", "always works", "definitely effective"]
        for pattern in guarantee_patterns:
            if pattern in response_lower:
                warnings.append(f"Contains inappropriate guarantee: '{pattern}'")
                safety_score -= 0.4

        return {"safe": safety_score > 0.5, "safety_score": max(safety_score, 0.0), "warnings": warnings}
    except Exception as e:
        logger.error(f"Error in comprehensive safety check: {str(e)}")
        return {"safe": False, "safety_score": 0.0, "warnings": ["Safety check failed"]}


async def log_safety_event(event_type: str, details: Any) -> None:
    """Log safety events for monitoring and compliance."""
    try:
        logger.warning(f"SAFETY_EVENT: {event_type} - {details}")
        # In production, this would log to a structured logging system
    except Exception as e:
        logger.error(f"Error logging safety event: {str(e)}")


async def log_medical_response(
    response: str, safety_score: float, disclaimers_added: bool, compliance_status: bool
) -> None:
    """Log medical responses for audit trails."""
    try:
        log_data = {
            "response_length": len(response),
            "safety_score": safety_score,
            "disclaimers_added": disclaimers_added,
            "compliance_status": compliance_status,
            "timestamp": "current_timestamp",  # Would use actual timestamp in production
        }
        logger.info(f"MEDICAL_RESPONSE_LOG: {log_data}")
        # In production, this would log to a structured audit system
    except Exception as e:
        logger.error(f"Error logging medical response: {str(e)}")


# Additional retrieval and processing actions


async def request_additional_sources() -> None:
    """Request additional sources when insufficient valid sources are found."""
    try:
        logger.info("Requesting additional sources due to insufficient valid sources")
        # In production, this would trigger additional source retrieval
    except Exception as e:
        logger.error(f"Error requesting additional sources: {str(e)}")


async def process_pubmed_sources(sources: List[Dict]) -> Dict[str, Any]:
    """Run authenticity screening on retrieved sources and filter suspicious entries."""
    if not sources:
        return {
            "filtered_sources": [],
            "suspicious_sources": [],
            "issues": ["No sources provided"],
            "authentic_summary": {},
            "insufficient_valid": True,
        }

    authenticity = await validate_source_authenticity(sources)
    details = authenticity.get("details") or []
    filtered_sources: List[Dict[str, Any]] = []
    suspicious_sources: List[Dict[str, Any]] = []

    for idx, source in enumerate(sources):
        detail = details[idx] if idx < len(details) else {}
        if detail.get("authentic"):
            filtered_sources.append(source)
        else:
            suspicious_sources.append(
                {
                    "source": source,
                    "reasons": detail.get("reasons", []),
                    "title": detail.get("title"),
                }
            )

    insufficient_valid = len(filtered_sources) < 2

    return {
        "filtered_sources": filtered_sources,
        "suspicious_sources": suspicious_sources,
        "issues": authenticity.get("issues", []),
        "authentic_summary": authenticity,
        "insufficient_valid": insufficient_valid,
    }


async def assess_pharmaceutical_relevance(source: Dict, query: str) -> Dict[str, Any]:
    """Assess pharmaceutical relevance of a source to the query using weighted heuristics."""
    try:
        content = source.get("page_content", "").lower()
        metadata = source.get("metadata", {})
        title = metadata.get("title", "").lower()
        abstract = metadata.get("abstract", "").lower()
        combined_text = " ".join(filter(None, [title, abstract, content]))

        query_terms = [term for term in re.findall(r"[a-zA-Z0-9-]+", query.lower()) if len(term) > 2]
        total_terms = len(query_terms)

        if total_terms == 0:
            return {"score": 0, "medical_context": False, "matches": 0, "total_terms": 0}

        matched_terms = [term for term in query_terms if term in combined_text]
        overlap_score = len(matched_terms) / total_terms if total_terms else 0

        pharma_keywords = {
            "pharmacokinetic": 1.0,
            "pharmacodynamic": 1.0,
            "drug interaction": 1.0,
            "clinical trial": 0.8,
            "adverse effect": 0.6,
            "dosage": 0.6,
            "bioavailability": 0.7,
            "metabolism": 0.5,
            "therapeutic": 0.4,
            "cyp": 0.7,
        }

        keyword_score = 0.0
        for keyword, weight in pharma_keywords.items():
            if keyword in combined_text:
                keyword_score += weight
        keyword_score = min(keyword_score / max(len(pharma_keywords), 1), 1.0)

        drug_indicators = ["drug", "medication", "pharmaco", "cyp", "enzyme", "metabolite"]
        medical_context = keyword_score > 0 or any(indicator in combined_text for indicator in drug_indicators)

        relevance_penalty = 0.0 if medical_context else 0.3
        score = max(0.0, min(1.0, (0.6 * overlap_score + 0.4 * keyword_score) - relevance_penalty))

        missing_terms = [term for term in query_terms if term not in combined_text]

        return {
            "score": round(score, 2),
            "medical_context": medical_context,
            "matches": len(matched_terms),
            "total_terms": total_terms,
            "missing_terms": missing_terms,
            "keyword_score": round(keyword_score, 2),
        }
    except Exception as e:
        logger.error(f"Error assessing pharmaceutical relevance: {str(e)}")
        return {
            "score": 0,
            "medical_context": False,
            "matches": 0,
            "total_terms": 0,
            "missing_terms": [],
            "error": str(e),
        }


async def filter_pharmaceutical_relevance_batch(sources: List[Dict], query: str) -> Dict[str, Any]:
    """Filter sources by pharmaceutical relevance using asynchronous relevance scoring."""
    if not sources:
        return {
            "filtered_sources": [],
            "warnings": ["No sources available for relevance filtering"],
            "high_relevance_count": 0,
        }

    filtered_sources: List[Dict[str, Any]] = []
    warnings: List[str] = []
    high_relevance_count = 0

    for source in sources:
        relevance = await assess_pharmaceutical_relevance(source, query)
        score = relevance.get("score", 0.0)
        medical_context = relevance.get("medical_context", False)

        if score < 0.6 and not medical_context:
            continue

        if score < 0.6:
            warnings.append(
                f"Source '{source.get('metadata', {}).get('title', 'Unknown source')}' has low relevance score {score}",
            )

        enriched_source = dict(source)
        metadata = dict(enriched_source.get("metadata", {}))
        metadata["relevance_score"] = score
        metadata["medical_context"] = medical_context
        metadata["query_matches"] = relevance.get("matches", 0)
        metadata["missing_terms"] = relevance.get("missing_terms", [])
        enriched_source["metadata"] = metadata

        indicators = await identify_pharmaceutical_indicators(enriched_source)
        metadata["pharmaceutical_focus"] = indicators

        if score >= 0.8:
            high_relevance_count += 1

        filtered_sources.append(enriched_source)

    if not filtered_sources:
        warnings.append("All sources were filtered out due to low pharmaceutical relevance")

    return {
        "filtered_sources": filtered_sources,
        "warnings": warnings,
        "high_relevance_count": high_relevance_count,
    }


async def identify_pharmaceutical_indicators(source: Dict) -> List[str]:
    """Identify pharmaceutical indicators in a source."""
    try:
        content = source.get("page_content", "").lower()

        indicators = []
        pharma_patterns = {
            "drug_names": r"\b[A-Z][a-z]+(?:in|ol|am|ex|ide)\b",
            "pk_parameters": r"\b(?:auc|cmax|tmax|clearance|half-life)\b",
            "drug_interactions": r"\b(?:interaction|inhibit|induce|substrate)\b",
            "clinical_endpoints": r"\b(?:efficacy|safety|adverse|side effects)\b",
        }

        for indicator_type, pattern in pharma_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                indicators.append(indicator_type)

        return indicators
    except Exception as e:
        logger.error(f"Error identifying pharmaceutical indicators: {str(e)}")
        return []


async def suggest_query_refinement(query: str) -> None:
    """Suggest query refinement for better pharmaceutical relevance."""
    try:
        logger.info(f"Suggesting refinement for query: {query}")
        # In production, this would provide specific refinement suggestions
    except Exception as e:
        logger.error(f"Error suggesting query refinement: {str(e)}")


async def calculate_title_similarity(title1: str, title2: str) -> float:
    """Calculate similarity between two titles."""
    try:
        # Simple word overlap similarity
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0
    except Exception as e:
        logger.error(f"Error calculating title similarity: {str(e)}")
        return 0.0


async def remove_duplicate_sources_batch(sources: List[Dict]) -> Dict[str, Any]:
    """Remove duplicate sources using PMID and title similarity heuristics."""
    if not sources:
        return {
            "unique_sources": [],
            "duplicates_removed": 0,
            "message": "No sources available for duplicate removal",
        }

    unique_sources: List[Dict[str, Any]] = []
    seen_pmids: Set[str] = set()
    processed_titles: List[str] = []

    for source in sources:
        metadata = source.get("metadata", {})
        pmid = str(metadata.get("pmid", "")).strip()

        if pmid and pmid in seen_pmids:
            continue

        title = str(metadata.get("title", "")).lower()
        is_duplicate_title = False
        for processed_title in processed_titles:
            similarity = await calculate_title_similarity(title, processed_title)
            if similarity > 0.9:
                is_duplicate_title = True
                break

        if is_duplicate_title:
            continue

        unique_sources.append(source)
        if pmid:
            seen_pmids.add(pmid)
        processed_titles.append(title)

    duplicates_removed = len(sources) - len(unique_sources)
    message = "No duplicate sources detected"
    if duplicates_removed > 0:
        message = f"Removed {duplicates_removed} duplicate sources"

    return {
        "unique_sources": unique_sources,
        "duplicates_removed": duplicates_removed,
        "message": message,
    }


async def assess_journal_quality(journal: str) -> Dict[str, Any]:
    """Assess journal quality and impact metrics."""
    try:
        journal_lower = journal.lower()

        # Tier 1 high-impact journals
        tier1_journals = [
            "nature",
            "science",
            "cell",
            "lancet",
            "new england journal of medicine",
            "jama",
            "nature medicine",
            "nature biotechnology",
        ]

        # Tier 2 specialized journals
        tier2_journals = [
            "clinical pharmacology",
            "drug metabolism",
            "pharmaceutical research",
            "journal of medicinal chemistry",
            "pharmacology",
            "toxicology",
        ]

        # Predatory journal indicators (simplified)
        predatory_indicators = ["predatory", "fake", "bogus"]

        if any(t1 in journal_lower for t1 in tier1_journals):
            quality_tier = "high"
            impact_factor = 10.0  # Simulated
        elif any(t2 in journal_lower for t2 in tier2_journals):
            quality_tier = "moderate"
            impact_factor = 5.0  # Simulated
        else:
            quality_tier = "standard"
            impact_factor = 2.0  # Simulated

        predatory_indicator = any(pred in journal_lower for pred in predatory_indicators)

        return {
            "quality_tier": quality_tier,
            "impact_factor": impact_factor,
            "reputation_score": impact_factor / 10.0,
            "predatory_indicator": predatory_indicator,
        }
    except Exception as e:
        logger.error(f"Error assessing journal quality: {str(e)}")
        return {"quality_tier": "unknown", "impact_factor": 0, "reputation_score": 0, "predatory_indicator": False}


async def calculate_quality_distribution(sources: List[Dict]) -> Dict[str, Any]:
    """Calculate quality distribution of sources."""
    try:
        high_quality = 0
        moderate_quality = 0
        low_quality = 0

        for source in sources:
            quality_flag = _get_source_value(source, "quality_flag", "STANDARD")
            if quality_flag == "HIGH_QUALITY":
                high_quality += 1
            elif quality_flag == "LOW_QUALITY":
                low_quality += 1
            else:
                moderate_quality += 1

        total = len(sources)
        return {
            "high_quality_count": high_quality,
            "moderate_quality_count": moderate_quality,
            "low_quality_count": low_quality,
            "high_quality_ratio": high_quality / total if total > 0 else 0,
            "moderate_quality_ratio": moderate_quality / total if total > 0 else 0,
            "low_quality_ratio": low_quality / total if total > 0 else 0,
        }
    except Exception as e:
        logger.error(f"Error calculating quality distribution: {str(e)}")
        return {"high_quality_ratio": 0, "moderate_quality_ratio": 0, "low_quality_ratio": 0}


async def enrich_sources_with_quality(sources: List[Dict]) -> Dict[str, Any]:
    """Annotate sources with journal quality metadata and return enriched ordering."""
    if not sources:
        return {
            "enriched_sources": [],
            "quality_distribution": {},
            "low_quality_warning": False,
        }

    enriched_sources: List[Dict[str, Any]] = []
    for source in sources:
        metadata = dict(source.get("metadata", {}))
        journal = metadata.get("journal", "")
        assessment = await assess_journal_quality(journal)

        metadata.update(
            {
                "journal_quality": assessment.get("quality_tier"),
                "impact_factor": assessment.get("impact_factor"),
                "journal_reputation": assessment.get("reputation_score"),
            }
        )

        quality_flag = "HIGH_QUALITY" if assessment.get("quality_tier") == "high" else "STANDARD"
        if assessment.get("predatory_indicator"):
            quality_flag = "LOW_QUALITY"
        metadata["quality_flag"] = quality_flag

        enriched_source = dict(source)
        enriched_source["metadata"] = metadata

        if quality_flag == "LOW_QUALITY":
            enriched_source = (
                await append_source_warning(
                    source=enriched_source,
                    warning="Potential predatory journal",
                )
                or enriched_source
            )

        enriched_sources.append(enriched_source)

    sorted_sources = await sort_sources_by_quality(sources=enriched_sources) or enriched_sources
    quality_distribution = await calculate_quality_distribution(sorted_sources)
    low_quality_warning = quality_distribution.get("high_quality_ratio", 0) < 0.3

    return {
        "enriched_sources": sorted_sources,
        "quality_distribution": quality_distribution,
        "low_quality_warning": low_quality_warning,
    }


async def get_medical_journal_database() -> List[str]:
    """Get database of known medical journals."""
    try:
        return [
            "Nature Medicine",
            "The Lancet",
            "New England Journal of Medicine",
            "JAMA",
            "BMJ",
            "Cell",
            "Science",
            "Clinical Pharmacology & Therapeutics",
            "Journal of Clinical Investigation",
            "Nature Biotechnology",
            "Drug Metabolism and Disposition",
            "Pharmaceutical Research",
            "Journal of Medicinal Chemistry",
            "PLoS Medicine",
            "Cochrane Database",
        ]
    except Exception as e:
        logger.error(f"Error getting medical journal database: {str(e)}")
        return []


# Enhanced processing actions


async def classify_study_type(content: str) -> str:
    """Classify study type from content."""
    try:
        content_lower = content.lower()

        if "systematic review" in content_lower or "meta-analysis" in content_lower:
            return "systematic_review"
        elif "randomized controlled trial" in content_lower or "rct" in content_lower:
            return "randomized_controlled_trial"
        elif "clinical trial" in content_lower:
            return "clinical_trial"
        elif "cohort study" in content_lower or "prospective" in content_lower:
            return "cohort_study"
        elif "case-control" in content_lower:
            return "case_control_study"
        elif "case report" in content_lower or "case series" in content_lower:
            return "case_report"
        else:
            return "observational_study"
    except Exception as e:
        logger.error(f"Error classifying study type: {str(e)}")
        return "unknown"


async def extract_drug_entities(content: str) -> List[str]:
    """Extract drug entities from content."""
    try:
        # Simple drug name patterns
        drug_patterns = [
            r"\b[A-Z][a-z]+(?:in|ol|am|ex|ide|one|ine)\b",  # Common drug suffixes
            r"\b(?:aspirin|warfarin|metformin|atorvastatin|lisinopril)\b",  # Common drugs
        ]

        drugs = []
        for pattern in drug_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            drugs.extend(matches)

        return list(set(drugs))  # Remove duplicates
    except Exception as e:
        logger.error(f"Error extracting drug entities: {str(e)}")
        return []


async def extract_cyp_enzymes(content: str) -> List[str]:
    """Extract CYP enzyme mentions from content."""
    try:
        cyp_pattern = r"\bCYP\s*\d+[A-Z]\d*\b"
        cyp_enzymes = re.findall(cyp_pattern, content, re.IGNORECASE)
        return list(set(cyp_enzymes))
    except Exception as e:
        logger.error(f"Error extracting CYP enzymes: {str(e)}")
        return []


async def extract_pk_parameters(content: str) -> Dict[str, Any]:
    """Extract pharmacokinetic parameters from content."""
    try:
        pk_params = {}

        # AUC pattern
        auc_pattern = r"AUC[^0-9]*([0-9.,]+(?:\s*(?:ng|μg|mg)\s*[•*×]?\s*h/mL)?)"
        auc_matches = re.findall(auc_pattern, content, re.IGNORECASE)
        if auc_matches:
            pk_params["auc"] = auc_matches[0]

        # Cmax pattern
        cmax_pattern = r"Cmax[^0-9]*([0-9.,]+(?:\s*(?:ng|μg|mg)/mL)?)"
        cmax_matches = re.findall(cmax_pattern, content, re.IGNORECASE)
        if cmax_matches:
            pk_params["cmax"] = cmax_matches[0]

        # Half-life pattern
        halflife_pattern = r"half-life[^0-9]*([0-9.,]+(?:\s*h(?:ours?)?)?)"
        halflife_matches = re.findall(halflife_pattern, content, re.IGNORECASE)
        if halflife_matches:
            pk_params["half_life"] = halflife_matches[0]

        return pk_params
    except Exception as e:
        logger.error(f"Error extracting PK parameters: {str(e)}")
        return {}


async def assess_evidence_level(study_type: str) -> int:
    """Assess evidence level based on study type."""
    try:
        evidence_levels = {
            "systematic_review": 10,
            "meta_analysis": 10,
            "randomized_controlled_trial": 8,
            "clinical_trial": 7,
            "cohort_study": 6,
            "case_control_study": 5,
            "case_report": 2,
            "observational_study": 4,
            "unknown": 1,
        }

        return evidence_levels.get(study_type, 1)
    except Exception as e:
        logger.error(f"Error assessing evidence level: {str(e)}")
        return 1


async def comprehensive_quality_assessment(source: Dict) -> Dict[str, Any]:
    """Comprehensive quality assessment of a source."""
    try:
        issues = []
        meets_standards = True

        metadata = source.get("metadata", {})
        content = source.get("page_content", "")

        # Check for required metadata
        if not metadata.get("title"):
            issues.append("Missing title")
            meets_standards = False

        if not metadata.get("pmid") and not metadata.get("doi"):
            issues.append("Missing PMID/DOI")
            meets_standards = False

        # Check content quality
        if len(content) < 100:
            issues.append("Content too short")
            meets_standards = False

        # Check for pharmaceutical relevance
        pharma_terms = ["drug", "medication", "pharmaceutical", "clinical", "pharmacokinetics"]
        if not any(term in content.lower() for term in pharma_terms):
            issues.append("Low pharmaceutical relevance")

        return {"meets_standards": meets_standards, "issues": issues, "quality_score": 1.0 - (len(issues) * 0.2)}
    except Exception as e:
        logger.error(f"Error in comprehensive quality assessment: {str(e)}")
        return {"meets_standards": False, "issues": ["Assessment failed"], "quality_score": 0.0}


async def escalate_source_quality_issue() -> None:
    """Escalate source quality issues."""
    try:
        logger.error("ESCALATION: No valid sources found - manual review required")
        # In production, this would trigger escalation procedures
    except Exception as e:
        logger.error(f"Error escalating source quality issue: {str(e)}")


# Response formatting actions


async def structure_moa_information(response: str) -> str:
    """Structure mechanism of action information."""
    try:
        if "mechanism of action" in response.lower() or "moa" in response.lower():
            # Add structure if not already present
            if "**Mechanism of Action:**" not in response:
                response = response.replace("mechanism of action", "**Mechanism of Action:**")
                response = response.replace("MOA", "**Mechanism of Action:**")
        return response
    except Exception as e:
        logger.error(f"Error structuring MOA information: {str(e)}")
        return response


async def structure_interaction_information(response: str) -> str:
    """Structure drug interaction information."""
    try:
        if "interaction" in response.lower():
            # Add structure if not already present
            if "**Drug Interaction:**" not in response:
                response = response.replace("drug interaction", "**Drug Interaction:**")
                response = response.replace("interaction", "**Interaction:**")
        return response
    except Exception as e:
        logger.error(f"Error structuring interaction information: {str(e)}")
        return response


async def structure_pk_information(response: str) -> str:
    """Structure pharmacokinetic information."""
    try:
        pk_terms = ["pharmacokinetic", "absorption", "distribution", "metabolism", "excretion"]
        if any(term in response.lower() for term in pk_terms):
            # Add structure if not already present
            if "**Pharmacokinetics:**" not in response:
                response = response.replace("pharmacokinetics", "**Pharmacokinetics:**")
                response = response.replace("pharmacokinetic", "**Pharmacokinetic:**")
        return response
    except Exception as e:
        logger.error(f"Error structuring PK information: {str(e)}")
        return response


async def sort_sources_by_quality(sources: List[Dict]) -> List[Dict]:
    """Sort sources by quality metrics."""
    try:
        # Simple sorting by impact factor (simulated)
        def get_quality_score(source):
            quality_flag = _get_source_value(source, "quality_flag", "STANDARD")
            impact_factor = _get_source_value(source, "impact_factor", 0)

            if quality_flag == "HIGH_QUALITY":
                return impact_factor + 10
            elif quality_flag == "LOW_QUALITY":
                return impact_factor - 5
            else:
                return impact_factor

        sorted_sources = sorted(sources, key=get_quality_score, reverse=True)
        return sorted_sources
    except Exception as e:
        logger.error(f"Error sorting sources by quality: {str(e)}")
        return sources


# Source validation actions for retrieval rails


async def validate_source_authenticity(sources: List[Dict]) -> Dict[str, Any]:
    """Validate authenticity of retrieved sources."""
    try:
        if not sources:
            return {
                "authentic_sources": 0,
                "suspicious_sources": 0,
                "authenticity_ratio": 0,
                "details": [],
                "issues": ["No sources provided for validation"],
            }

        authentic_count = 0
        suspicious_count = 0
        issues: List[str] = []
        details: List[Dict[str, Any]] = []
        known_journals = {journal.lower() for journal in await get_medical_journal_database()}

        for source in sources:
            metadata = source.get("metadata", {})
            pmid = str(metadata.get("pmid", "")).strip()
            doi = str(metadata.get("doi", "")).strip()
            journal = metadata.get("journal", "").strip().lower()
            url = metadata.get("source_url", "") or metadata.get("url", "")

            reasons = []

            if pmid:
                if not re.match(r"^\d{7,8}$", pmid):
                    reasons.append("Invalid PMID format")
            else:
                reasons.append("Missing PMID")

            if doi:
                if not re.match(r"^10\.\d{4,}/.+", doi, re.IGNORECASE):
                    reasons.append("Invalid DOI format")
            else:
                reasons.append("Missing DOI")

            if journal:
                matched_journal = any(journal == known for known in known_journals)
                if not matched_journal:
                    reasons.append("Journal not found in trusted medical list")
            else:
                reasons.append("Missing journal information")

            if url and any(term in url.lower() for term in ["blog", "wordpress", "medium.com", "wikipedia"]):
                reasons.append("Source URL appears non-scholarly")

            detail_entry = {
                "title": metadata.get("title", "Unknown title"),
                "pmid": pmid or None,
                "doi": doi or None,
                "journal": metadata.get("journal", "Unknown journal"),
                "authentic": len(reasons) == 0,
                "reasons": reasons,
            }
            details.append(detail_entry)

            if reasons:
                suspicious_count += 1
                issues.append(f"Source '{detail_entry['title']}' flagged: {', '.join(reasons)}")
            else:
                authentic_count += 1

        authenticity_ratio = authentic_count / len(sources) if sources else 0

        return {
            "authentic_sources": authentic_count,
            "suspicious_sources": suspicious_count,
            "authenticity_ratio": round(authenticity_ratio, 2),
            "details": details,
            "issues": issues,
        }
    except Exception as e:
        logger.error(f"Error validating source authenticity: {str(e)}")
        return {
            "authentic_sources": 0,
            "suspicious_sources": len(sources),
            "authenticity_ratio": 0,
            "details": [],
            "issues": [f"Validation error: {str(e)}"],
        }


async def verify_pmid(source: Dict) -> Dict[str, Any]:
    """Verify PMID format and validity."""
    try:
        pmid = source.get("metadata", {}).get("pmid", "")

        if not pmid:
            return {"valid": False, "reason": "No PMID provided"}

        # PMID format validation
        if re.match(r"^\d{7,8}$", str(pmid)):
            return {"valid": True, "pmid": pmid}
        else:
            return {"valid": False, "reason": "Invalid PMID format"}
    except Exception as e:
        logger.error(f"Error verifying PMID: {str(e)}")
        return {"valid": False, "reason": "PMID verification failed"}


async def verify_journal_authenticity(source: Dict) -> Dict[str, Any]:
    """Verify journal authenticity."""
    try:
        journal = source.get("metadata", {}).get("journal", "").lower()

        if not journal:
            return {"valid": False, "reason": "No journal provided"}

        # Known reputable medical journals (simplified list)
        reputable_journals = [
            "nature",
            "science",
            "cell",
            "lancet",
            "new england journal of medicine",
            "jama",
            "bmj",
            "plos",
            "cochrane",
            "clinical pharmacology",
        ]

        is_reputable = any(rep_journal in journal for rep_journal in reputable_journals)

        return {"valid": True, "reputable": is_reputable, "journal": journal}  # Assume valid unless clearly predatory
    except Exception as e:
        logger.error(f"Error verifying journal: {str(e)}")
        return {"valid": False, "reason": "Journal verification failed"}


_COLANG_ACTION_PATTERN = re.compile(r"\bexecute\s+([a-zA-Z_][\w]*)")


def _collect_actions_from_colang(rails_root: Path) -> Set[str]:
    """Parse Colang rails to extract referenced action names."""
    actions: Set[str] = set()

    try:
        colang_files = list(rails_root.glob("*.co"))
    except OSError as exc:  # pragma: no cover - filesystem access issues
        logger.warning(f"Failed to enumerate Colang rails: {exc}")
        return actions

    for colang_file in sorted(colang_files):
        try:
            content = colang_file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(f"Failed to read Colang file {colang_file}: {exc}")
            continue

        for match in _COLANG_ACTION_PATTERN.finditer(content):
            actions.add(match.group(1))

    return actions


def _self_check_registered_actions(registered_actions: Set[str]) -> Set[str]:
    """Compare registered actions against Colang flows and log discrepancies."""
    rails_root = Path(__file__).resolve().parent / "rails"
    referenced_actions = _collect_actions_from_colang(rails_root)

    missing = referenced_actions - registered_actions
    if missing:
        logger.warning(
            "Actions referenced in Colang flows but not registered: %s",
            ", ".join(sorted(missing)),
        )
    else:
        logger.info("All Colang-referenced actions have registrations.")

    unused = registered_actions - referenced_actions
    if unused:
        logger.debug(
            "Registered actions not referenced in flows: %s",
            ", ".join(sorted(unused)),
        )

    return missing


def init(app):
    """
    Initialize and register all medical safety actions with NeMo Guardrails.

    Args:
        app: NeMo Guardrails application instance

    Returns:
        Dict containing registration status and action tracking information
    """
    registration_status = {
        "total_expected": 0,
        "successfully_registered": [],
        "missing_actions": [],
        "wrapped_actions": [],
        "registration_errors": [],
        "colang_referenced_actions": set(),
        "actions_status": {},
    }

    try:
        medical_actions = MedicalSafetyActions()

        # Define names and attribute mappings for robust registration
        names = [
            ("check_medical_context", "check_medical_context"),
            ("detect_medical_jailbreak", "detect_medical_jailbreak"),
            ("scan_medical_pii", "scan_medical_pii"),
            ("mask_medical_pii", "mask_medical_pii"),
            ("get_medical_disclaimer", "get_medical_disclaimer"),
            ("medical_hallucination_check", "medical_hallucination_check"),
        ]

        # Build expected actions list tolerating missing attributes
        expected_actions = []
        missing_actions = []

        # Process medical actions class methods
        for public_name, attr_name in names:
            func = getattr(medical_actions, attr_name, None)
            if func is None:
                missing_actions.append(public_name)
                continue
            expected_actions.append((public_name, func))

        # Add standalone functions
        standalone_functions = [
            # Helper functions
            ("classify_medical_query_type", classify_medical_query_type),
            ("validate_pharmaceutical_context", validate_pharmaceutical_context),
            ("assess_medical_toxicity", assess_medical_toxicity),
            ("validate_against_pubmed_sources", validate_against_pubmed_sources),
            ("assess_regulatory_compliance", assess_regulatory_compliance),
            ("process_pubmed_sources", process_pubmed_sources),
            ("filter_pharmaceutical_relevance_batch", filter_pharmaceutical_relevance_batch),
            ("remove_duplicate_sources_batch", remove_duplicate_sources_batch),
            ("enrich_sources_with_quality", enrich_sources_with_quality),
            ("evaluate_hallucination_risk", evaluate_hallucination_risk),
            ("evaluate_fact_check_result", evaluate_fact_check_result),
            ("evaluate_regulatory_compliance_flow", evaluate_regulatory_compliance_flow),
            ("filter_sensitive_response_content", filter_sensitive_response_content),
            ("append_pharmaceutical_warnings_response", append_pharmaceutical_warnings_response),
            ("append_evidence_quality_summary", append_evidence_quality_summary),
            ("ensure_citation_block", ensure_citation_block),
            ("enhance_response_quality_format", enhance_response_quality_format),
            # Additional missing actions
            ("classify_medical_response_type", classify_medical_response_type),
            ("filter_sensitive_medical_info", filter_sensitive_medical_info),
            ("assess_evidence_levels", assess_evidence_levels),
            ("format_source_citations", format_source_citations),
            ("validate_citations", validate_citations),
            ("comprehensive_safety_check", comprehensive_safety_check),
            ("log_safety_event", log_safety_event),
            ("log_medical_response", log_medical_response),
            ("validate_source_authenticity", validate_source_authenticity),
            ("verify_pmid", verify_pmid),
            ("verify_journal_authenticity", verify_journal_authenticity),
            # Retrieval and processing actions
            ("update_source_metadata", update_source_metadata),
            ("set_source_flag", set_source_flag),
            ("append_source_warning", append_source_warning),
            ("request_additional_sources", request_additional_sources),
            ("assess_pharmaceutical_relevance", assess_pharmaceutical_relevance),
            ("identify_pharmaceutical_indicators", identify_pharmaceutical_indicators),
            ("suggest_query_refinement", suggest_query_refinement),
            ("calculate_title_similarity", calculate_title_similarity),
            ("assess_journal_quality", assess_journal_quality),
            ("calculate_quality_distribution", calculate_quality_distribution),
            ("get_medical_journal_database", get_medical_journal_database),
            ("ensure_disclaimer", ensure_disclaimer),
            # Processing actions
            ("classify_study_type", classify_study_type),
            ("extract_drug_entities", extract_drug_entities),
            ("extract_cyp_enzymes", extract_cyp_enzymes),
            ("extract_pk_parameters", extract_pk_parameters),
            ("assess_evidence_level", assess_evidence_level),
            ("comprehensive_quality_assessment", comprehensive_quality_assessment),
            ("escalate_source_quality_issue", escalate_source_quality_issue),
            # Formatting actions
            ("structure_moa_information", structure_moa_information),
            ("structure_interaction_information", structure_interaction_information),
            ("structure_pk_information", structure_pk_information),
            ("sort_sources_by_quality", sort_sources_by_quality),
            # Journal quality database
            ("get_medical_journal_database", get_medical_journal_database),
        ]

        expected_actions.extend(standalone_functions)
        registration_status["total_expected"] = len(expected_actions)

        # Track registered actions for verification
        registered_actions = []
        registration_errors = []

        # Register all actions
        for action_name, action_func in expected_actions:
            try:
                app.register_action(action_func, action_name)
                registered_actions.append(action_name)
                registration_status["successfully_registered"].append(action_name)
                registration_status["actions_status"][action_name] = {
                    "status": "registered",
                    "type": "direct",
                    "error": None,
                }
            except Exception as e:
                error_msg = f"Failed to register action '{action_name}': {e}"
                logger.warning(error_msg)
                missing_actions.append(action_name)
                registration_errors.append(error_msg)
                registration_status["actions_status"][action_name] = {
                    "status": "failed",
                    "type": "direct",
                    "error": str(e),
                }

        # Log registry status
        logger.info(
            "Medical safety actions registered: %s successful, %s failed",
            len(registered_actions),
            len(missing_actions),
        )
        if missing_actions:
            logger.warning("Missing actions: %s", ", ".join(sorted(missing_actions)))

        # Add a catch-all error wrapper for unregistered actions
        def create_safe_action_wrapper(action_name):
            """Create a safe wrapper that returns a default payload for unregistered actions."""

            async def safe_wrapper(*args, **kwargs):
                logger.warning(f"Unregistered action called: {action_name}")
                # Return a safe default payload
                return {
                    "error": f"Action '{action_name}' is not registered",
                    "safe_default": True,
                    "payload": "Action execution skipped due to missing registration",
                }

            return safe_wrapper

        registered_set = set(registered_actions)
        missing_from_flows = _self_check_registered_actions(registered_set)
        actions_requiring_wrappers = set(missing_actions) | missing_from_flows

        registration_status["missing_actions"] = list(missing_actions)
        registration_status["registration_errors"] = registration_errors
        registration_status["colang_referenced_actions"] = missing_from_flows

        # Enhanced validation as per Comment 5: fail fast in dev mode and emit warnings
        dev_mode = os.getenv("GUARDRAILS_DEV_MODE", "false").lower() == "true"

        if missing_from_flows:
            warning_msg = f"Colang flows reference unregistered actions: {', '.join(sorted(missing_from_flows))}"
            logger.warning(warning_msg)

            if dev_mode:
                error_msg = f"DEV MODE: Failing fast due to action mismatches. {warning_msg}"
                logger.error(error_msg)
                registration_status["registration_errors"].append(error_msg)
                raise ValueError(error_msg)

        unused_actions = registered_set - registration_status.get("colang_referenced_actions", set())
        if unused_actions:
            unused_msg = f"Registered actions not referenced in Colang flows: {', '.join(sorted(unused_actions))}"
            logger.info(unused_msg)
            registration_status["unused_actions"] = list(unused_actions)

        for action_name in sorted(actions_requiring_wrappers):
            if action_name in registered_set:
                continue
            try:
                app.register_action(create_safe_action_wrapper(action_name), action_name)
                registered_actions.append(action_name)
                registered_set.add(action_name)
                registration_status["wrapped_actions"].append(action_name)
                registration_status["actions_status"][action_name] = {
                    "status": "wrapped",
                    "type": "fallback",
                    "error": None,
                }
                logger.info(f"Registered safe wrapper for unregistered action: {action_name}")
            except Exception as e:
                error_msg = f"Failed to register safe wrapper for '{action_name}': {e}"
                logger.error(error_msg)
                registration_status["actions_status"][action_name] = {
                    "status": "failed",
                    "type": "wrapper",
                    "error": str(e),
                }

        logger.info("Medical safety actions registered successfully")

        # Store registration status on the app object for retrieval
        if hasattr(app, "actions_registration_status"):
            app.actions_registration_status = registration_status
        else:
            setattr(app, "actions_registration_status", registration_status)

        return registration_status

    except Exception as e:
        error_msg = f"Error registering medical safety actions: {str(e)}"
        logger.error(error_msg)
        registration_status["registration_errors"].append(error_msg)
        return registration_status


def get_medical_journal_database() -> Dict[str, Any]:
    """
    Get medical journal quality database with fallback to built-in defaults.

    Reads from guardrails/kb/journals.yml if present, otherwise uses built-in data.

    Returns:
        Dictionary containing journal quality tiers and quality indicators
    """
    try:
        # Try to read from the journal database file
        journals_file = Path(__file__).parent / "kb" / "journals.yml"

        if journals_file.exists():
            try:
                import yaml

                with journals_file.open("r", encoding="utf-8") as f:
                    journal_data = yaml.safe_load(f)
                    if journal_data and "journals" in journal_data:
                        logger.info(f"Loaded journal database from {journals_file}")
                        return journal_data
                    else:
                        logger.warning(f"Invalid journal database format in {journals_file}")
            except ImportError:
                logger.warning("PyYAML not available, using built-in journal database")
            except Exception as e:
                logger.warning(f"Error reading journal database from {journals_file}: {e}")

        # Fallback to built-in defaults
        logger.info("Using built-in journal quality database")
        return {
            "journals": {
                "tier1": [
                    "Nature Medicine",
                    "New England Journal of Medicine",
                    "The Lancet",
                    "Science",
                    "Nature",
                    "Cell",
                    "JAMA",
                    "British Medical Journal",
                    "Clinical Pharmacology & Therapeutics",
                    "Nature Reviews Drug Discovery",
                ],
                "tier2": [
                    "Journal of Medicinal Chemistry",
                    "Pharmaceutical Research",
                    "Clinical Pharmacokinetics",
                    "European Journal of Pharmaceutical Sciences",
                    "Drug Discovery Today",
                    "Journal of Pharmacology and Experimental Therapeutics",
                    "Molecular Pharmaceutics",
                    "AAPS Journal",
                ],
                "tier3": [
                    "European Journal of Clinical Pharmacology",
                    "Journal of Clinical Pharmacy and Therapeutics",
                    "International Journal of Clinical Pharmacology and Therapeutics",
                    "Pharmacotherapy",
                    "Journal of Pharmaceutical Sciences",
                ],
                "tier4": [
                    "Indian Journal of Pharmaceutical Sciences",
                    "Current Pharmaceutical Design",
                    "Open Access journals (various)",
                ],
            },
            "impact_factor_ranges": {
                "tier1": "Usually > 15",
                "tier2": "Usually 3-15",
                "tier3": "Usually 1-3",
                "tier4": "Usually < 1 or unranked",
            },
            "quality_indicators": {
                "positive": [
                    "peer-reviewed",
                    "indexed in PubMed",
                    "indexed in MEDLINE",
                    "impact factor available",
                    "published by reputable publisher",
                ],
                "negative": [
                    "predatory journal",
                    "pay-to-publish without peer review",
                    "not indexed in major databases",
                    "unclear editorial process",
                ],
            },
            "source": "built-in defaults",
        }

    except Exception as e:
        logger.error(f"Error in get_medical_journal_database: {e}")
        return {"journals": {"tier1": [], "tier2": [], "tier3": [], "tier4": []}, "error": str(e)}
