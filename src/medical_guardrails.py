"""Medical safety guardrails and validation module for pharmaceutical research systems."""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRUTHY_VALUES = {"true", "1", "yes", "on"}
_FALSEY_VALUES = {"false", "0", "no", "off"}


def _env_flag(name: str, *, default: bool = True) -> bool:
    """Return boolean flag from environment with sensible defaults."""

    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default

    value = raw.strip().lower()
    if value in _TRUTHY_VALUES:
        return True
    if value in _FALSEY_VALUES:
        return False
    return default


def resolve_medical_guardrails_config_path() -> Path:
    """Resolve the medical guardrails JSON config path deterministically.

    Resolution order:
    1. Explicit MEDICAL_GUARDRAILS_CONFIG_PATH environment override
    2. Repository-relative config/medical_guardrails.json
    """

    override = os.getenv("MEDICAL_GUARDRAILS_CONFIG_PATH")
    if override and override.strip():
        return Path(override).expanduser().resolve()

    return (Path(__file__).resolve().parent.parent / "config" / "medical_guardrails.json").resolve()


# Check if medical guardrails are enabled via environment variable (default: enabled)
ENABLE_MEDICAL_GUARDRAILS = _env_flag("ENABLE_MEDICAL_GUARDRAILS", default=True)

# Optional Presidio imports for advanced PII/PHI detection
PRESIDIO_AVAILABLE = False
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    PRESIDIO_AVAILABLE = True
    logger.info("Presidio detected - will use for advanced PII/PHI detection when enabled")
except ImportError:
    logger.info("Presidio not available - using regex-based PII/PHI detection")
    AnalyzerEngine = None
    AnonymizerEngine = None
    NlpEngineProvider = None

# Optional NeMo Guardrails imports
NEMO_GUARDRAILS_AVAILABLE = False
_NEMO_IMPORT_ERROR = None
if ENABLE_MEDICAL_GUARDRAILS:
    try:
        from nemoguardrails import LLMRails, RailsConfig

        NEMO_GUARDRAILS_AVAILABLE = True
    except ImportError as e:
        _NEMO_IMPORT_ERROR = e
        logger.info("NeMo Guardrails not available - using lightweight validators only")
        LLMRails = None
        RailsConfig = None
else:
    logger.info("Medical guardrails disabled via ENABLE_MEDICAL_GUARDRAILS environment variable")
    logger.info("Using lightweight validators only - skipping heavy medical dependencies")

try:
    from langchain_core.language_models.llms import LLM as _LangChainLLM
except ImportError:  # pragma: no cover - optional dependency
    _LangChainLLM = None


class _GuardrailsNullLLM(_LangChainLLM if _LangChainLLM is not None else object):
    """Minimal LLM stub that satisfies NeMo Guardrails when real providers are unavailable."""

    def __init__(self) -> None:  # pragma: no cover - trivial initializer
        if hasattr(super(), "__init__"):
            super().__init__()

    @property
    def _llm_type(self) -> str:  # pragma: no cover - simple metadata
        return "guardrails-dummy"

    @property
    def _identifying_params(self) -> dict[str, Any]:  # pragma: no cover - simple metadata
        return {"provider": "dummy"}

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> str:
        return ""


@dataclass
class ValidationResult:
    """Result of medical validation check."""

    is_valid: bool
    severity: str  # "low", "medium", "high", "critical"
    issues: list[str]
    recommendations: list[str]
    metadata: dict[str, Any]


@dataclass
class PIIDetectionResult:
    """Result of PII/PHI detection."""

    detected: bool
    entities: list[dict[str, Any]]
    masked_text: str
    confidence: float


class MedicalGuardrails:
    """Medical safety guardrails for pharmaceutical research applications.

    Provides comprehensive safety validation including PII/PHI detection,
    medical context validation, jailbreak detection, and regulatory compliance
    checks for medical and pharmaceutical applications.

    Example usage:
        guardrails = MedicalGuardrails("config/medical_guardrails.json")
        result = guardrails.validate_medical_query("What is the dosage for warfarin?")
    """

    def __init__(
        self,
        config_path: str,
        actions: Any | None = None,
        enable_nemo_guardrails: bool = True,
        guardrails_root: str | None = None,
        nemo_config_path: str | None = None,
        enabled: bool | None = None,
    ):
        """Initialize medical guardrails with configuration.

        Args:
            config_path: Path to configuration file with medical validation rules
            actions: Optional actions object for NeMo Guardrails integration
            enable_nemo_guardrails: Whether to initialize NeMo Guardrails runtime
            guardrails_root: Optional directory containing NeMo Guardrails assets (config.yml, rails)
            nemo_config_path: Optional direct path to a NeMo Guardrails configuration YAML file
            enabled: Optional explicit enable/disable flag. If None, falls back to environment variable detection.
        """
        self.config_path = Path(config_path)
        self.actions = actions
        self.config = self._load_config()
        self.nemo_rails = None
        self.guardrails_root = Path(guardrails_root) if guardrails_root else None
        self.nemo_config_path = Path(nemo_config_path) if nemo_config_path else None
        self.guardrails_actions_status = None  # Will store action registration status

        # Set enabled flag - if explicitly provided, use it; otherwise fall back to environment variable
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = ENABLE_MEDICAL_GUARDRAILS

        # Initialize NeMo Guardrails if available and enabled
        if self.enabled and enable_nemo_guardrails and NEMO_GUARDRAILS_AVAILABLE:
            try:
                self._initialize_nemo_guardrails()
            except Exception as e:
                logger.warning(f"Failed to initialize NeMo Guardrails: {e}")
                logger.info("Falling back to lightweight validators")
        elif not self.enabled:
            logger.info("Medical guardrails explicitly disabled")
            logger.info("Using lightweight validators only - skipping heavy medical dependencies")

        # Add runtime check/log for missing medical dependencies
        if self.enabled:
            missing_deps = []
            if not PRESIDIO_AVAILABLE:
                missing_deps.append("presidio-analyzer, presidio-anonymizer")
            if enable_nemo_guardrails and not NEMO_GUARDRAILS_AVAILABLE:
                missing_deps.append("nemoguardrails")

            if missing_deps:
                missing_str = ", ".join(missing_deps)
                error_msg = (
                    f"Medical guardrails enabled (ENABLE_MEDICAL_GUARDRAILS=true) but required dependencies are missing: {missing_str}. "
                    f"Install with: pip install -r requirements-medical.txt"
                )
                if "nemoguardrails" in missing_deps:
                    # Critical dependencies missing - raise error
                    raise ImportError(error_msg)

                logger.warning(error_msg)
                logger.warning("Falling back to regex-based PII/PHI detection")

        # Medical context patterns
        self.medical_patterns = {
            "drug_names": [
                r"\b(?:acetaminophen|ibuprofen|aspirin|warfarin|metformin|lisinopril|atorvastatin)\b",
                r"\b[a-z]+(?:cillin|mycin|azole|pril|statin|olol)\b",
                r"\b\w+(?:mab|nib|tide)\b",  # Biologics patterns
            ],
            "medical_conditions": [
                r"\b(?:hypertension|diabetes|depression|anxiety|cancer|pneumonia)\b",
                r"\b(?:cardiovascular|respiratory|neurological|psychiatric)\b",
                r"\b(?:diagnosis|treatment|therapy|medication|prescription)\b",
            ],
            "dosage_patterns": [
                r"\b\d+\s*(?:mg|mcg|g|ml|units?)\b",
                r"\b(?:once|twice|three times?)\s+(?:daily|per day|a day)\b",
                r"\bevery\s+\d+\s+hours?\b",
            ],
            "clinical_terms": [
                r"\b(?:clinical trial|study|research|review|efficacy|safety|adverse)\b",
                r"\b(?:pharmacokinetics?|pharmacodynamics?|bioavailability)\b",
                r"\b(?:contraindication|interactions?|side effect)\b",
            ],
        }

        # PII/PHI patterns with medical context anchoring
        self.pii_patterns = {
            "names": [
                r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # Simple name pattern
                r"\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+\b",
            ],
            "phone_numbers": [r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", r"\(\d{3}\)\s*\d{3}[-.]?\d{4}\b"],
            "email": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
            "medical_record_numbers": [
                # Anchor MRN patterns to medical context terms when Presidio unavailable
                r"(?:patient|medical record|MRN|record)[\s:]+\d{5,8}\b",
                r"\bMRN[\s:]*\d+\b",
                r"\bmedical record(?:\s+number)?[\s:]*\d+\b",
            ],
            "ssn": [
                # Anchor SSN to medical/insurance context to reduce false positives
                r"(?:SSN|social security|insurance)[\s:]*\d{3}-\d{2}-\d{4}\b",
                r"(?:patient|billing|insurance)[\s\w]*SSN[\s:]*\d{3}-\d{2}-\d{4}\b",
                r"\b\d{3}-\d{2}-\d{4}\b",  # Keep general pattern but lower priority
            ],
            "dates": [
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
                r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2,4}\b",
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4}\b",
            ],
        }
        self.pii_mask_flags = {
            "names": 0,
            "phone_numbers": re.IGNORECASE,
            "email": re.IGNORECASE,
            "medical_record_numbers": re.IGNORECASE,
            "ssn": re.IGNORECASE,
            "dates": re.IGNORECASE,
        }

        # Jailbreak detection patterns
        self.jailbreak_patterns = [
            r"ignore.*(?:previous|above|prior)\s+(?:instructions?|rules?|guidelines?)",
            r"forget.*(?:instructions?|rules?|guidelines?|constraints?)",
            r"act\s+as\s+(?:if|though).*not\s+(?:ai|assistant|bot)",
            r"pretend.*(?:human|doctor|medical professional)",
            r"roleplay.*(?:doctor|physician|medical)",
            r"(?:jailbreak|bypass|circumvent|override).*(?:safety|guidelines?|rules?)",
            r"ignore.*(?:safety|medical|ethical)\s+(?:guidelines?|protocols?)",
            r"provide.*(?:illegal|harmful|dangerous).*(?:advice|information|guidance)",
        ]

        # Regulatory compliance keywords
        self.regulatory_keywords = {
            "fda_warnings": [
                "black box",
                "boxed warning",
                "contraindicated",
                "serious adverse",
                "life-threatening",
                "fatal",
                "death",
                "suicide",
            ],
            "prescription_only": [
                "prescription only",
                "rx only",
                "controlled substance",
                "schedule II",
                "schedule III",
                "schedule IV",
            ],
            "clinical_trials": [
                "investigational",
                "experimental",
                "off-label",
                "unlabeled use",
                "clinical trial",
                "investigational new drug",
            ],
        }

    def _initialize_nemo_guardrails(self):
        """Initialize NeMo Guardrails runtime using the configured asset paths.

        Checks for guardrails paths in the following order:
        1. Explicitly provided nemo_config_path
        2. Explicitly provided guardrails_root
        3. GUARDRAILS_ROOT environment variable
        4. Package-relative path (src/../guardrails)
        5. Current directory guardrails folder

        Raises:
            Exception: If NeMo Guardrails initialization fails
        """
        try:
            local_actions_path = Path(__file__).parent.parent / "guardrails" / "actions.py"

            # Determine guardrails root path
            if self.nemo_config_path:
                guardrails_config_path = self.nemo_config_path
                guardrails_root = guardrails_config_path.parent
            elif self.guardrails_root:
                guardrails_root = self.guardrails_root
                guardrails_config_path = guardrails_root / "config.yml"
            else:
                env_guardrails_root = os.environ.get("GUARDRAILS_ROOT")
                if env_guardrails_root:
                    guardrails_root = Path(env_guardrails_root)
                else:
                    package_guardrails = Path(__file__).parent.parent / "guardrails"
                    if package_guardrails.exists():
                        guardrails_root = package_guardrails
                    else:
                        guardrails_root = Path("guardrails")

                guardrails_config_path = guardrails_root / "config.yml"

            # Log the resolved paths for transparency
            logger.info("NeMo Guardrails initialization - resolved guardrails_root: %s", guardrails_root.absolute())
            logger.info("NeMo Guardrails initialization - config.yml path: %s", guardrails_config_path.absolute())

            # Check config presence and provide remediation steps if missing
            if not guardrails_config_path.exists():
                logger.info("NeMo Guardrails config.yml not found at %s", guardrails_config_path.absolute())
                logger.info(
                    "To enable guardrails: set GUARDRAILS_ROOT environment variable or supply nemo_config_path parameter"
                )
                logger.info("Continuing with fallback validation only")
                return

            # Initialize NeMo Guardrails
            config = RailsConfig.from_path(str(guardrails_root))
            try:
                self.nemo_rails = LLMRails(config)
            except Exception as model_error:  # pragma: no cover - fallback for missing providers
                logger.warning(
                    "NeMo Guardrails model initialization failed (%s); using dummy LLM fallback.",
                    model_error,
                )
                dummy_llm = _GuardrailsNullLLM() if _LangChainLLM is not None else None
                self.nemo_rails = LLMRails(config, llm=dummy_llm)

            # Load and register actions
            actions_candidates = []
            guardrails_actions_path = guardrails_root / "actions.py"
            if guardrails_actions_path.exists():
                actions_candidates.append(guardrails_actions_path)
            if local_actions_path.exists() and local_actions_path not in actions_candidates:
                actions_candidates.append(local_actions_path)

            actions_loaded = False
            for actions_path in actions_candidates:
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"guardrails_actions_{actions_path.stem}",
                        actions_path,
                    )
                    if not spec or not spec.loader:
                        logger.warning("Unable to create import spec for guardrails actions at %s", actions_path)
                        continue

                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)  # type: ignore[attr-defined]
                    init_actions = getattr(module, "init", None)
                    if callable(init_actions):
                        # Capture the action registration status returned by init
                        registration_status = init_actions(self.nemo_rails)
                        if registration_status and isinstance(registration_status, dict):
                            self.guardrails_actions_status = registration_status
                            # Log the counts of actions loaded
                            total_expected = registration_status.get("total_expected", 0)
                            registered_count = len(registration_status.get("successfully_registered", []))
                            wrapped_count = len(registration_status.get("wrapped_actions", []))
                            logger.info(
                                "NeMo Guardrails actions loaded: %d expected, %d registered, %d wrapped",
                                total_expected,
                                registered_count,
                                wrapped_count,
                            )
                        logger.info("NeMo Guardrails actions registered from %s", actions_path)
                        actions_loaded = True
                        break
                    logger.warning(
                        "Guardrails actions module at %s does not expose an init(app) callable", actions_path
                    )
                except Exception as actions_error:
                    logger.exception("Failed to load guardrails actions from %s", actions_path)

            if not actions_loaded:
                logger.warning(
                    "NeMo Guardrails actions module could not be loaded from expected paths: %s", actions_candidates
                )

            # Log successful initialization with action counts
            if self.guardrails_actions_status:
                action_summary = (
                    f"with {len(self.guardrails_actions_status.get('successfully_registered', []))} actions"
                )
            else:
                action_summary = "without custom actions"
            logger.info("NeMo Guardrails initialized successfully %s", action_summary)

        except Exception as e:
            logger.error("Error initializing NeMo Guardrails: %s", e)
            raise

    async def run_input_validation_rails(self, user_message: str) -> dict[str, Any]:
        """Run input validation using NeMo Guardrails flows.

        Args:
            user_message: User input to validate

        Returns:
            Dictionary containing validation results from rails flows
        """
        # If guardrails are disabled, return a simple pass-through result
        if not self.enabled:
            return {
                "is_valid": True,
                "severity": "low",
                "issues": [],
                "recommendations": [],
                "nemo_rails_used": False,
                "sanitized_query": user_message,
                "metadata": {"guardrails_enabled": False, "note": "Medical guardrails are disabled"},
            }

        try:
            if not self.nemo_rails:
                logger.info("NeMo Guardrails not available, using fallback validation")
                fallback_result = self.validate_medical_query(user_message)
                # Normalize the fallback return shape to match rails-powered path
                return {
                    "is_valid": fallback_result.get("is_valid", True),
                    "severity": fallback_result.get("severity", "low"),
                    "issues": fallback_result.get("issues", []),
                    "recommendations": fallback_result.get("recommendations", []),
                    "nemo_rails_used": False,
                    "sanitized_query": fallback_result.get("sanitized_query", user_message),
                    "metadata": {
                        **fallback_result.get("metadata", {}),
                        "flow_runs": {},
                        "checks_performed": fallback_result.get("checks_performed", {}),
                        "fallback_used": True,
                    },
                }

            flow_runs: dict[str, Any] = {}
            rails_context = await self._execute_flow("medical input validation", user_message=user_message)
            flow_runs["medical_input_validation"] = rails_context
            flow_variables = self._extract_flow_context(rails_context)

            sanitized_query = user_message or ""
            sanitized_candidates: list[str] = []

            def _register_sanitized_candidate(candidate: Any) -> None:
                if isinstance(candidate, str) and candidate.strip():
                    sanitized_candidates.append(candidate)

            _register_sanitized_candidate(flow_variables.get("user_message"))

            jailbreak_flow = await self._execute_flow("check jailbreak attempts", user_message=user_message)
            flow_runs["check_jailbreak_attempts"] = jailbreak_flow

            pii_flow = await self._execute_flow("pii detection and masking", user_message=user_message)
            flow_runs["pii_detection_and_masking"] = pii_flow
            pii_flow_context = self._extract_flow_context(pii_flow)
            _register_sanitized_candidate(pii_flow_context.get("user_message"))
            _register_sanitized_candidate(pii_flow_context.get("masked_text"))

            pharma_flow = await self._execute_flow("pharmaceutical context validation", user_message=user_message)
            flow_runs["pharmaceutical_context_validation"] = pharma_flow

            toxicity_flow = await self._execute_flow("toxicity screening", user_message=user_message)
            flow_runs["toxicity_screening"] = toxicity_flow

            issues: list[str] = []
            recommendations: list[str] = []
            severity = "low"

            def escalate(new_severity: str) -> None:
                nonlocal severity
                if self._severity_rank(new_severity) > self._severity_rank(severity):
                    severity = new_severity

            medical_context = flow_variables.get("medical_context")
            if not medical_context:
                medical_context = await self._safe_invoke_action("check_medical_context", query=user_message)

            pharma_context = flow_variables.get("pharma_context")
            if not pharma_context:
                pharma_context = await self._safe_invoke_action("validate_pharmaceutical_context", query=user_message)

            pii_scan = flow_variables.get("pii_scan")
            if not pii_scan:
                pii_scan = await self._safe_invoke_action("scan_medical_pii", text=user_message)

            jailbreak_detected = flow_variables.get("jailbreak_detected")
            if jailbreak_detected is None:
                jailbreak_detected = await self._safe_invoke_action("detect_medical_jailbreak", query=user_message)

            toxicity_score = flow_variables.get("toxicity_score")
            if toxicity_score is None:
                toxicity_score = await self._safe_invoke_action("assess_medical_toxicity", text=user_message)

            is_valid = True

            if medical_context and not medical_context.get("valid", True):
                is_valid = False
                escalate("high")
                issues.append("Medical context validation failed")
                recommendations.append("Restrict queries to pharmaceutical research topics")

            if pharma_context and not pharma_context.get("valid", True):
                is_valid = False
                escalate("high")
                issues.append("Query falls outside pharmaceutical research scope")
                recommendations.append("Decline personal medical advice requests")

            if pii_scan and pii_scan.get("detected"):
                is_valid = False
                escalate("critical")
                issues.append("PII/PHI detected in input")
                recommendations.append("Mask or remove sensitive health identifiers")

            if jailbreak_detected:
                is_valid = False
                escalate("high")
                issues.append("Potential jailbreak attempt detected")
                recommendations.append("Reject attempts to bypass guardrails")

            if pii_scan and isinstance(pii_scan, dict) and pii_scan.get("detected") and user_message:
                mask_types = pii_scan.get("types") or [
                    detection.get("type") for detection in pii_scan.get("detections", []) if isinstance(detection, dict)
                ]
                masked_via_action = await self._safe_invoke_action(
                    "mask_medical_pii",
                    text=user_message,
                    detected_pii=mask_types or [],
                )
                _register_sanitized_candidate(masked_via_action)

                pii_fallback = self._detect_pii_phi(user_message)
                _register_sanitized_candidate(pii_fallback.get("masked_text"))

            sanitized_query = self._select_sanitized_query(user_message, sanitized_candidates)

            if isinstance(toxicity_score, (int, float)) and toxicity_score >= 0.8:
                is_valid = False
                escalate("critical")
                issues.append("High-toxicity intent detected")
                recommendations.append("Escalate request for human safety review")
            elif isinstance(toxicity_score, (int, float)) and toxicity_score >= 0.5:
                escalate("medium")
                recommendations.append("Flag content for additional monitoring")

            return {
                "is_valid": is_valid,
                "severity": severity,
                "issues": issues,
                "recommendations": recommendations,
                "nemo_rails_used": True,
                "rails_variables": flow_variables,
                "sanitized_query": sanitized_query,
                "metadata": {
                    "medical_context": medical_context,
                    "pharmaceutical_context": pharma_context,
                    "pii_detected": bool(pii_scan and pii_scan.get("detected")),
                    "toxicity_score": toxicity_score,
                    "jailbreak_detected": bool(jailbreak_detected),
                    "flow_runs": flow_runs,
                    "sanitized_query": sanitized_query,
                    "actions_wrapped": self.guardrails_actions_status.get("wrapped_actions", [])
                    if self.guardrails_actions_status
                    else [],
                    "actions_registration_status": self.guardrails_actions_status,
                },
            }

        except Exception as e:
            logger.error(f"Error running input validation rails: {e}")
            # Fallback to lightweight validation
            fallback_result = self.validate_medical_query(user_message)
            # Normalize the exception fallback return shape to match rails-powered path
            return {
                "is_valid": fallback_result.get("is_valid", True),
                "severity": fallback_result.get("severity", "low"),
                "issues": fallback_result.get("issues", []),
                "recommendations": fallback_result.get("recommendations", []),
                "nemo_rails_used": False,
                "sanitized_query": fallback_result.get("sanitized_query", user_message),
                "metadata": {
                    **fallback_result.get("metadata", {}),
                    "flow_runs": {},
                    "checks_performed": fallback_result.get("checks_performed", {}),
                    "fallback_used": True,
                    "error": str(e),
                },
            }

    async def run_output_validation_rails(self, bot_message: str, sources: list[dict] | None = None) -> dict[str, Any]:
        """Run output validation using NeMo Guardrails flows.

        Args:
            bot_message: Generated response to validate
            sources: Optional source documents used in generation

        Returns:
            Dictionary containing validation results from rails flows
        """
        # If guardrails are disabled, return a simple pass-through result
        if not self.enabled:
            return {
                "is_valid": True,
                "severity": "low",
                "issues": [],
                "recommendations": [],
                "nemo_rails_used": False,
                "validated_response": bot_message,
                "metadata": {"guardrails_enabled": False, "note": "Medical guardrails are disabled"},
            }

        try:
            if not self.nemo_rails:
                logger.info("NeMo Guardrails not available, using fallback validation")
                fallback_result = self.validate_medical_response(bot_message, sources or [])
                # Normalize the fallback return shape to match rails-powered path
                return {
                    "is_valid": fallback_result.get("is_valid", True),
                    "severity": fallback_result.get("severity", "low"),
                    "issues": fallback_result.get("issues", []),
                    "recommendations": fallback_result.get("recommendations", []),
                    "nemo_rails_used": False,
                    "validated_response": bot_message,
                    "metadata": {
                        **fallback_result.get("metadata", {}),
                        "flow_runs": {},
                        "checks_performed": fallback_result.get("checks_performed", {}),
                        "fallback_used": True,
                    },
                }

            original_message = bot_message or ""
            initial_disclaimer_present = self._contains_medical_disclaimer(original_message)

            working_message = original_message
            flow_inputs = {"bot_message": working_message, "sources": sources or []}
            flow_runs: dict[str, Any] = {}
            flow_context_snapshots: dict[str, Any] = {}

            def refresh_flow_state(flow_key: str, flow_result: dict[str, Any], current: str) -> str:
                context_snapshot = self._extract_flow_context(flow_result)
                if context_snapshot:
                    flow_context_snapshots[flow_key] = context_snapshot
                    updated_message = context_snapshot.get("bot_message")
                    if isinstance(updated_message, str) and updated_message.strip():
                        return updated_message
                return current

            rails_context = await self._execute_flow("medical disclaimer enforcement", **flow_inputs)
            flow_runs["medical_disclaimer_enforcement"] = rails_context
            working_message = refresh_flow_state("medical_disclaimer_enforcement", rails_context, working_message)

            severity = "low"
            issues: list[str] = []
            recommendations: list[str] = []

            def escalate(new_severity: str) -> None:
                nonlocal severity
                if self._severity_rank(new_severity) > self._severity_rank(severity):
                    severity = new_severity

            hallucination_check = None
            hallucination_flow = {}
            if sources:
                hallucination_flow = await self._execute_flow(
                    "hallucination detection medical", bot_message=working_message, sources=sources
                )
                flow_runs["hallucination_detection"] = hallucination_flow
                working_message = refresh_flow_state("hallucination_detection", hallucination_flow, working_message)
                hallucination_check = await self._safe_invoke_action(
                    "medical_hallucination_check", response=working_message, sources=sources
                )
                if hallucination_check and hallucination_check.get("detected"):
                    severity_level = hallucination_check.get("severity", "medium")
                    escalate(severity_level if severity_level in {"low", "medium", "high", "critical"} else "high")
                    if severity_level == "high":
                        issues.append("High-severity hallucinations detected in response")
                    else:
                        issues.append("Potential hallucinations detected in response")
                        recommendations.append("Review highlighted claims before delivery")

            fact_check = None
            fact_check_flow = {}
            if sources:
                fact_check_flow = await self._execute_flow(
                    "fact check against pubmed", bot_message=working_message, sources=sources
                )
                flow_runs["fact_check_against_pubmed"] = fact_check_flow
                working_message = refresh_flow_state("fact_check_against_pubmed", fact_check_flow, working_message)
                fact_check = await self._safe_invoke_action(
                    "validate_against_pubmed_sources", claims=working_message, sources=sources
                )
                support_ratio = fact_check.get("support_ratio") if fact_check else None
                if isinstance(support_ratio, (int, float)) and support_ratio < 0.5:
                    issues.append("Less than half of claims are supported by provided sources")
                    recommendations.append("Augment response with better-aligned literature")
                    escalate("medium")

            compliance_flow = await self._execute_flow(
                "regulatory compliance check", bot_message=working_message, sources=sources or []
            )
            flow_runs["regulatory_compliance_check"] = compliance_flow
            working_message = refresh_flow_state("regulatory_compliance_check", compliance_flow, working_message)

            compliance = await self._safe_invoke_action("assess_regulatory_compliance", response=working_message)
            if compliance and not compliance.get("compliant", True):
                issues.extend(compliance.get("violations", []))
                recommendations.append("Rephrase response to meet regulatory guidance")
                escalate("high")

            filtering_flow = await self._execute_flow(
                "sensitive information filtering", bot_message=working_message, sources=sources or []
            )
            flow_runs["sensitive_information_filtering"] = filtering_flow
            working_message = refresh_flow_state("sensitive_information_filtering", filtering_flow, working_message)

            working_message = (
                await self._safe_invoke_action("filter_sensitive_medical_info", response=working_message)
                or working_message
            )

            citation_block = ""
            if sources:
                citation_block = await self._safe_invoke_action("format_source_citations", sources=sources)
                if citation_block:
                    working_message = f"{working_message}\n\n📚 **Sources:**\n{citation_block}"
                citation_validation = await self._safe_invoke_action(
                    "validate_citations", response=working_message, sources=sources
                )
                if citation_validation and citation_validation.get("invalid_citations"):
                    escalate("medium")
                    issues.append("Response cites references not present in source set")
                    recommendations.append("Align citations with retrieved literature")

            final_flow = await self._execute_flow(
                "final medical safety validation", bot_message=working_message, sources=sources or []
            )
            flow_runs["final_medical_safety_validation"] = final_flow
            working_message = refresh_flow_state("final_medical_safety_validation", final_flow, working_message)

            safety_assessment = await self._safe_invoke_action("comprehensive_safety_check", response=working_message)
            if safety_assessment and not safety_assessment.get("safe", True):
                escalate("high")
                issues.append("Final safety check failed for response")
                recommendations.append("Rework response to eliminate unsafe language")
            elif safety_assessment and safety_assessment.get("warnings"):
                recommendations.extend(safety_assessment["warnings"])

            evidence_levels = await self._safe_invoke_action("assess_evidence_levels", sources=sources or [])

            final_disclaimer_present = self._contains_medical_disclaimer(working_message)
            disclaimer_added = final_disclaimer_present and not initial_disclaimer_present

            return {
                "is_valid": severity not in ("high", "critical"),
                "severity": severity,
                "issues": issues,
                "recommendations": recommendations,
                "nemo_rails_used": True,
                "validated_response": working_message,
                "metadata": {
                    "hallucination_check": hallucination_check,
                    "fact_check": fact_check,
                    "regulatory_compliance": compliance,
                    "safety_assessment": safety_assessment,
                    "evidence_levels": evidence_levels,
                    "citations_appended": bool(citation_block.strip()),
                    "flow_runs": flow_runs,
                    "flow_context_snapshots": flow_context_snapshots,
                    "rails_variables": flow_context_snapshots.get("medical_disclaimer_enforcement", {}),
                    "disclaimer_added": disclaimer_added,
                    "actions_wrapped": self.guardrails_actions_status.get("wrapped_actions", [])
                    if self.guardrails_actions_status
                    else [],
                    "actions_registration_status": self.guardrails_actions_status,
                },
            }

        except Exception as e:
            logger.error(f"Error running output validation rails: {e}")
            # Fallback to lightweight validation
            fallback_result = self.validate_medical_response(bot_message, sources or [])
            # Normalize the exception fallback return shape to match rails-powered path
            return {
                "is_valid": fallback_result.get("is_valid", True),
                "severity": fallback_result.get("severity", "low"),
                "issues": fallback_result.get("issues", []),
                "recommendations": fallback_result.get("recommendations", []),
                "nemo_rails_used": False,
                "validated_response": bot_message,
                "metadata": {
                    **fallback_result.get("metadata", {}),
                    "flow_runs": {},
                    "checks_performed": fallback_result.get("checks_performed", {}),
                    "fallback_used": True,
                    "error": str(e),
                },
            }

    async def run_retrieval_validation_rails(self, retrieved_documents: list[dict], user_query: str) -> dict[str, Any]:
        """Run retrieval validation using NeMo Guardrails flows.

        Args:
            retrieved_documents: Documents retrieved from sources
            user_query: Original user query

        Returns:
            Dictionary containing validation results for retrieved documents
        """
        # If guardrails are disabled, return a simple pass-through result
        if not self.enabled:
            return {
                "documents_valid": True,
                "filtered_documents": retrieved_documents,
                "issues": [],
                "recommendations": [],
                "nemo_rails_used": False,
                "metadata": {
                    "guardrails_enabled": False,
                    "note": "Medical guardrails are disabled",
                    "original_count": len(retrieved_documents),
                    "filtered_count": len(retrieved_documents),
                },
            }

        try:
            if not self.nemo_rails:
                logger.info("NeMo Guardrails not available, using fallback validation")
                fallback_result = self._validate_retrieved_documents_fallback(retrieved_documents, user_query)
                fallback_result["nemo_rails_used"] = False
                return fallback_result

            # Execute retrieval validation flows sequentially
            working_documents = retrieved_documents.copy()
            flow_runs: dict[str, Any] = {}
            issues: list[str] = []
            recommendations: list[str] = []
            warnings: list[str] = []

            # Flow 1: Validate PubMed sources
            logger.info("Running validate pubmed sources flow")
            pubmed_flow = await self._execute_flow(
                "validate pubmed sources", retrieved_documents=working_documents, user_query=user_query
            )
            flow_runs["validate_pubmed_sources"] = pubmed_flow
            pubmed_context = self._extract_flow_context(pubmed_flow)
            if pubmed_context.get("retrieved_documents"):
                working_documents = pubmed_context["retrieved_documents"]
            if pubmed_context.get("warnings"):
                warnings.extend(pubmed_context["warnings"])

            # Flow 2: Medical relevance filtering
            logger.info("Running medical relevance filtering flow")
            relevance_flow = await self._execute_flow(
                "medical relevance filtering", retrieved_documents=working_documents, user_query=user_query
            )
            flow_runs["medical_relevance_filtering"] = relevance_flow
            relevance_context = self._extract_flow_context(relevance_flow)
            if relevance_context.get("retrieved_documents"):
                working_documents = relevance_context["retrieved_documents"]
            if relevance_context.get("warnings"):
                warnings.extend(relevance_context["warnings"])

            # Flow 3: Duplicate source removal
            logger.info("Running duplicate source removal flow")
            duplicate_flow = await self._execute_flow(
                "duplicate source removal", retrieved_documents=working_documents, user_query=user_query
            )
            flow_runs["duplicate_source_removal"] = duplicate_flow
            duplicate_context = self._extract_flow_context(duplicate_flow)
            if duplicate_context.get("retrieved_documents"):
                working_documents = duplicate_context["retrieved_documents"]
            if duplicate_context.get("duplicates_removed"):
                warnings.append(f"Removed {duplicate_context['duplicates_removed']} duplicate sources")

            # Flow 4: Impact factor assessment
            logger.info("Running impact factor assessment flow")
            impact_flow = await self._execute_flow(
                "impact factor assessment", retrieved_documents=working_documents, user_query=user_query
            )
            flow_runs["impact_factor_assessment"] = impact_flow
            impact_context = self._extract_flow_context(impact_flow)
            if impact_context.get("retrieved_documents"):
                working_documents = impact_context["retrieved_documents"]
            if impact_context.get("warnings"):
                warnings.extend(impact_context["warnings"])

            # Flow 5: Final source quality control
            logger.info("Running final source quality control flow")
            quality_flow = await self._execute_flow(
                "final source quality control", retrieved_documents=working_documents, user_query=user_query
            )
            flow_runs["final_source_quality_control"] = quality_flow
            quality_context = self._extract_flow_context(quality_flow)
            if quality_context.get("retrieved_documents"):
                working_documents = quality_context["retrieved_documents"]
            if quality_context.get("quality_issues"):
                quality_issues = quality_context["quality_issues"]
                if quality_issues:
                    issues.append(f"Quality control filtered out {len(quality_issues)} sources")
                    recommendations.append("Consider broadening search terms for more high-quality sources")

            # Determine if documents are valid
            documents_valid = len(working_documents) > 0
            if not documents_valid:
                issues.append("No documents passed validation")
                recommendations.append("Refine search query or expand search terms")

            # Check for significant document loss
            original_count = len(retrieved_documents)
            filtered_count = len(working_documents)
            if filtered_count < original_count * 0.5:
                warnings.append(f"Significant document filtering: {original_count} -> {filtered_count}")
                recommendations.append("Consider reviewing search criteria if too many documents were filtered")

            return {
                "documents_valid": documents_valid,
                "filtered_documents": working_documents,
                "issues": issues,
                "recommendations": recommendations,
                "warnings": warnings,
                "nemo_rails_used": True,
                "metadata": {
                    "guardrails_enabled": True,
                    "original_count": original_count,
                    "filtered_count": filtered_count,
                    "documents_removed": original_count - filtered_count,
                    "flow_runs": flow_runs,
                    "actions_wrapped": self.guardrails_actions_status.get("wrapped_actions", [])
                    if self.guardrails_actions_status
                    else [],
                    "actions_registration_status": self.guardrails_actions_status,
                },
            }

        except Exception as e:
            logger.error(f"Error running retrieval validation rails: {e}")
            fallback_result = self._validate_retrieved_documents_fallback(retrieved_documents, user_query)
            fallback_result["nemo_rails_used"] = False
            fallback_result["metadata"]["error"] = str(e)
            fallback_result["metadata"]["fallback_used"] = True
            return fallback_result

    async def _execute_flow(self, flow_name: str, **inputs: Any) -> dict[str, Any]:
        """Execute a NeMo Guardrails flow if available, returning context variables."""
        if not self.nemo_rails:
            return {}

        try:
            if hasattr(self.nemo_rails, "run_flow_async"):
                return await self.nemo_rails.run_flow_async(flow_name, **inputs)

            if hasattr(self.nemo_rails, "run_flow"):
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: self.nemo_rails.run_flow(flow_name, **inputs))

            # Fallback: attempt to use generate API with explicit flow routing if supported
            if hasattr(self.nemo_rails, "generate_async") and "user_message" in inputs:
                messages = [{"role": "user", "content": inputs["user_message"]}]
                return await self.nemo_rails.generate_async(messages=messages, flow=flow_name)

        except Exception as exc:
            logger.debug(f"Flow execution for '{flow_name}' failed: {exc}")

        return {}

    def _extract_flow_context(self, flow_result: Any) -> dict[str, Any]:
        """Extract variables from a flow execution result regardless of structure."""
        context_data: dict[str, Any] = {}

        if isinstance(flow_result, dict):
            for key in ("context", "contexts", "variables", "rails_context", "state"):
                value = flow_result.get(key)
                if isinstance(value, dict):
                    context_data.update(value)

            # Some responses may embed variables under "output" or "response"
            for key in ("output", "response"):
                value = flow_result.get(key)
                if isinstance(value, dict):
                    for nested_key in ("context", "variables"):
                        nested = value.get(nested_key)
                        if isinstance(nested, dict):
                            context_data.update(nested)

        return context_data

    @staticmethod
    def _select_sanitized_query(original: str | None, candidates: list[Any]) -> str:
        """Return the first non-empty sanitized candidate, preferring changes over original text."""
        baseline = original or ""

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip() and candidate != baseline:
                return candidate

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate

        return baseline

    @staticmethod
    def _contains_medical_disclaimer(text: str | None) -> bool:
        """Return True when the supplied text already includes a medical disclaimer marker."""
        try:
            # Use standardized disclaimer detection from guardrails.actions
            from guardrails.actions import contains_medical_disclaimer

            return contains_medical_disclaimer(text)
        except ImportError:
            # Fallback to simple detection
            if not text:
                return False
            return "medical disclaimer" in text.lower()

    async def _safe_invoke_action(self, action_name: str, **kwargs: Any) -> Any:
        """Invoke a registered action, tolerating missing integrations."""
        try:
            if self.nemo_rails:
                app = getattr(self.nemo_rails, "app", None)
                if app:
                    for candidate in ("execute_action_async", "execute_action", "call_action_async", "call_action"):
                        if hasattr(app, candidate):
                            executor = getattr(app, candidate)
                            try:
                                if inspect.iscoroutinefunction(executor):
                                    return await executor(action_name, **kwargs)
                                return executor(action_name, **kwargs)
                            except TypeError:
                                # Some executors expect kwargs bundled in a dict
                                if inspect.iscoroutinefunction(executor):
                                    return await executor(action_name, kwargs)
                                return executor(action_name, kwargs)
                            except Exception as exc:
                                logger.debug(f"Action executor '{candidate}' failed for {action_name}: {exc}")

            if self.actions and hasattr(self.actions, action_name):
                action = getattr(self.actions, action_name)
                if inspect.iscoroutinefunction(action):
                    return await action(**kwargs)
                return action(**kwargs)

            try:
                from guardrails import actions as guardrail_actions

                if hasattr(guardrail_actions, action_name):
                    action = getattr(guardrail_actions, action_name)
                    if inspect.iscoroutinefunction(action):
                        return await action(**kwargs)
                    return action(**kwargs)
            except ImportError:
                logger.debug("Guardrails actions module unavailable for action invocation")

        except Exception as exc:
            logger.debug(f"Action invocation failed for '{action_name}': {exc}")

        return None

    def _validate_retrieved_documents_fallback(
        self, retrieved_documents: list[dict], user_query: str
    ) -> dict[str, Any]:
        """Fallback validation for retrieved documents."""
        try:
            validation_result = {
                "documents_valid": True,
                "filtered_documents": retrieved_documents,
                "issues": [],
                "recommendations": [],
                "metadata": {
                    "original_count": len(retrieved_documents),
                    "filtered_count": len(retrieved_documents),
                    "duplicates_removed": 0,
                },
            }

            # Basic validation - check for PMID and medical relevance
            valid_documents = []
            for doc in retrieved_documents:
                metadata = doc.get("metadata", {})
                pmid = metadata.get("pmid")
                content = doc.get("page_content", "").lower()

                # Check for medical relevance
                medical_terms = ["drug", "medication", "pharmaceutical", "clinical", "medical"]
                is_medical = any(term in content for term in medical_terms)

                if pmid or is_medical:
                    valid_documents.append(doc)
                else:
                    validation_result["issues"].append(
                        f"Document lacks medical relevance: {metadata.get('title', 'Unknown')}"
                    )

            validation_result["filtered_documents"] = valid_documents
            validation_result["metadata"]["filtered_count"] = len(valid_documents)

            if len(valid_documents) < len(retrieved_documents):
                validation_result["recommendations"].append("Some documents were filtered due to low medical relevance")

            if len(valid_documents) == 0:
                validation_result["documents_valid"] = False
                validation_result["recommendations"].append("No valid medical documents found")

            return validation_result

        except Exception as e:
            logger.error(f"Error in fallback document validation: {e}")
            return {
                "documents_valid": False,
                "filtered_documents": [],
                "issues": [f"Validation error: {str(e)}"],
                "recommendations": ["Manual document review required"],
                "metadata": {"error": str(e)},
            }

    def validate_medical_query(self, query: str) -> dict[str, Any]:
        """Validate a medical query for safety and appropriateness.

        Args:
            query: User query to validate

        Returns:
            Dictionary containing validation results and recommendations
        """
        # If guardrails are disabled, return a simple pass-through result
        if not self.enabled:
            return {
                "is_valid": True,
                "severity": "low",
                "issues": [],
                "recommendations": [],
                "checks_performed": {},
                "metadata": {"guardrails_enabled": False, "note": "Medical guardrails are disabled"},
                "sanitized_query": query or "",
            }

        try:
            normalized_query = query or ""
            pii_result: dict[str, Any] = {
                "detected": False,
                "entities": [],
                "masked_text": normalized_query,
                "confidence": 0.0,
            }

            if normalized_query:
                pii_result = self._detect_pii_phi(normalized_query)

            sanitized_query = pii_result.get("masked_text") or normalized_query

            validation_results = {
                "is_valid": True,
                "severity": "low",
                "issues": [],
                "recommendations": [],
                "checks_performed": {
                    "pii_phi_detection": bool(normalized_query),
                    "medical_context_check": False,
                    "jailbreak_detection": False,
                    "regulatory_compliance": False,
                },
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query_length": len(normalized_query),
                    "contains_medical_terms": False,
                    "pii_detection": {
                        "detected": pii_result.get("detected", False),
                        "entities": pii_result.get("entities", []),
                        "confidence": pii_result.get("confidence"),
                    },
                },
                "sanitized_query": sanitized_query,
            }

            if not normalized_query:
                validation_results["is_valid"] = False
                validation_results["issues"].append("Empty query provided")
                validation_results["recommendations"].append("Submit a medical research question for analysis")
                return validation_results

            # Check if query contains medical context
            medical_context = self._check_medical_context(normalized_query)
            validation_results["checks_performed"]["medical_context_check"] = True
            validation_results["metadata"]["contains_medical_terms"] = medical_context["is_medical"]

            if medical_context["is_medical"]:
                if pii_result["detected"]:
                    validation_results["is_valid"] = False
                    validation_results["severity"] = "critical"
                    validation_results["issues"].append("PII/PHI detected in query")
                    validation_results["recommendations"].append("Mask or remove sensitive health identifiers")

                # Check regulatory compliance considerations
                regulatory_check = self._assess_regulatory_compliance(normalized_query)
                validation_results["checks_performed"]["regulatory_compliance"] = True

                if regulatory_check["requires_disclaimer"]:
                    validation_results["recommendations"].append("Medical disclaimer required")
                    validation_results["metadata"]["disclaimer_type"] = regulatory_check["disclaimer_type"]

                # Check for inappropriate medical advice requests
                advice_check = self._check_medical_advice_request(normalized_query)
                if advice_check["is_advice_request"]:
                    if advice_check["severity"] == "high":
                        validation_results["is_valid"] = False
                        validation_results["severity"] = "high"
                        validation_results["issues"].append("Request for specific medical advice detected")
                        validation_results["recommendations"].append("Redirect to licensed healthcare provider")
                    else:
                        validation_results["recommendations"].append("Include general information disclaimer")

            jailbreak_detected = self._detect_jailbreak_attempts(normalized_query)
            validation_results["checks_performed"]["jailbreak_detection"] = True

            if jailbreak_detected:
                validation_results["is_valid"] = False
                validation_results["severity"] = "high"
                validation_results["issues"].append("Potential jailbreak attempt detected")
                validation_results["recommendations"].append("Query appears to attempt circumventing safety guidelines")

            validation_results["metadata"]["sanitized_query"] = sanitized_query

            logger.info(f"Medical query validation completed: {validation_results['is_valid']}")
            return validation_results

        except Exception as e:
            logger.error(f"Error validating medical query: {e}")
            return {
                "is_valid": False,
                "severity": "critical",
                "issues": [f"Validation error: {str(e)}"],
                "recommendations": ["Manual review required"],
                "checks_performed": {},
                "metadata": {"error": str(e)},
                "sanitized_query": query or "",
            }

    def validate_medical_response(self, response: str, sources: list[dict]) -> dict[str, Any]:
        """Validate a medical response for safety and accuracy.

        Args:
            response: Generated response to validate
            sources: List of source documents used

        Returns:
            Dictionary containing validation results and recommendations
        """
        # If guardrails are disabled, return a simple pass-through result
        if not self.enabled:
            source_list = list(sources or [])
            return {
                "is_valid": True,
                "severity": "low",
                "issues": [],
                "recommendations": [],
                "checks_performed": {},
                "metadata": {
                    "guardrails_enabled": False,
                    "note": "Medical guardrails are disabled",
                    "response_length": len(response or ""),
                    "source_count": len(source_list),
                    "disclaimer_added": self._contains_medical_disclaimer(response),
                },
            }

        try:
            sources_list = list(sources) if sources else []
            validation_results = {
                "is_valid": True,
                "severity": "low",
                "issues": [],
                "recommendations": [],
                "checks_performed": {
                    "content_validation": False,
                    "source_validation": False,
                    "claim_validation": False,
                    "disclaimer_check": False,
                    "regulatory_compliance": False,
                },
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "response_length": len(response),
                    "source_count": len(sources_list),
                    "claim_validation": {"claims_assessed": 0, "notes": "Not evaluated"},
                    "disclaimer_added": self._contains_medical_disclaimer(response),
                },
            }

            # Validate response content
            content_validation = self._validate_response_content(response)
            validation_results["checks_performed"]["content_validation"] = True

            if not content_validation["is_appropriate"]:
                validation_results["is_valid"] = False
                validation_results["severity"] = content_validation["severity"]
                validation_results["issues"].extend(content_validation["issues"])
                validation_results["recommendations"].extend(content_validation["recommendations"])

            # Validate sources against PubMed/medical literature
            source_validation = self._validate_against_pubmed_sources(response, sources_list)
            validation_results["checks_performed"]["source_validation"] = True

            if not source_validation["sources_appropriate"]:
                validation_results["severity"] = max(validation_results["severity"], "medium", key=self._severity_rank)
                validation_results["issues"].extend(source_validation["issues"])
                validation_results["recommendations"].extend(source_validation["recommendations"])

            claim_validation_summary: dict[str, Any] = {"claims_assessed": 0, "notes": "Claim validation skipped"}
            if sources_list:
                claim_validation_summary = self._run_claim_level_checks(response, sources_list)
                validation_results["checks_performed"]["claim_validation"] = bool(
                    claim_validation_summary.get("claims_assessed")
                )
                validation_results["metadata"]["claim_validation"] = claim_validation_summary

                unsupported_claims = claim_validation_summary.get("unsupported_claims") or []
                support_ratio = claim_validation_summary.get("support_ratio")
                if unsupported_claims:
                    if self._severity_rank("medium") > self._severity_rank(validation_results["severity"]):
                        validation_results["severity"] = "medium"
                    sample = ", ".join(claim["claim"] for claim in unsupported_claims[:2])
                    validation_results["issues"].append(
                        "Claims lacking observable support in supplied sources: " + sample
                    )
                    validation_results["recommendations"].append(
                        "Reconcile unsupported claims with cited literature or remove them."
                    )
                    if isinstance(support_ratio, (int, float)) and support_ratio < 0.5 and len(unsupported_claims) > 1:
                        validation_results["is_valid"] = False
                elif validation_results["checks_performed"]["claim_validation"]:
                    validation_results["recommendations"].append(
                        "Claim-level validation completed; review summary for alignment confidence."
                    )
                else:
                    validation_results["issues"].append(
                        "No declarative claims detected for fact checking; hallucination screening is limited."
                    )
                    validation_results["recommendations"].append(
                        "Ensure the response includes verifiable statements or run full guardrail output validation."
                    )
            else:
                validation_results["metadata"]["claim_validation"] = claim_validation_summary

            # Check for appropriate disclaimers
            disclaimer_check = self._check_medical_disclaimers(response)
            validation_results["checks_performed"]["disclaimer_check"] = True

            if disclaimer_check["requires_disclaimer"] and not disclaimer_check["has_disclaimer"]:
                validation_results["recommendations"].append(f"Add {disclaimer_check['disclaimer_type']} disclaimer")

            # Check regulatory compliance
            regulatory_check = self._assess_regulatory_compliance(response)
            validation_results["checks_performed"]["regulatory_compliance"] = True

            if regulatory_check["violations"]:
                validation_results["severity"] = max(validation_results["severity"], "high", key=self._severity_rank)
                validation_results["issues"].extend(regulatory_check["violations"])
                validation_results["recommendations"].extend(regulatory_check["recommendations"])

            logger.info(f"Medical response validation completed: {validation_results['is_valid']}")
            return validation_results

        except Exception as e:
            logger.error(f"Error validating medical response: {e}")
            return {
                "is_valid": False,
                "severity": "critical",
                "issues": [f"Validation error: {str(e)}"],
                "recommendations": ["Manual review required"],
                "checks_performed": {},
                "metadata": {
                    "error": str(e),
                    "disclaimer_added": self._contains_medical_disclaimer(response),
                },
            }

    def _detect_pii_phi(self, text: str) -> dict[str, Any]:
        """Detect personally identifiable information and protected health information.

        Uses Presidio when available for advanced detection, otherwise falls back to regex patterns.
        """
        # Check if Presidio is available and should be used
        use_presidio = PRESIDIO_AVAILABLE and self.config.get("use_presidio_for_pii", False)

        if use_presidio:
            try:
                return self._detect_pii_phi_with_presidio(text)
            except Exception as e:
                logger.warning(f"Presidio PII/PHI detection failed, falling back to regex: {e}")
                # Fall back to regex-based detection

        # Regex-based detection (fallback or when Presidio not available)
        try:
            detected_entities = []
            masked_text = text
            detection_mode = self.config.get("pii_detection_mode", "balanced")

            for category, patterns in self.pii_patterns.items():
                mask_token = f"[{category.upper()}]"
                flags = self.pii_mask_flags.get(category, re.IGNORECASE)

                # Apply mode-specific filtering
                filtered_patterns = self._filter_patterns_by_mode(patterns, category, detection_mode)

                for pattern in filtered_patterns:
                    compiled = re.compile(pattern, flags)
                    matches = list(compiled.finditer(text))

                    for match in matches:
                        # Apply context validation for stricter modes
                        if detection_mode == "strict" and category in ["medical_record_numbers", "ssn"]:
                            if not self._validate_medical_context_for_entity(text, match, category):
                                continue

                        entity_type = "mrn" if category == "medical_record_numbers" else category
                        entity = {
                            "type": entity_type,
                            "text": match.group(),
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": 0.8,  # Static confidence for regex patterns
                        }
                        detected_entities.append(entity)

                    masked_text = compiled.sub(mask_token, masked_text)

            return {
                "detected": len(detected_entities) > 0,
                "entities": detected_entities,
                "masked_text": masked_text,
                "confidence": max([e["confidence"] for e in detected_entities], default=0.0),
            }

        except Exception as e:
            logger.error(f"Error detecting PII/PHI: {e}")
            return {"detected": False, "entities": [], "masked_text": text, "confidence": 0.0, "error": str(e)}

    def _detect_pii_phi_with_presidio(self, text: str) -> dict[str, Any]:
        """Detect PII/PHI using Presidio analyzer and anonymizer."""
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

            # Extract detected entities
            detected_entities = []
            for result in analyzer_results:
                entity = {
                    "type": result.entity_type,
                    "text": text[result.start : result.end],
                    "start": result.start,
                    "end": result.end,
                    "confidence": result.score,
                }
                detected_entities.append(entity)

            # Anonymize the text
            if analyzer_results:
                anonymized_result = self._anonymizer.anonymize(text=text, analyzer_results=analyzer_results)
                masked_text = anonymized_result.text
            else:
                masked_text = text

            return {
                "detected": len(detected_entities) > 0,
                "entities": detected_entities,
                "masked_text": masked_text,
                "confidence": max([e["confidence"] for e in detected_entities], default=0.0)
                if detected_entities
                else 0.0,
            }

        except Exception as e:
            logger.error(f"Error in Presidio PII/PHI detection: {e}")
            raise

    def _check_medical_context(self, query: str) -> dict[str, Any]:
        """Check if query is in medical context."""
        try:
            medical_indicators = []
            query_lower = query.lower()

            for category, patterns in self.medical_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, query_lower)
                    if matches:
                        medical_indicators.extend([(category, match) for match in matches])

            is_medical = len(medical_indicators) > 0

            return {
                "is_medical": is_medical,
                "indicators": medical_indicators,
                "confidence": min(len(medical_indicators) * 0.3, 1.0),
                "categories": list({indicator[0] for indicator in medical_indicators}),
            }

        except Exception as e:
            logger.error(f"Error checking medical context: {e}")
            return {"is_medical": False, "indicators": [], "confidence": 0.0, "error": str(e)}

    def _detect_jailbreak_attempts(self, query: str) -> bool:
        """Detect potential jailbreak attempts."""
        try:
            query_lower = query.lower()

            for pattern in self.jailbreak_patterns:
                if re.search(pattern, query_lower):
                    logger.warning(f"Potential jailbreak attempt detected: {pattern}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error detecting jailbreak attempts: {e}")
            return False

    def _validate_against_pubmed_sources(self, claims: str, sources: list[dict]) -> dict[str, Any]:
        """Validate claims against PubMed sources."""
        try:
            validation_result = {"sources_appropriate": True, "issues": [], "recommendations": []}

            # Check if sources are from reputable medical databases
            reputable_sources = ["pubmed", "medline", "cochrane", "nejm", "jama", "lancet"]

            for source in sources:
                source_info = source.get("metadata", {})
                journal = source_info.get("journal", "").lower()
                pmid = source_info.get("pmid")

                if not pmid and not any(repo in journal for repo in reputable_sources):
                    validation_result["sources_appropriate"] = False
                    validation_result["issues"].append(f"Non-medical source detected: {journal}")
                    validation_result["recommendations"].append("Use peer-reviewed medical literature")

                # Check source recency (prefer sources within 10 years for most medical topics)
                year = source_info.get("year")
                if year and isinstance(year, (int, str)):
                    try:
                        year_int = int(year)
                        current_year = datetime.now().year
                        if current_year - year_int > 10:
                            validation_result["recommendations"].append(
                                f"Consider more recent sources (source from {year})"
                            )
                    except ValueError:
                        pass

            return validation_result

        except Exception as e:
            logger.error(f"Error validating sources: {e}")
            return {
                "sources_appropriate": False,
                "issues": [f"Source validation error: {str(e)}"],
                "recommendations": ["Manual source review required"],
            }

    def _run_claim_level_checks(self, response: str, sources: list[dict]) -> dict[str, Any]:
        """Perform lightweight claim extraction and check alignment with supplied sources."""
        claims = self._extract_candidate_claims(response)
        summary: dict[str, Any] = {
            "claims_assessed": len(claims),
            "support_ratio": None,
            "unsupported_claims": [],
            "claim_support": [],
        }

        if not claims:
            summary["notes"] = "No declarative claims detected in response body."
            return summary

        if not sources:
            summary["notes"] = "Claim validation skipped because no sources were supplied."
            return summary

        support_assessment = self._assess_claim_support(claims, sources)
        claim_support = support_assessment.get("claim_support", [])
        support_ratio = support_assessment.get("support_ratio")

        summary["claim_support"] = claim_support
        summary["support_ratio"] = round(float(support_ratio), 2) if support_ratio is not None else None
        summary["unsupported_claims"] = [entry for entry in claim_support if not entry.get("supported")]

        return summary

    def _extract_candidate_claims(self, response: str) -> list[str]:
        """Extract declarative sentences that are suitable for fact checking."""
        candidates: list[str] = []
        if not response:
            return candidates

        sentences = re.split(r"(?<=[.!?])\s+", response)
        exclusion_markers = ("disclaimer", "⚠️", "medical guidance")

        for sentence in sentences:
            cleaned = sentence.strip()
            if len(cleaned) < 25:
                continue
            lowered = cleaned.lower()
            if any(marker in lowered for marker in exclusion_markers):
                continue
            if cleaned.startswith("**") or cleaned.startswith("☑"):
                continue
            candidates.append(cleaned)

        return candidates

    def _assess_claim_support(self, claims: list[str], sources: list[dict]) -> dict[str, Any]:
        """Approximate claim support by matching key terms against provided sources."""
        combined_segments: list[str] = []
        for source in sources:
            combined_segments.append(str(source.get("page_content") or source.get("content", "")))
            metadata = source.get("metadata", {})
            if isinstance(metadata, dict):
                for field in ("title", "abstract", "summary", "journal"):
                    value = metadata.get(field)
                    if value:
                        combined_segments.append(str(value))

        combined_corpus = " ".join(combined_segments).lower()
        if not combined_corpus.strip():
            return {
                "claim_support": [
                    {"claim": claim, "matched_terms": [], "coverage": 0.0, "supported": False} for claim in claims
                ],
                "support_ratio": 0.0,
            }

        token_pattern = r"[a-zA-Z][a-zA-Z\-]{3,}"
        source_terms = set(re.findall(token_pattern, combined_corpus))
        stopwords: set[str] = {
            "the",
            "that",
            "this",
            "with",
            "from",
            "have",
            "patients",
            "study",
            "shows",
            "which",
            "their",
            "there",
            "about",
            "into",
            "among",
            "after",
            "before",
            "were",
            "those",
            "they",
            "these",
            "such",
            "based",
            "using",
            "during",
            "however",
            "reported",
        }

        claim_support_entries: list[dict[str, Any]] = []
        supported_count = 0

        for claim in claims:
            tokens = [token for token in re.findall(token_pattern, claim.lower()) if token not in stopwords]
            unique_tokens = set(tokens)
            matched_terms = [token for token in unique_tokens if token in source_terms]
            coverage = len(matched_terms) / max(1, len(unique_tokens))
            supported = coverage >= 0.3 or len(matched_terms) >= 2

            if supported:
                supported_count += 1

            claim_support_entries.append(
                {
                    "claim": claim,
                    "matched_terms": sorted(matched_terms)[:8],
                    "coverage": round(coverage, 3),
                    "supported": supported,
                }
            )

        support_ratio = supported_count / max(1, len(claims))

        return {"claim_support": claim_support_entries, "support_ratio": support_ratio}

    def _generate_medical_disclaimer(self, response_type: str) -> str:
        """Generate appropriate medical disclaimer."""
        disclaimers = {
            "general": "This information is for educational purposes only and is not intended as medical advice. Consult with a healthcare professional for personalized medical guidance.",
            "drug_information": "This drug information is for educational purposes only. Always consult your healthcare provider before starting, stopping, or changing any medication.",
            "dosage": "Dosage information provided is for reference only. Never adjust medication dosages without consulting your healthcare provider.",
            "interactions": "Drug interaction information is for educational purposes. Always inform your healthcare provider about all medications and supplements you are taking.",
            "drug_interactions": "Drug interaction information is for educational purposes. Always inform your healthcare provider about all medications and supplements you are taking.",
            "research": "This information is based on current research and may not reflect the most recent findings. Consult current medical literature and healthcare professionals for the latest guidance.",
        }

        return disclaimers.get(response_type, disclaimers["general"])

    def _assess_regulatory_compliance(self, response: str) -> dict[str, Any]:
        """Assess regulatory compliance requirements."""
        try:
            compliance_check = {
                "requires_disclaimer": False,
                "disclaimer_type": "general",
                "violations": [],
                "recommendations": [],
            }

            response_lower = response.lower()

            # Check for FDA warning requirements
            for keyword in self.regulatory_keywords["fda_warnings"]:
                if keyword in response_lower:
                    compliance_check["requires_disclaimer"] = True
                    compliance_check["disclaimer_type"] = "fda_warning"
                    compliance_check["recommendations"].append("Include FDA black box warning information")

            # Check for prescription medication mentions
            for keyword in self.regulatory_keywords["prescription_only"]:
                if keyword in response_lower:
                    compliance_check["requires_disclaimer"] = True
                    compliance_check["disclaimer_type"] = "prescription_only"

            # Check for investigational drug mentions
            for keyword in self.regulatory_keywords["clinical_trials"]:
                if keyword in response_lower:
                    compliance_check["requires_disclaimer"] = True
                    compliance_check["disclaimer_type"] = "investigational"
                    compliance_check["recommendations"].append("Clarify investigational status and FDA approval")

            return compliance_check

        except Exception as e:
            logger.error(f"Error assessing regulatory compliance: {e}")
            return {
                "requires_disclaimer": True,
                "disclaimer_type": "general",
                "violations": [f"Compliance check error: {str(e)}"],
                "recommendations": ["Manual compliance review required"],
            }

    def _check_medical_advice_request(self, query: str) -> dict[str, Any]:
        """Check if query is requesting specific medical advice."""
        advice_patterns = [
            r"what should i (?:take|do|use)",
            r"how much (?:should i take|do i need)",
            r"is it safe (?:for me|to take)",
            r"can i (?:take|stop|increase|decrease)",
            r"should i (?:see a doctor|go to hospital|call 911)",
            r"what (?:medication|treatment|dose) (?:should|do) i",
            r"diagnose (?:me|my)",
        ]

        query_lower = query.lower()
        is_advice_request = any(re.search(pattern, query_lower) for pattern in advice_patterns)

        # Determine severity
        high_risk_patterns = [
            r"diagnose",
            r"emergency",
            r"urgent",
            r"should i see",
            r"call 911",
            r"go to hospital",
            r"chest pain",
            r"difficulty breathing",
        ]

        severity = "high" if any(re.search(pattern, query_lower) for pattern in high_risk_patterns) else "medium"

        return {"is_advice_request": is_advice_request, "severity": severity if is_advice_request else "low"}

    def _validate_response_content(self, response: str) -> dict[str, Any]:
        """Validate response content for appropriateness."""
        validation = {"is_appropriate": True, "severity": "low", "issues": [], "recommendations": []}

        # Check for inappropriate definitive medical statements
        definitive_patterns = [
            r"you have",
            r"you are diagnosed with",
            r"you definitely",
            r"you must",
            r"you should definitely",
            r"this will cure",
        ]

        response_lower = response.lower()
        for pattern in definitive_patterns:
            if re.search(pattern, response_lower):
                validation["is_appropriate"] = False
                validation["severity"] = "high"
                validation["issues"].append("Contains definitive medical statements")
                validation["recommendations"].append("Use conditional language and recommend professional consultation")

        return validation

    def _check_medical_disclaimers(self, response: str) -> dict[str, Any]:
        """Check if response contains appropriate medical disclaimers."""
        disclaimer_indicators = [
            "educational purposes",
            "not medical advice",
            "consult healthcare",
            "professional medical",
            "disclaimer",
            "for informational",
        ]

        has_disclaimer = any(indicator in response.lower() for indicator in disclaimer_indicators)

        # Determine if disclaimer is required
        medical_content_indicators = [
            "dosage",
            "medication",
            "treatment",
            "therapy",
            "diagnosis",
            "side effects",
            "interactions",
            "contraindications",
        ]

        requires_disclaimer = any(indicator in response.lower() for indicator in medical_content_indicators)

        return {
            "has_disclaimer": has_disclaimer,
            "requires_disclaimer": requires_disclaimer,
            "disclaimer_type": "medical_advice" if requires_disclaimer else "general",
        }

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "pii_detection_enabled": True,
            "jailbreak_detection_enabled": True,
            "medical_context_validation": True,
            "regulatory_compliance_checks": True,
            "use_presidio_for_pii": PRESIDIO_AVAILABLE,  # Use Presidio when available
            "pii_detection_mode": "balanced",  # "strict", "balanced", "relaxed"
            "disclaimer_requirements": {
                "always_include": False,
                "conditional_triggers": ["medication", "dosage", "treatment"],
            },
            "logging": {"audit_enabled": True, "log_level": "INFO"},
        }

    def _severity_rank(self, severity: str) -> int:
        """Rank severity levels for comparison."""
        ranks = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        return ranks.get(severity, 0)

    def _select_sanitized_query(self, original_query: str, candidates: list[str]) -> str:
        """Select the best sanitized query and strip bracketed safety tags.

        Args:
            original_query: The original user query
            candidates: List of sanitized query candidates

        Returns:
            Best sanitized query with safety context tags stripped
        """
        try:
            # If we have good candidates, use the longest non-empty one (most likely to be properly sanitized)
            valid_candidates = [c for c in candidates if c and c.strip()]
            if valid_candidates:
                best_candidate = max(valid_candidates, key=len)
            else:
                best_candidate = original_query or ""

            # Strip bracketed safety context tags that may have been appended
            # This ensures clean output even if the rails modification wasn't applied
            safety_tag_patterns = [
                r"\[MEDICAL_SAFETY_CONTEXT:[^\]]*\]",
                r"\[SAFETY_FLAG:[^\]]*\]",
                r"\[SAFETY_CONTEXT:[^\]]*\]",
                r"\[PHARMA_CONTEXT:[^\]]*\]",
            ]

            cleaned_query = best_candidate
            for pattern in safety_tag_patterns:
                cleaned_query = re.sub(pattern, "", cleaned_query, flags=re.IGNORECASE)

            # Clean up any extra whitespace
            cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()

            return cleaned_query

        except Exception as e:
            logger.warning(f"Error selecting sanitized query: {e}")
            # Fallback: return original query with basic tag stripping
            cleaned = re.sub(r"\[[A-Z_]+:[^\]]*\]", "", original_query or "", flags=re.IGNORECASE)
            return re.sub(r"\s+", " ", cleaned).strip()

    def _filter_patterns_by_mode(self, patterns: list[str], category: str, mode: str) -> list[str]:
        """Filter patterns based on detection mode to control sensitivity."""
        if mode == "relaxed":
            # In relaxed mode, only use the most specific patterns
            if category == "medical_record_numbers":
                return [p for p in patterns if "MRN" in p or "medical record" in p]
            elif category == "ssn":
                return [p for p in patterns if "SSN" in p or "social security" in p]
        elif mode == "strict":
            # In strict mode, use all patterns including context-anchored ones
            return patterns
        else:  # balanced mode
            # In balanced mode, prefer context-anchored patterns but include general ones
            return patterns

    def _validate_medical_context_for_entity(self, text: str, match: re.Match, category: str) -> bool:
        """Validate that detected entity appears in appropriate medical context."""
        # Extract context window around the match
        start_pos = max(0, match.start() - 50)
        end_pos = min(len(text), match.end() + 50)
        context = text[start_pos:end_pos].lower()

        if category == "medical_record_numbers":
            medical_keywords = ["patient", "medical", "record", "mrn", "hospital", "clinic", "admission"]
            return any(keyword in context for keyword in medical_keywords)
        elif category == "ssn":
            medical_keywords = ["patient", "insurance", "billing", "medical", "healthcare", "ssn", "social security"]
            return any(keyword in context for keyword in medical_keywords)

        # For other categories, less strict context validation
        return True


__all__ = ["MedicalGuardrails", "ValidationResult", "PIIDetectionResult"]
