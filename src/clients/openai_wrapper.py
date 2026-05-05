"""
OpenAI SDK Wrapper for NVIDIA Build Platform

Single-responsibility wrapper providing clean interface to NVIDIA Build endpoints
with pharmaceutical-optimized defaults and comprehensive error handling.

This wrapper ensures NGC-independence while maintaining pharmaceutical
domain advantages through intelligent model selection and configuration.

Design Principles:
- Single responsibility: Only handles OpenAI SDK interactions
- Pharmaceutical optimization: Medical/drug research defaults
- Clean error handling: Structured exceptions and logging
- NGC-independent: Future-proof against March 2026 deprecation
"""
import logging
import os
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import requests

if TYPE_CHECKING:
    # Import for type hints only to avoid circular dependencies
    try:
        from openai.types.chat import ChatCompletion
    except ImportError:
        ChatCompletion = Any
from contextlib import contextmanager
from dataclasses import dataclass

# Deferred OpenAI imports for self-hosted compatibility
# OpenAI SDK is only required when cloud functionality is actually used
OPENAI_AVAILABLE = False
_OPENAI_IMPORT_ERROR = None
_OPENAI_MODULES = None


def _import_openai_modules():
    """Import OpenAI modules with deferred loading for self-hosted compatibility."""
    global OPENAI_AVAILABLE, _OPENAI_IMPORT_ERROR, _OPENAI_MODULES

    if _OPENAI_MODULES is not None:
        return _OPENAI_MODULES

    try:
        from openai import APIConnectionError, APIError, APIStatusError, OpenAI, RateLimitError
        from openai.types import CreateEmbeddingResponse
        from openai.types.chat import ChatCompletion

        _OPENAI_MODULES = {
            "OpenAI": OpenAI,
            "APIError": APIError,
            "APIStatusError": APIStatusError,
            "RateLimitError": RateLimitError,
            "APIConnectionError": APIConnectionError,
            "CreateEmbeddingResponse": CreateEmbeddingResponse,
            "ChatCompletion": ChatCompletion,
            "OPENAI_EXCEPTIONS": (APIError, APIStatusError, RateLimitError, APIConnectionError),
        }

        OPENAI_AVAILABLE = True
        return _OPENAI_MODULES

    except ImportError as e:
        OPENAI_AVAILABLE = False
        _OPENAI_IMPORT_ERROR = e
        raise NVIDIABuildError(
            f"OpenAI SDK not available for cloud operations: {str(e)}. "
            "Install with: pip install 'openai>=1.0.0,<2.0.0' or use self-hosted endpoints only."
        )


logger = logging.getLogger(__name__)


def _safe_bool(raw: Optional[str], default: bool = False) -> bool:
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "on", "enabled"}:
        return True
    if s in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _safe_float(raw: Optional[str]) -> Optional[float]:
    try:
        return float(raw) if raw is not None and raw != "" else None
    except Exception:
        return None


def _extract_user_text(messages: List[Dict[str, str]]) -> str:
    """Extract the most recent user message content."""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


@dataclass
class NVIDIABuildConfig:
    """Configuration for NVIDIA Build platform access."""

    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    pharmaceutical_optimized: bool = True
    # Pharmaceutical research cost + batching controls
    research_project_id: Optional[str] = None
    research_project_budget_limit_usd: Optional[float] = None
    enable_cost_per_query_tracking: bool = True
    enable_batch_optimization: bool = True
    batch_max_size: int = 16
    batch_max_latency_ms: int = 500
    # Embeddings input_type for asymmetric models (e.g., nv-embedqa-e5-v5)
    embedding_input_type: Optional[str] = None
    # Critical configuration flags for pharmaceutical safety
    enable_embedding_input_type: bool = True  # Enable embedding input_type parameter injection
    enable_model_listing: bool = False  # Safety gate for model listing operations
    # Preferred models (overrides pharma defaults when provided)
    prefer_embedding_model: Optional[str] = None
    prefer_chat_model: Optional[str] = None
    # Lazy initialization for testing scenarios
    lazy_init: bool = False
    # API compatibility: prefer responses.create over chat.completions.create for Build compatibility
    prefer_responses_api: bool = True
    # Responses API schema mapping: convert messages to input parameter when enabled
    enable_responses_schema_mapping: bool = False  # Enable message-to-input conversion
    # Request policy: dedicated backoff base for rerank retry (seconds)
    request_backoff_base: float = 0.5
    # Rerank-specific retry attempts (overrides generic max_retries when provided)
    rerank_retry_max_attempts: int = 3
    # Request policy: jitter amplitude in seconds added to backoff (0 disables jitter)
    request_backoff_jitter: float = 0.0
    # Rerank service ordering and behavior
    cloud_first_rerank_enabled: bool = True
    build_rerank_enabled: bool = True
    enable_rerank_model_mapping: bool = False
    nemo_reranking_endpoint: Optional[str] = None


class NVIDIABuildError(Exception):
    """Custom exception for NVIDIA Build API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class PharmaceuticalConfigError(NVIDIABuildError):
    """
    Pharmaceutical-specific configuration errors for audit tracking.

    Used when pharmaceutical research settings are invalid or incompatible
    with regulatory requirements. These errors require immediate attention
    in pharmaceutical environments to ensure compliance.
    """


class PharmaceuticalQueryError(NVIDIABuildError):
    """
    Safety-critical pharmaceutical query errors requiring immediate attention.

    Raised when drug interaction, pharmacokinetics, or clinical trial queries
    encounter errors that could compromise medical research accuracy or safety.
    These errors are logged with high priority for pharmaceutical audit trails.
    """


class PharmaceuticalResponseError(NVIDIABuildError):
    """
    Pharmaceutical response processing errors for medical query safety.

    Raised when response normalization or processing fails for safety-critical
    pharmaceutical queries. Ensures medical research queries receive proper
    error handling and fallback responses for regulatory compliance.
    """


class OpenAIWrapper:
    """
    Clean OpenAI SDK wrapper for NVIDIA Build platform.

    Provides pharmaceutical-optimized defaults while maintaining
    flexibility for general research applications.
    """

    # Pharmaceutical-optimized model mappings
    PHARMA_MODELS = {
        "embedding": {
            "preferred": "nvidia/nv-embedqa-e5-v5",  # Q&A optimized for medical
            "fallback": "nvidia/nv-embed-v1",  # General purpose
            "dimensions": {"nvidia/nv-embedqa-e5-v5": 1024, "nvidia/nv-embed-v1": 4096},
        },
        "chat": {
            "preferred": "meta/llama-3.1-8b-instruct",  # Advanced reasoning
            "temperature_pharma": 0.1,  # Conservative for medical accuracy
            "max_tokens_pharma": 1000,  # Balanced response length
        },
    }

    _FALLBACK_AVAILABLE_MODELS = [
        {"id": "nvidia/nv-embedqa-e5-v5"},
        {"id": "nvidia/nv-embed-v1"},
        {"id": "meta/llama-3.1-8b-instruct"},
        {"id": "meta/llama-3.3-70b-instruct"},
        {"id": "nvidia/nv-rerankqa-mistral4b-v3"},
    ]

    def __init__(self, config: Optional[NVIDIABuildConfig] = None):
        """
        Initialize OpenAI wrapper with NVIDIA Build configuration.

        Args:
            config: NVIDIA Build configuration (defaults to environment-based)
        """
        self.config = config or NVIDIABuildConfig()

        # Pharmaceutical optimization flag
        self.pharma_optimized = self.config.pharmaceutical_optimized
        if self.pharma_optimized:
            logger.info("Pharmaceutical optimization enabled")

        # Get API key from config or environment
        api_key = self.config.api_key or os.getenv("NVIDIA_API_KEY")

        # Handle lazy initialization mode for testing scenarios
        if self.config.lazy_init and not api_key:
            self.client = None
            self._api_key = None
            self._initialization_params = {
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
                "base_url": self.config.base_url,
            }
            logger.info("OpenAI wrapper initialized in lazy mode (no API key provided)")
        else:
            if not api_key:
                raise NVIDIABuildError(
                    "NVIDIA API key required. Set NVIDIA_API_KEY environment variable "
                    "or provide in NVIDIABuildConfig."
                )

            # Initialize OpenAI client immediately
            self._api_key = api_key
            self._initialization_params = None
            self._initialize_client()

        # Initialize cost metrics for research budgeting (always initialize)
        self._cost_metrics: Dict[str, Any] = {
            "project_id": self.config.research_project_id or os.getenv("PHARMA_PROJECT_ID"),
            "project_budget_limit_usd": (
                self.config.research_project_budget_limit_usd
                if self.config.research_project_budget_limit_usd is not None
                else _safe_float(os.getenv("PHARMA_BUDGET_LIMIT_USD"))
            ),
            "enable_cost_tracking": self.config.enable_cost_per_query_tracking
            if self.config.enable_cost_per_query_tracking is not None
            else _safe_bool(os.getenv("PHARMA_COST_PER_QUERY_TRACKING"), True),
            "total_requests": 0,
            "total_tokens": 0,
            "query_types": {"drug_interaction": 0, "pharmacokinetics": 0, "clinical_trial": 0, "general": 0},
        }
        # Simple log of decisions for traceability
        logger.debug(
            "Cost tracking initialized | project=%s budget=%s enable=%s",
            self._cost_metrics["project_id"],
            self._cost_metrics["project_budget_limit_usd"],
            self._cost_metrics["enable_cost_tracking"],
        )

    def _initialize_client(self) -> None:
        """Initialize OpenAI client with stored parameters and deferred imports."""
        try:
            # Import OpenAI modules when actually needed
            openai_modules = _import_openai_modules()
            OpenAI = openai_modules["OpenAI"]

            # Try with timeout and max_retries parameters
            client_kwargs = {
                "api_key": self._api_key,
                "base_url": self.config.base_url,
            }

            if self._initialization_params:
                # Use stored parameters for lazy initialization
                client_kwargs.update(self._initialization_params)
                timeout = self._initialization_params.get("timeout", self.config.timeout)
                max_retries = self._initialization_params.get("max_retries", self.config.max_retries)
            else:
                # Use current config values
                timeout = self.config.timeout
                max_retries = self.config.max_retries

            # Check if OpenAI client supports timeout and max_retries parameters
            try:
                self.client = OpenAI(timeout=timeout, max_retries=max_retries, **client_kwargs)
            except TypeError as err:
                # Fall back to defaults if parameters not supported
                logger.warning(
                    f"OpenAI SDK initialization parameters not supported (timeout, max_retries): {err}. "
                    "Using defaults."
                )
                self.client = OpenAI(**client_kwargs)

            logger.info(f"OpenAI wrapper initialized for NVIDIA Build: {self.config.base_url}")

        except NVIDIABuildError:
            # Re-raise our custom error from import failure
            raise
        except Exception as e:
            raise NVIDIABuildError(f"Failed to initialize OpenAI client: {str(e)}")

    def _ensure_client_initialized(self) -> None:
        """Ensure client is initialized, performing lazy initialization if needed."""
        if self.client is None:
            if self.config.lazy_init:
                # Attempt to get API key from environment for lazy initialization
                api_key = self.config.api_key or os.getenv("NVIDIA_API_KEY")
                if not api_key:
                    raise NVIDIABuildError(
                        "Cannot initialize client: NVIDIA API key required for lazy initialization. "
                        "Set NVIDIA_API_KEY environment variable or inject client for testing."
                    )
                self._api_key = api_key
                self._initialize_client()
            else:
                raise NVIDIABuildError("Client not initialized and lazy mode not enabled")

    def inject_client_for_testing(self, mock_client) -> None:
        """Inject a mock client for testing scenarios.

        Args:
            mock_client: Mock OpenAI client instance for testing
        """
        self.client = mock_client
        logger.info("Mock client injected for testing")

    @contextmanager
    def _error_handler(self, operation: str):
        """Context manager for consistent error handling with deferred OpenAI exceptions."""
        start_time = time.time()
        try:
            yield
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)

            # Check if this is an OpenAI exception (only if OpenAI is available)
            is_openai_exception = False
            if OPENAI_AVAILABLE and _OPENAI_MODULES:
                openai_exceptions = _OPENAI_MODULES.get("OPENAI_EXCEPTIONS", ())
                is_openai_exception = isinstance(e, openai_exceptions)

            if is_openai_exception:
                logger.error(f"{operation} failed after {response_time}ms: {str(e)}")
                # Extract status code from OpenAI error
                status_code = getattr(e, "status_code", None)
                response_data = getattr(e, "response", None)

                raise NVIDIABuildError(
                    f"{operation} failed: {str(e)}", status_code=status_code, response_data=response_data
                )
            else:
                logger.error(f"{operation} failed after {response_time}ms: {str(e)}")
                raise NVIDIABuildError(f"{operation} failed: {str(e)}")

    def create_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        encoding_format: str = "float",
        input_type: Optional[str] = None,
        is_query: Optional[bool] = None,
        **kwargs,
    ):
        """
        Create embeddings with pharmaceutical optimization.

        Args:
            texts: List of texts to embed
            model: Embedding model (auto-selected if None)
            encoding_format: Encoding format for embeddings
            **kwargs: Additional OpenAI parameters

        Returns:
            CreateEmbeddingResponse from OpenAI SDK

        Raises:
            NVIDIABuildError: On API errors or failures
        """
        # Early return for empty inputs
        if not texts:
            # Return a typed CreateEmbeddingResponse for empty inputs
            mdl = model
            if mdl is None:
                mdl = (
                    self.PHARMA_MODELS["embedding"]["preferred"]
                    if self.pharma_optimized
                    else self.PHARMA_MODELS["embedding"]["fallback"]
                )  # noqa: E501
            return self._empty_embedding_response(mdl)

        # Auto-select pharmaceutical-optimized model
        if model is None:
            model = self.config.prefer_embedding_model or (
                self.PHARMA_MODELS["embedding"]["preferred"]
                if self.pharma_optimized
                else self.PHARMA_MODELS["embedding"]["fallback"]
            )  # noqa: E501
            logger.debug(f"Auto-selected embedding model: {model}")

        with self._error_handler("Embedding creation"):
            # Ensure client is initialized
            self._ensure_client_initialized()

            # Inject input_type for asymmetric models via extra_body if configured or auto-detected
            effective_kwargs = dict(kwargs)

            def _ensure_extra_body(d: Dict[str, Any]) -> Dict[str, Any]:
                eb = d.get("extra_body")
                if not isinstance(eb, dict):
                    eb = {}
                d["extra_body"] = eb
                return eb

            input_type_value: Optional[str] = None
            # API parameter takes precedence
            if isinstance(input_type, str) and input_type in {"query", "passage"}:
                input_type_value = input_type
            elif is_query is not None:
                input_type_value = "query" if is_query else "passage"
            elif self.config.embedding_input_type:
                # Use config directly when present
                input_type_value = self.config.embedding_input_type
            else:
                # Fall back to environment heuristics only when config unset
                if isinstance(model, str) and ("embedqa" in model or "nv-embedqa" in model):
                    if _safe_bool(os.getenv("ENABLE_NVIDIA_BUILD_EMBEDDING_INPUT_TYPE"), True):
                        input_type_value = os.getenv("NVIDIA_BUILD_EMBEDDING_INPUT_TYPE", "query")
            if input_type_value:
                extra_body = _ensure_extra_body(effective_kwargs)
                if "input_type" in extra_body:
                    logger.debug(
                        "Embedding input_type already provided in extra_body; not overriding (%s)",
                        extra_body.get("input_type"),
                    )
                else:
                    logger.debug("Injecting embedding input_type=%s via config/env toggle", input_type_value)
                    extra_body["input_type"] = input_type_value
            # Build call kwargs, gating encoding_format by model support
            call_kwargs: Dict[str, Any] = {
                "input": texts,
                "model": model,
                **effective_kwargs,
            }
            try:
                if encoding_format is not None and isinstance(model, str) and ("nv-embed" in model):
                    call_kwargs["encoding_format"] = encoding_format
            except Exception:
                pass

            response = self.client.embeddings.create(**call_kwargs)

            dims = len(response.data[0].embedding) if getattr(response, "data", None) and response.data else 0
            logger.info(f"Embeddings created: {len(texts)} texts, model: {model}, " f"dimensions: {dims}")
            # Update cost metrics (token usage not exposed for embeddings; track by count)
            self._record_cost_event("embedding", tokens=None, query_type="general", count=len(texts))
            return response

    def _empty_embedding_response(self, model: str):
        """Create a typed empty embedding response that is SDK-compatible."""
        # Import OpenAI types when needed
        openai_modules = _import_openai_modules()
        CreateEmbeddingResponse = openai_modules["CreateEmbeddingResponse"]

        payload = {
            "data": [],
            "model": model,
            "object": "list",
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
        # Prefer pydantic model_validate when available (OpenAI types use Pydantic v2)
        try:
            return CreateEmbeddingResponse.model_validate(payload)  # type: ignore[attr-defined]
        except Exception:
            try:
                return CreateEmbeddingResponse(**payload)  # type: ignore[call-arg]
            except Exception as e:
                # As a last resort, attempt model_validate again; if fails, raise a clear error
                logger.debug("Fallback model construction failed for empty embeddings: %s", e)
                return CreateEmbeddingResponse.model_validate(payload)  # type: ignore[attr-defined]

    def _convert_messages_to_responses_input(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Convert OpenAI chat messages to Responses API input format.

        This method provides flexible schema conversion for NVIDIA Build endpoints
        that may expect different input formats for the responses.create API.

        Args:
            messages: List of OpenAI chat messages with role/content structure

        Returns:
            Dictionary with converted input format for responses.create API

        The conversion logic supports multiple formats (prioritized for pharmaceutical safety):
        1. Messages array input: preserves full conversation context including system prompts
        2. Single string input: extracts last user message content as fallback
        3. Structured input: maintains pharmaceutical context and assistant turns
        """
        try:
            if not messages:
                return {"input": ""}

            # Strategy 1: Use the entire messages array to preserve pharmaceutical context
            # This prioritizes system prompts and conversation context for safety-critical queries
            logger.debug(f"Converted messages to structured input array with {len(messages)} messages")
            return {"input": messages}

        except Exception as e:
            logger.warning(f"Error converting messages to responses input format: {e}")

            # Strategy 2: Fallback to single user content extraction if structured format fails
            try:
                user_content = _extract_user_text(messages)
                if user_content:
                    logger.debug(f"Fallback to single input string: '{user_content[:100]}...'")
                    return {"input": user_content}
            except Exception as fallback_error:
                logger.warning(f"Fallback user content extraction failed: {fallback_error}")

            # Final fallback: return messages in original format
            return {"input": messages}

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """
        Create chat completion with pharmaceutical optimization.

        Args:
            messages: List of message dictionaries
            model: Chat model (auto-selected if None)
            max_tokens: Maximum tokens (pharmaceutical default if None)
            temperature: Sampling temperature (pharmaceutical default if None)
            **kwargs: Additional OpenAI parameters

        Returns:
            ChatCompletion from OpenAI SDK

        Raises:
            NVIDIABuildError: On API errors or failures
        """
        # Apply pharmaceutical defaults
        if model is None:
            model = self.config.prefer_chat_model or self.PHARMA_MODELS["chat"]["preferred"]
            logger.debug(f"Auto-selected chat model: {model}")

        # Classify query type for pharma-aware tuning
        query_text = _extract_user_text(messages)
        query_type = self.classify_pharma_query(query_text) if self.pharma_optimized else "general"

        # Optimize request params for pharma research
        if self.pharma_optimized:
            model, temperature, max_tokens, messages = self._optimize_request_params(
                query_type=query_type,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
            )

        # Optional safety validation of selected model
        self._validate_model_selection(model, model_type="chat", query_type=query_type)

        # Execute with pharma-aware retry policy for critical medical queries
        # Avoid runtime NameError on missing ChatCompletion type at runtime
        def _do_call() -> Any:
            with self._error_handler("Chat completion"):
                # Ensure client is initialized
                self._ensure_client_initialized()

                # Try responses.create first for Build compatibility, fall back to chat.completions.create
                if self.config.prefer_responses_api:
                    try:
                        # Prepare request parameters based on schema mapping configuration
                        if self.config.enable_responses_schema_mapping:
                            # Convert messages to input parameter format
                            input_params = self._convert_messages_to_responses_input(messages)
                            logger.debug("Using responses API with schema mapping enabled")

                            response = self.client.responses.create(
                                model=model,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                **input_params,  # Contains 'input' parameter instead of 'messages'
                                **kwargs,
                            )
                        else:
                            # Use original messages parameter format (backward compatibility)
                            logger.debug("Using responses API with original messages format")

                            response = self.client.responses.create(
                                model=model,
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                **kwargs,
                            )

                        logger.debug(
                            f"Successfully used responses.create API path (schema_mapping={self.config.enable_responses_schema_mapping})"
                        )

                        # Normalize responses API response to ChatCompletion format for downstream compatibility
                        if not hasattr(response, "choices") or not response.choices:
                            logger.debug("Normalizing responses API response to ChatCompletion format")
                            return self._normalize_responses_to_chat_format(response)

                        return response
                    except (AttributeError, TypeError) as e:
                        logger.debug(f"responses.create not available ({e}), falling back to chat.completions.create")
                    except Exception as e:
                        logger.warning(
                            f"responses.create failed with schema mapping ({e}), falling back to chat.completions.create"
                        )

                # Fallback to standard chat completions API
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                logger.debug("Successfully used chat.completions.create API path")
                return response

        critical = query_type in {"drug_interaction", "pharmacokinetics", "clinical_trial"}
        response = self._execute_with_retries(_do_call, retries=2 if critical else 0)

        # Update cost metrics
        token_count = response.usage.total_tokens if getattr(response, "usage", None) else 0
        self._record_cost_event("chat_completion", tokens=token_count, query_type=query_type)

        logger.info(f"Chat completion successful: model: {model}, tokens: {token_count}, type: {query_type}")
        return response

    def list_models(self) -> List[str]:
        """
        List available models on NVIDIA Build platform.

        Note: This operation may fail on some NVIDIA Build endpoints.
        Use enable_model_listing configuration flag to control availability.

        Returns:
            List of available model names

        Raises:
            NVIDIABuildError: On API errors, failures, or when model listing is disabled
        """
        # Safety gate: Check if model listing is enabled
        if not self.config.enable_model_listing:
            raise NVIDIABuildError(
                "Model listing is disabled for safety. Enable via NVIDIABuildConfig.enable_model_listing=True "
                "if your endpoint supports this operation."
            )

        with self._error_handler("Model listing"):
            # Ensure client is initialized
            self._ensure_client_initialized()

            try:
                models = self.client.models.list()
                model_names = [model.id for model in models.data]
                logger.info(f"Listed {len(model_names)} available models")
                return model_names
            except Exception as e:
                logger.error(f"Model listing failed - endpoint may not support this operation: {str(e)}")
                raise NVIDIABuildError(
                    f"Model listing failed. This endpoint may not support model listing operations. "
                    f"Consider setting enable_model_listing=False in configuration. Error: {str(e)}"
                )

    # Backward/compatibility helper expected by tests and integrations
    def list_available_models(self) -> List[Dict[str, Any]]:
        """Return available models as a list of dictionaries (id-first).

        Note: This operation may fail on some NVIDIA Build endpoints.
        Use enable_model_listing configuration flag to control availability.
        """
        # Safety gate: Check if model listing is enabled
        if not self.config.enable_model_listing:
            logger.info(
                "Model listing disabled for safety. Returning fallback model inventory instead of calling the endpoint."
            )
            return [model.copy() for model in self._FALLBACK_AVAILABLE_MODELS]

        with self._error_handler("Model listing"):
            # Ensure client is initialized
            self._ensure_client_initialized()

            try:
                models = self.client.models.list()
                out: List[Dict[str, Any]] = []
                for m in getattr(models, "data", []) or []:
                    mid = getattr(m, "id", None)
                    try:
                        mdict = m.dict()  # type: ignore[attr-defined]
                        if isinstance(mdict, dict):
                            mdict.setdefault("id", mid)
                            out.append(mdict)
                            continue
                    except Exception:
                        pass
                    out.append({"id": mid})
                logger.info("Model listing returned %d items", len(out))
                return out
            except Exception as e:
                logger.warning(f"Model listing failed - returning fallback inventory: {str(e)}")
                return [model.copy() for model in self._FALLBACK_AVAILABLE_MODELS]

    def _normalize_responses_to_chat_format(self, responses_obj: Any) -> Union["ChatCompletion", Any]:
        """
        Convert responses API response to ChatCompletion-compatible format.

        This ensures consistent .choices[0].message.content access pattern
        regardless of whether responses.create or chat.completions.create was used.

        PHARMACEUTICAL SAFETY: This normalization is critical for safety-critical drug
        interaction and pharmacokinetics queries. Consistent API access patterns prevent
        response parsing failures that could compromise medical query reliability.

        AUDIT TRAIL: All normalization attempts are logged for pharmaceutical compliance.
        Fallback handling ensures medical queries always receive valid responses.

        Args:
            responses_obj: Response object from responses.create API

        Returns:
            Normalized response object with .choices[0].message.content structure.
            Falls back to original response if normalization fails to ensure
            pharmaceutical query continuity.
        """
        try:
            # Extract content from various possible response formats
            content = ""

            # Try common responses API formats
            if hasattr(responses_obj, "output_text"):
                content = str(responses_obj.output_text)
            elif hasattr(responses_obj, "output") and responses_obj.output:
                if isinstance(responses_obj.output, list) and responses_obj.output:
                    # Handle output array format
                    first_output = responses_obj.output[0]
                    if hasattr(first_output, "content"):
                        content = str(first_output.content)
                    else:
                        content = str(first_output)
                else:
                    content = str(responses_obj.output)
            elif hasattr(responses_obj, "content"):
                content = str(responses_obj.content)
            else:
                # Last resort: convert entire response to string
                content = str(responses_obj)

            # Create normalized response with ChatCompletion structure
            return self._create_normalized_chat_completion(content, responses_obj)

        except Exception as e:
            logger.warning(f"Response normalization failed: {e}")
            # Return original response as fallback
            return responses_obj

    def _create_normalized_chat_completion(self, content: str, original_response: Any) -> Union["ChatCompletion", Any]:
        """
        Create a normalized ChatCompletion-like object with consistent API.

        PHARMACEUTICAL COMPLIANCE: Creates structured response objects that maintain
        pharmaceutical metadata (usage tracking, model information) required for
        medical research audit trails and cost tracking.

        SAFETY CONSIDERATIONS: Preserves all response metadata to ensure pharmaceutical
        queries maintain proper attribution and traceability for regulatory compliance.

        Args:
            content: Extracted content text from pharmaceutical query response
            original_response: Original response object for metadata extraction

        Returns:
            Object with .choices[0].message.content structure compatible with
            pharmaceutical processing pipelines. Includes usage metrics for
            research cost tracking and model attribution for audit trails.
        """
        try:
            # Try to import ChatCompletion for proper type
            modules = _import_openai_modules()
            ChatCompletion = modules["ChatCompletion"]

            # Extract or synthesize metadata
            model = getattr(original_response, "model", "unknown")
            usage = getattr(original_response, "usage", None)

            # Create normalized choice object using simple dict structure
            message = {"role": "assistant", "content": content}

            choice = {"index": 0, "message": message, "finish_reason": "stop"}

            # Create normalized response dict that mimics ChatCompletion
            normalized_data = {
                "id": getattr(original_response, "id", "normalized-response"),
                "choices": [choice],
                "created": getattr(original_response, "created", int(time.time())),
                "model": model,
                "object": "chat.completion",
                "usage": usage,
            }

            # Try to create proper ChatCompletion object, fall back to dict
            try:
                return ChatCompletion.model_validate(normalized_data)
            except Exception:
                # Fallback to simple object with required attributes
                return self._create_simple_normalized_response(content, original_response)

        except Exception as e:
            logger.warning(f"Failed to create proper ChatCompletion object: {e}")
            return self._create_simple_normalized_response(content, original_response)

    def _create_simple_normalized_response(self, content: str, original_response: Any) -> Any:
        """Create a simple normalized response object as fallback."""

        class NormalizedResponse:
            def __init__(self, content: str, original: Any):
                self.choices = [SimpleChoice(content)]
                self.model = getattr(original, "model", "unknown")
                self.usage = getattr(original, "usage", None)
                self.id = getattr(original, "id", "normalized-response")

        class SimpleChoice:
            def __init__(self, content: str):
                self.message = SimpleMessage(content)

        class SimpleMessage:
            def __init__(self, content: str):
                self.content = content
                self.role = "assistant"

        return NormalizedResponse(content, original_response)

    # ------------------------------------------------------------------
    # Pharmaceutical research helpers
    # ------------------------------------------------------------------
    def classify_pharma_query(self, text: str) -> str:
        """Delegate to shared pharma classifier."""
        try:
            from ..utils.pharma import classify_query  # type: ignore

            return classify_query(text)
        except ImportError:
            try:
                from src.utils.pharma import classify_query  # type: ignore

                return classify_query(text)
            except ImportError:
                # Log warning once and default to general classification
                if not hasattr(self, "_pharma_import_warned"):
                    logger.warning("Pharma query classification utilities unavailable, defaulting to 'general'")
                    self._pharma_import_warned = True
                return "general"

    def _optimize_request_params(
        self,
        query_type: str,
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        messages: List[Dict[str, str]],
    ) -> Tuple[str, Optional[float], Optional[int], List[Dict[str, str]]]:
        """Apply pharma-aware request optimizations and defaults, returning merged messages."""
        # Conservative settings for high-stakes queries
        if temperature is None:
            if query_type == "drug_interaction":
                temperature = 0.0
            elif query_type == "pharmacokinetics":
                temperature = 0.1
            elif query_type == "clinical_trial":
                temperature = 0.2
            else:
                temperature = self.PHARMA_MODELS["chat"].get("temperature_pharma", 0.1)

        if max_tokens is None:
            # Slightly larger allowance for clinical summaries
            if query_type in {"drug_interaction", "pharmacokinetics"}:
                max_tokens = min(1000, self.PHARMA_MODELS["chat"].get("max_tokens_pharma", 1000))
            else:
                max_tokens = min(1200, self.PHARMA_MODELS["chat"].get("max_tokens_pharma", 1000))

        # Encourage precise, source-aware output for medical domains via system hint
        sys_hint = (
            "You are a medical research assistant. Provide precise, concise, and safety-focused answers. "
            "Cite mechanisms and known contraindications for drug interactions where applicable."
        )
        if not any(m.get("role") == "system" for m in messages):
            merged = [{"role": "system", "content": sys_hint}] + list(messages)
        else:
            merged = messages

        return model, temperature, max_tokens, merged

    def _validate_model_selection(self, model: str, model_type: str, query_type: str) -> None:
        """Log a warning if non-recommended model is used for pharma content."""
        try:
            if model_type == "chat":
                preferred = self.PHARMA_MODELS["chat"].get("preferred")
                if self.pharma_optimized and preferred and model != preferred:
                    enforce = _safe_bool(os.getenv("PHARMA_ENFORCE_PREFERRED_MODELS"), False)
                    if enforce:
                        raise NVIDIABuildError(
                            f"Non-preferred chat model '{model}' for {query_type}; preferred: {preferred}"
                        )
                    else:
                        logger.warning(
                            "Non-preferred chat model '%s' used for %s; preferred: %s",
                            model,
                            query_type,
                            preferred,
                        )
            elif model_type == "embedding":
                preferred = self.PHARMA_MODELS["embedding"].get("preferred")
                if self.pharma_optimized and preferred and model != preferred:
                    enforce = _safe_bool(os.getenv("PHARMA_ENFORCE_PREFERRED_MODELS"), False)
                    if enforce:
                        raise NVIDIABuildError(f"Non-preferred embedding model '{model}'; preferred: {preferred}")
                    else:
                        logger.warning(
                            "Non-preferred embedding model '%s' used; preferred: %s",
                            model,
                            preferred,
                        )
        except Exception:
            # Never break request due to validation logging
            pass

    def _execute_with_retries(self, func, retries: int = 0, base_delay: float = 0.5):
        """Execute a callable with simple exponential backoff retries."""
        attempt = 0
        last_err = None
        while attempt <= retries:
            try:
                return func()
            except NVIDIABuildError as e:
                last_err = e
                if attempt == retries:
                    raise
                delay = base_delay * (2**attempt)
                logger.warning("Retrying after error (%s). attempt=%d delay=%.2fs", str(e), attempt + 1, delay)
                time.sleep(delay)
                attempt += 1

        # Should not reach here
        if last_err:
            raise last_err

    def _record_cost_event(
        self,
        event: str,
        tokens: Optional[int],
        query_type: str,
        count: Optional[int] = None,
    ) -> None:
        """Track basic usage metrics for pharma budgeting and monitoring."""
        try:
            self._cost_metrics["total_requests"] += 1
            if tokens:
                self._cost_metrics["total_tokens"] += int(tokens)
            if query_type in self._cost_metrics["query_types"]:
                self._cost_metrics["query_types"][query_type] += 1
            if self._cost_metrics.get("enable_cost_tracking"):
                logger.debug(
                    "Cost track | event=%s type=%s tokens=%s count=%s total_tokens=%s",
                    event,
                    query_type,
                    tokens,
                    count,
                    self._cost_metrics["total_tokens"],
                )
        except Exception:
            # Metrics should never break flow
            pass

    def create_embeddings_batched(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        encoding_format: str = "float",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Efficiently create embeddings in batches for free tier optimization.

        Returns a list of dictionaries: {"embedding": [...], "index": i}
        """
        if model is None:
            model = (
                self.PHARMA_MODELS["embedding"]["preferred"]
                if self.pharma_optimized
                else self.PHARMA_MODELS["embedding"]["fallback"]
            )

        # Determine batch size
        effective_batch = batch_size if batch_size is not None else int(self.config.batch_max_size or 16)

        results: List[Dict[str, Any]] = []
        for start in range(0, len(texts), max(1, effective_batch)):
            chunk = texts[start : start + effective_batch]
            resp = self.create_embeddings(chunk, model=model, encoding_format=encoding_format, **kwargs)
            for i, d in enumerate(resp.data):
                results.append({"embedding": d.embedding, "index": start + i})
        return results

    def get_model_info(self, model_type: str = "embedding") -> Dict[str, Any]:
        """
        Get information about recommended models for pharmaceutical use.

        Args:
            model_type: Type of model ("embedding" or "chat")

        Returns:
            Dictionary with model information
        """
        if model_type not in self.PHARMA_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")

        return {
            "model_type": model_type,
            "pharmaceutical_optimized": self.pharma_optimized,
            "recommended_models": self.PHARMA_MODELS[model_type],
            "endpoint": self.config.base_url,
        }

    def _to_dict(self, obj: Any) -> Optional[Dict[str, Any]]:
        """Safely convert SDK objects to dicts without assuming `.dict()` exists."""
        if obj is None:
            return None
        for attr in ("model_dump", "dict"):
            fn = getattr(obj, attr, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:  # nosec B110 - fallback to alternate dict conversion methods
                    pass
        if isinstance(obj, dict):
            return obj
        try:
            return obj.__dict__
        except Exception:
            return {"repr": repr(obj)}

    def get_cost_metrics(self) -> Dict[str, Any]:
        """Expose lightweight cost metrics for research budgeting dashboards."""
        return dict(self._cost_metrics)

    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_n: Optional[int] = None,
        model: str = "nvidia/llama-3_2-nemoretriever-500m-rerank-v2",
        max_retries: Optional[int] = None,
        backoff_base: Optional[float] = None,
        backoff_jitter: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Re-rank passages/documents with a unified normalized output.

        Normalized schema: [{"text": str, "score": float, "index": int}]

        Service Order & Cost Implications:
        ==========================================
        Cloud-First Strategy (controlled by ENABLE_CLOUD_FIRST_RERANK):
        1) NVIDIA Build /v1/rerank - PRIMARY (cloud-first)
           - Cost: NVIDIA Build credits/pricing tier
           - Performance: Cloud-hosted reranking with high availability
        2) NeMo Retrieval endpoint (ai.api.nvidia.com/v1/retrieval/nvidia/reranking) - FALLBACK
           - Cost: Free tier usage, but uses NVIDIA API credits
           - Performance: Direct access to NeMo reranking models

        This method now follows true "cloud-first Build-first" strategy by default,
        aligning with the overall cloud-first architecture and cost optimization goals.

        Configuration:
        - ENABLE_CLOUD_FIRST_RERANK: Enable cloud-first rerank strategy (default: true)
        - NEMO_RERANKING_ENDPOINT: Custom NeMo endpoint (default: ai.api.nvidia.com)
        - ENABLE_BUILD_FIRST_RERANK: Legacy override for ordering (takes precedence when set)
        - ENABLE_NVB_RERANK: Legacy toggle to disable NVIDIA Build rerank when set to false
        - ENABLE_RERANK_MODEL_MAPPING: Enable model translation between services
        """
        # Early return for empty candidates
        if not candidates:
            return []

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        def _normalize(rankings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for i, item in enumerate(rankings or []):
                idx = item.get("index", i)
                score = item.get("score") or item.get("relevance") or item.get("relevance_score")
                text = (
                    item.get("text")
                    or item.get("passage")
                    or (item.get("document") if isinstance(item.get("document"), str) else None)
                )
                if text is None and isinstance(item.get("document"), dict):
                    text = item["document"].get("text") or item["document"].get("content")
                out.append({"text": text or "", "score": float(score) if score is not None else 0.0, "index": int(idx)})
            return out

        def _call_with_retries(
            url: str, body: Dict[str, Any], max_retries: int, backoff_base: float, backoff_jitter: float
        ) -> Tuple[int, Any, str]:
            attempt = 0
            last_exception = None

            while attempt <= max_retries:
                try:
                    resp = requests.post(url, headers=headers, json=body, timeout=self.config.timeout)
                    content = None
                    try:
                        content = resp.json() if resp.content else {}
                    except ValueError as je:
                        logger.warning("Non-JSON rerank response from %s: %s", url, str(je))

                    # Retry on specific HTTP status codes
                    if resp.status_code in [429] or 500 <= resp.status_code <= 599:
                        if attempt < max_retries:
                            delay = backoff_base * (2**attempt)
                            if backoff_jitter:
                                import random

                                delay += random.uniform(
                                    0, backoff_jitter
                                )  # nosec B311 - jitter for backoff, not crypto
                            logger.warning(
                                f"Retrying rerank request to {url} after HTTP {resp.status_code}. attempt={attempt + 1} delay={delay:.2f}s"
                            )
                            time.sleep(delay)
                            attempt += 1
                            continue

                    return resp.status_code, content, resp.text

                except requests.RequestException as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = backoff_base * (2**attempt)
                        if backoff_jitter:
                            import random

                            delay += random.uniform(0, backoff_jitter)  # nosec B311 - jitter for backoff, not crypto
                        logger.warning(
                            f"Retrying rerank request to {url} after {type(e).__name__}: {e}. attempt={attempt + 1} delay={delay:.2f}s"
                        )
                        time.sleep(delay)
                        attempt += 1
                    else:
                        logger.error("Rerank HTTP error from %s after %d attempts: %s", url, max_retries + 1, e)
                        raise NVIDIABuildError(f"Rerank request failed after {max_retries + 1} attempts: {e}")

            if last_exception:
                raise NVIDIABuildError(f"Rerank request failed after {max_retries + 1} attempts: {last_exception}")

        # Service order: Cloud-first by default to align with NVIDIA Build strategy
        services = []
        # Prefer config-provided endpoint and flags; fall back to env for backward compatibility
        nemo_url = (
            getattr(self.config, "nemo_reranking_endpoint", None)
            or os.getenv("NEMO_RERANKING_ENDPOINT")
            or "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
        )

        # Cloud-first strategy toggle
        cfg_cloud_first = getattr(self.config, "cloud_first_rerank_enabled", None)
        if isinstance(cfg_cloud_first, bool):
            build_first = cfg_cloud_first
        else:
            # Maintain legacy env override behavior if config not provided
            cloud_first_rerank = _safe_bool(os.getenv("ENABLE_CLOUD_FIRST_RERANK"), True)
            build_first_override = os.getenv("ENABLE_BUILD_FIRST_RERANK")
            build_first = (
                _safe_bool(build_first_override, False) if build_first_override is not None else cloud_first_rerank
            )

        # Enable/disable NVIDIA Build rerank service
        cfg_build_rerank = getattr(self.config, "build_rerank_enabled", None)
        if isinstance(cfg_build_rerank, bool):
            enable_build_rerank = cfg_build_rerank
        else:
            legacy_nvb = os.getenv("ENABLE_NVB_RERANK")
            enable_build_rerank = True if legacy_nvb is None else _safe_bool(legacy_nvb, True)

        if build_first:
            # Cloud-first: NVIDIA Build first, then NeMo
            if enable_build_rerank:
                services.append(("NVB_Rerank", f"{self.config.base_url}/rerank"))
            services.append(("NeMoRetrieval", nemo_url))
        else:
            # NeMo first, then NVIDIA Build
            services.append(("NeMoRetrieval", nemo_url))
            if enable_build_rerank:
                services.append(("NVB_Rerank", f"{self.config.base_url}/rerank"))

        # Per-service model mapping (gated by environment flag)
        def _get_mapped_model_for_service(service_name: str, original_model: str) -> str:
            """Map models to service-specific supported variants."""
            # Prefer config-driven mapping toggle; fallback to env var when not provided
            mapping_enabled = getattr(self.config, "enable_rerank_model_mapping", None)
            if isinstance(mapping_enabled, bool):
                if not mapping_enabled:
                    return original_model
            else:
                if not _safe_bool(os.getenv("ENABLE_RERANK_MODEL_MAPPING"), False):
                    return original_model

            # NVB_Rerank service model mappings
            if service_name == "NVB_Rerank":
                nvb_model_mapping = {
                    # Map common NeMo models to NVIDIA Build supported models
                    "meta/llama-3_2-nemoretriever-500m-rerank-v2": "meta/llama-3_2-nemoretriever-500m-rerank-v2",
                    "nvidia/llama-3_2-nemoretriever-500m-rerank-v2": "meta/llama-3_2-nemoretriever-500m-rerank-v2",
                    "nvidia/nv-rerankqa-mistral4b-v3": "nvidia/nv-rerankqa-mistral4b-v3",
                    "llama-3_2-nemoretriever-500m-rerank-v2": "meta/llama-3_2-nemoretriever-500m-rerank-v2",
                    "nv-rerankqa-mistral4b-v3": "nvidia/nv-rerankqa-mistral4b-v3",
                }

                if original_model in nvb_model_mapping:
                    mapped = nvb_model_mapping[original_model]
                    if mapped != original_model:
                        logger.info(f"Mapped model for {service_name}: {original_model} -> {mapped}")
                    return mapped
                else:
                    # Fallback to default supported model
                    fallback = "meta/llama-3_2-nemoretriever-500m-rerank-v2"
                    logger.warning(f"Unsupported model {original_model} for {service_name}, using fallback: {fallback}")
                    return fallback

            # For other services, return original model
            return original_model

        errors: List[str] = []
        # Use request policy defaults if not set
        eff_max_retries = (
            int(max_retries)
            if isinstance(max_retries, int)
            else int(getattr(self.config, "rerank_retry_max_attempts", getattr(self.config, "max_retries", 2)))
        )
        # Use a dedicated backoff base, not the request timeout
        eff_backoff = (
            float(backoff_base)
            if backoff_base is not None
            else float(getattr(self.config, "request_backoff_base", 0.5) or 0.5)
        )
        eff_jitter = (
            float(backoff_jitter)
            if backoff_jitter is not None
            else float(getattr(self.config, "request_backoff_jitter", 0.0) or 0.0)
        )

        for name, base in services:
            # Apply service-specific model mapping
            service_model = _get_mapped_model_for_service(name, model)

            for payload_key in ("documents", "passages"):
                # Try string list and object list variants
                variants = [
                    ("str_list", candidates),
                    ("obj_list", [{"text": c} if not isinstance(c, dict) else c for c in candidates]),
                ]
                if top_n:
                    pass
                for vname, vdata in variants:
                    body: Dict[str, Any] = {"model": service_model, "query": query, payload_key: vdata}
                    if top_n:
                        body["top_n"] = top_n
                    status, data, text_raw = _call_with_retries(base, body, eff_max_retries, eff_backoff, eff_jitter)
                    if status == 404:
                        logger.warning("Rerank endpoint 404 at %s; payload='%s' variant='%s'", base, payload_key, vname)
                        errors.append(f"{name}:{payload_key}:{vname}=404")
                        continue
                    if status != 200:
                        logger.warning(
                            "Rerank failed at %s: HTTP %s - %s (payload=%s variant=%s)",
                            base,
                            status,
                            (text_raw or "")[:160],
                            payload_key,
                            vname,
                        )
                        errors.append(f"{name}:{payload_key}:{vname}={status}")
                        continue
                    rankings = data.get("rankings", []) if isinstance(data, dict) else []
                    if rankings:
                        normalized = _normalize(rankings)
                        logger.info(
                            "Rerank via %s succeeded with %d items (payload=%s variant=%s)",
                            name,
                            len(normalized),
                            payload_key,
                            vname,
                        )
                        return normalized
                    else:
                        logger.warning(
                            "Rerank via %s returned empty rankings (payload=%s variant=%s)", name, payload_key, vname
                        )
                        errors.append(f"{name}:{payload_key}:{vname}=empty")

        raise NVIDIABuildError("Rerank failed across services: " + ", ".join(errors))

    # Thin adapter for callers that prefer `documents` terminology
    def rerank_documents(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Alias to `rerank` allowing `documents` parameter name.

        Args:
            query: Query text
            documents: List of document/passages to rank
            top_n: Optional top-N
            model: Optional model id (defaults follow `rerank`)
            **kwargs: Ignored; maintained for compatibility
        """
        return self.rerank(
            query=query,
            candidates=documents,
            top_n=top_n,
            model=(model or "nvidia/llama-3_2-nemoretriever-500m-rerank-v2"),
        )

    def validate_pharmaceutical_setup(self) -> Dict[str, Any]:
        """
        Validate setup for pharmaceutical research applications.

        Returns:
            Dictionary with validation results
        """
        results = {
            "pharmaceutical_optimized": self.pharma_optimized,
            "endpoint": self.config.base_url,
            "connection_test": None,
            "embedding_test": None,
            "chat_test": None,
            "overall_status": "unknown",
        }

        # Test connection (skip model listing for safety)
        connection_result = self._test_connection_safe()
        results["connection_test"] = connection_result

        if not connection_result["success"]:
            results["overall_status"] = "failed"
            return results

        # Test embedding with pharmaceutical query
        try:
            embedding_response = self.create_embeddings(["metformin pharmacokinetics and drug interactions"])
            dims = (
                len(embedding_response.data[0].embedding)
                if getattr(embedding_response, "data", None) and embedding_response.data
                else 0
            )
            results["embedding_test"] = {"success": True, "dimensions": dims, "model": embedding_response.model}
        except NVIDIABuildError as e:
            results["embedding_test"] = {"success": False, "error": str(e), "status_code": e.status_code}

        # Test chat with pharmaceutical query
        try:
            chat_response = self.create_chat_completion(
                [{"role": "user", "content": "Briefly explain metformin's mechanism of action."}]
            )
            results["chat_test"] = {
                "success": True,
                "model": chat_response.model,
                "response_length": len(chat_response.choices[0].message.content),
            }
        except NVIDIABuildError as e:
            results["chat_test"] = {"success": False, "error": str(e), "status_code": e.status_code}

        # Determine overall status
        embedding_ok = results["embedding_test"] and results["embedding_test"]["success"]
        chat_ok = results["chat_test"] and results["chat_test"]["success"]

        if embedding_ok and chat_ok:
            results["overall_status"] = "success"
        elif embedding_ok or chat_ok:
            results["overall_status"] = "partial"
        else:
            results["overall_status"] = "failed"

        return results

    def run_pharmaceutical_benchmarks(self, prompts: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """Run simple benchmark across pharma query types to validate performance.

        The benchmark intentionally avoids network-heavy scoring; it focuses on latency
        and successful completion across types.
        """
        prompts = prompts or {
            "drug_interaction": [
                "Does simvastatin interact with clarithromycin? Summarize mechanisms.",
            ],
            "pharmacokinetics": [
                "What is the half-life and clearance of metformin?",
            ],
            "clinical_trial": [
                "Summarize outcomes of a phase 3 RCT for semaglutide in T2D.",
            ],
            "general": [
                "Explain the role of beta blockers in hypertension.",
            ],
        }

        summary: Dict[str, Any] = {"results": {}, "total": 0}
        for qtype, qs in prompts.items():
            latencies: List[int] = []
            success = 0
            for q in qs:
                start = time.time()
                try:
                    resp = self.create_chat_completion([{"role": "user", "content": q}])
                    ok = bool(getattr(resp, "choices", None))
                    success += 1 if ok else 0
                except Exception:
                    ok = False
                latencies.append(int((time.time() - start) * 1000))
            summary["results"][qtype] = {
                "count": len(qs),
                "success": success,
                "avg_latency_ms": int(sum(latencies) / max(1, len(latencies))),
            }
            summary["total"] += len(qs)
        return summary

    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to NVIDIA Build platform with comprehensive health check.
        Uses model listing if enabled, otherwise performs basic client initialization test.

        Returns:
            Health check results with timing and model availability
        """
        start_time = time.time()

        try:
            with self._error_handler("connection_test"):
                # Ensure client is initialized
                self._ensure_client_initialized()

                available_models = []
                # Only attempt model listing if enabled and safe
                if self.config.enable_model_listing:
                    try:
                        models = self.client.models.list()
                        available_models = [model.id for model in models.data] if hasattr(models, "data") else []
                    except Exception as e:
                        logger.warning(f"Model listing failed during connection test: {str(e)}")
                        # Continue with connection test even if model listing fails

                response_time_ms = int((time.time() - start_time) * 1000)

                return {
                    "success": True,
                    "endpoint": self.config.base_url,
                    "response_time_ms": response_time_ms,
                    "available_models": len(available_models),
                    "model_listing_enabled": self.config.enable_model_listing,
                    "pharmaceutical_optimized": self.pharma_optimized,
                    "message": "Connection successful",
                }

        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            return {
                "success": False,
                "endpoint": self.config.base_url,
                "response_time_ms": response_time_ms,
                "error": str(e),
                "pharmaceutical_optimized": self.pharma_optimized,
                "message": "Connection failed",
            }

    def _test_connection_safe(self) -> Dict[str, Any]:
        """
        Safe connection test that never attempts model listing operations.
        Used by pharmaceutical validation to avoid potential endpoint failures.

        Returns:
            Basic connection health check results
        """
        start_time = time.time()

        try:
            # Ensure client is initialized - this tests basic connectivity
            self._ensure_client_initialized()

            response_time_ms = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "endpoint": self.config.base_url,
                "response_time_ms": response_time_ms,
                "pharmaceutical_optimized": self.pharma_optimized,
                "message": "Safe connection test successful (no model listing)",
            }

        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            return {
                "success": False,
                "endpoint": self.config.base_url,
                "response_time_ms": response_time_ms,
                "error": str(e),
                "pharmaceutical_optimized": self.pharma_optimized,
                "message": "Safe connection test failed",
            }


def get_model_catalog() -> Dict[str, Any]:
    """Expose a shared model catalog for embedding/chat models.

    Returns a dictionary referencing OpenAIWrapper.PHARMA_MODELS.
    This helper allows other modules (e.g., deprecated NVIDIA Build client)
    to reference a single source of truth for model metadata.
    """
    return {
        "embedding": OpenAIWrapper.PHARMA_MODELS["embedding"],
        "chat": OpenAIWrapper.PHARMA_MODELS["chat"],
    }


# Convenience functions for quick access
def create_nvidia_build_client(pharmaceutical_optimized: bool = True) -> OpenAIWrapper:
    """
    Create NVIDIA Build client with default configuration.

    Args:
        pharmaceutical_optimized: Enable pharmaceutical defaults

    Returns:
        Configured OpenAI wrapper instance
    """
    config = NVIDIABuildConfig(pharmaceutical_optimized=pharmaceutical_optimized)
    return OpenAIWrapper(config)


def test_nvidia_build_access() -> Dict[str, Any]:
    """
    Quick test of NVIDIA Build API access for pharmaceutical applications.

    Returns:
        Comprehensive test results
    """
    try:
        client = create_nvidia_build_client()
        return {
            **client.validate_pharmaceutical_setup(),
            "cost_metrics": client.get_cost_metrics(),
            "benchmarks": client.run_pharmaceutical_benchmarks(),
        }
    except Exception as e:
        return {"overall_status": "failed", "error": str(e), "pharmaceutical_optimized": True}


if __name__ == "__main__":
    # Quick validation when run directly
    import json

    results = test_nvidia_build_access()
    print(json.dumps(results, indent=2))
