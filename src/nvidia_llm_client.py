"""
NVIDIA Build API LLM Client Wrapper

This module provides a focused wrapper for NVIDIA Build API LLM chat completions,
specifically optimized for pharmaceutical and medical use cases.

Supported Models:
- meta/llama-3.1-8b-instruct (8B) - Fast responses for simple queries
- meta/llama-3.3-70b-instruct (70B) - Complex medical reasoning

API Documentation: https://build.nvidia.com/
Rate Limits: 10,000 requests/month (free tier)

Pharmaceutical Optimizations:
- Lower default temperature (0.3) for medical accuracy
- Conservative token limits (800) for focused answers
- Comprehensive logging for audit trails
- Retry logic for reliability

WARNING: This tool is for research and educational purposes only.
Always consult qualified healthcare professionals for medical decisions.

Example Usage:
    >>> from nvidia_llm_client import NVIDIALLMClient
    >>> client = NVIDIALLMClient()
    >>> response = client.generate_simple("What is aspirin?", model="8b")
    >>> print(response)

    Or use the convenience function:
    >>> from nvidia_llm_client import generate_with_nvidia
    >>> response = generate_with_nvidia("What is aspirin?", model="8b")
"""
import logging
import os
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# Configure module logger
logger = logging.getLogger(__name__)


def _parse_env_float(var_name: str, default: float) -> float:
    value = os.getenv(var_name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_env_int(var_name: str, default: int) -> int:
    value = os.getenv(var_name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ============================================================================
# Custom Exceptions
# ============================================================================


class NVIDIALLMError(Exception):
    """Base exception for NVIDIA LLM client errors."""



class NVIDIALLMAPIError(NVIDIALLMError):
    """
    API-specific errors (rate limits, authentication, model unavailable).

    This exception is raised when the NVIDIA Build API returns an error response,
    such as rate limiting, authentication failures, or model unavailability.

    Attributes:
        status_code: HTTP status code if available (e.g., 401, 404, 429)
        response_data: Response data from the API if available
    """

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class NVIDIALLMConfigError(NVIDIALLMError):
    """
    Configuration errors (missing API key, invalid model).

    This exception is raised when the client is misconfigured, such as
    missing API keys or invalid model selections.
    """



# ============================================================================
# Deferred Import Helper
# ============================================================================


def _get_openai_client(enable_logging: bool = True):
    """
    Import and return OpenAI client class and exception classes with deferred loading.

    This follows the pattern from openai_wrapper.py to handle environments
    where the OpenAI SDK may not be installed.

    Args:
        enable_logging: Whether to log import failures.

    Returns:
        Tuple of (OpenAI, APIError, APIStatusError, RateLimitError, APIConnectionError)

    Raises:
        NVIDIALLMConfigError: If OpenAI SDK is not installed
    """
    try:
        from openai import APIConnectionError, APIError, APIStatusError, OpenAI, RateLimitError

        return OpenAI, APIError, APIStatusError, RateLimitError, APIConnectionError
    except ImportError as e:
        if enable_logging:
            logger.error("OpenAI SDK not installed. Install with: pip install openai>=1.0.0")
        raise NVIDIALLMConfigError(
            "OpenAI SDK is required for NVIDIA Build API client. " "Install it with: pip install openai>=1.0.0"
        ) from e


def _instantiate_openai_client(openai_cls, api_key: str, base_url: str):
    """Create an OpenAI-compatible client while tolerating minimal test doubles."""
    try:
        return openai_cls(api_key=api_key, base_url=base_url)
    except TypeError:
        client = openai_cls()
        try:
            setattr(client, "api_key", api_key)
        except Exception:
            pass
        try:
            setattr(client, "base_url", base_url)
        except Exception:
            pass
        return client


# ============================================================================
# NVIDIA LLM Client
# ============================================================================


class NVIDIALLMClient:
    """
    Client wrapper for NVIDIA Build API LLM chat completions.

    This client provides a focused interface for generating text using NVIDIA's
    hosted Llama models (8B and 70B) via the OpenAI-compatible API.

    The client is optimized for pharmaceutical and medical use cases with:
    - Conservative default parameters (temp=0.3)
    - Comprehensive error handling and retry logic
    - Detailed logging for audit trails
    - Support for both simple and complex queries

    Attributes:
        MODELS: Mapping of model shortcuts to full model IDs
        BASE_URL: NVIDIA Build API endpoint
        DEFAULT_TEMPERATURE: Conservative temperature for medical accuracy
        DEFAULT_MAX_TOKENS: Default token limit for responses
        MAX_RETRIES: Maximum retry attempts for transient errors
        RETRY_DELAY_SECONDS: Initial retry delay (with exponential backoff)
    """

    # Model mappings
    MODELS = {"8b": "meta/llama-3.1-8b-instruct", "70b": "meta/llama-3.3-70b-instruct"}

    # API configuration
    BASE_URL = "https://integrate.api.nvidia.com/v1"

    # Pharmaceutical-optimized defaults
    DEFAULT_TEMPERATURE = _parse_env_float("LLM_TEMPERATURE", 0.3)  # Lower for medical accuracy
    DEFAULT_MAX_TOKENS = _parse_env_int("LLM_MAX_TOKENS", 800)  # Sufficient for patient answers

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 1.0

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, enable_logging: bool = True):
        """
        Initialize NVIDIA LLM client.

        Args:
            api_key: NVIDIA API key (or set NVIDIA_API_KEY env var)
            base_url: Override default NVIDIA Build API endpoint
            enable_logging: Enable detailed logging (default: True)

        Raises:
            NVIDIALLMConfigError: If API key is missing or invalid
        """
        # Configure logging immediately so it applies to all initialization logs
        self.enable_logging = enable_logging

        # Retrieve API key from parameter or environment
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            error_msg = (
                "NVIDIA API key not found. Please provide api_key parameter or "
                "set NVIDIA_API_KEY environment variable. "
                "Get your API key at: https://build.nvidia.com/"
            )
            if self.enable_logging:
                logger.error(error_msg)
            raise NVIDIALLMConfigError(error_msg)

        # Configure base URL
        self.base_url = base_url or self.BASE_URL
        self.default_temperature = _parse_env_float("LLM_TEMPERATURE", self.DEFAULT_TEMPERATURE)
        self.default_max_tokens = _parse_env_int("LLM_MAX_TOKENS", self.DEFAULT_MAX_TOKENS)
        self.last_usage_tokens = None
        self.model = None

        # Store OpenAI error classes for downstream handling
        self.APIError = None
        self.APIStatusError = None
        self.RateLimitError = None
        self.APIConnectionError = None

        # Initialize OpenAI client with deferred import
        try:
            OpenAI, APIError, APIStatusError, RateLimitError, APIConnectionError = _get_openai_client(
                self.enable_logging
            )
            self.APIError = APIError
            self.APIStatusError = APIStatusError
            self.RateLimitError = RateLimitError
            self.APIConnectionError = APIConnectionError
            self.client = _instantiate_openai_client(OpenAI, self.api_key, self.base_url)
            if self.enable_logging:
                logger.info(f"NVIDIA LLM client initialized: base_url={self.base_url}")
        except NVIDIALLMConfigError as e:
            # Re-raise configuration errors without wrapping to preserve original message
            if self.enable_logging:
                logger.error(f"Configuration error: {e}")
            raise
        except Exception as e:
            # Wrap unexpected initialization failures
            if self.enable_logging:
                logger.error(f"Failed to initialize OpenAI client: {e}")
            raise NVIDIALLMConfigError(f"Failed to initialize client: {e}") from e

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = "8b",
        temperature: float = None,
        max_tokens: int = None,
        **kwargs,
    ) -> str:
        """
        Generate text using NVIDIA Build API chat completions.

        This is the main generation method that accepts a list of messages
        in OpenAI chat format and returns the generated text.

        Note:
            Invalid model identifiers are logged and automatically coerced to the
            default 8B model to maintain safe behavior.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     Example: [{"role": "user", "content": "What is aspirin?"}]
            model: Model identifier ("8b"/"70b" shortcuts or their full NVIDIA IDs).
                Defaults to "8b". Invalid values log a warning and fall back to "8b".
            temperature: Sampling temperature (default: 0.3 for medical accuracy)
            max_tokens: Maximum tokens to generate (default: 800)
            **kwargs: Additional parameters passed to OpenAI API

        Returns:
            Generated text string

        Raises:
            NVIDIALLMAPIError: If API request fails

        Example:
            >>> messages = [{"role": "user", "content": "What is aspirin?"}]
            >>> response = client.generate(messages, model="8b")
        """
        # Resolve model identifier; fallback to default if unsupported
        if model in self.MODELS:
            full_model_id = self.MODELS[model]
        elif model in self.MODELS.values():
            full_model_id = model
        else:
            valid_values = list(self.MODELS.keys()) + list(self.MODELS.values())
            if self.enable_logging:
                logger.warning(
                    "Unsupported model '%s'. Falling back to default '8b' (%s). Allowed values: %s",
                    model,
                    self.MODELS["8b"],
                    valid_values,
                )
            full_model_id = self.MODELS["8b"]

        # Apply defaults
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        # Log request
        if self.enable_logging:
            logger.info(
                f"Generating with model={full_model_id}, temperature={temperature}, "
                f"max_tokens={max_tokens}, messages={len(messages)}"
            )
        self.model = full_model_id

        # Make API call with retry logic
        start_time = time.time()
        self.last_usage_tokens = None
        try:
            response = self._retry_with_backoff(
                self.client.chat.completions.create,
                model=full_model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Extract generated text
            if not response.choices:
                raise NVIDIALLMAPIError("API returned empty response")

            generated_text = response.choices[0].message.content

            if isinstance(generated_text, list):
                segments = []
                for segment in generated_text:
                    if isinstance(segment, str):
                        segments.append(segment)
                    elif isinstance(segment, dict):
                        text_value = segment.get("text") or segment.get("value")
                        if text_value:
                            segments.append(str(text_value))
                    else:
                        text_value = getattr(segment, "text", None) or getattr(segment, "value", None)
                        if text_value:
                            segments.append(str(text_value))
                generated_text = "".join(segments) if segments else None
            elif isinstance(generated_text, dict):
                generated_text = generated_text.get("text") or generated_text.get("value")
            elif generated_text is None:
                generated_text = None
            else:
                attr_text = getattr(generated_text, "text", None) or getattr(generated_text, "value", None)
                if attr_text:
                    generated_text = attr_text

            if not generated_text:
                raise NVIDIALLMAPIError("API returned empty content")

            # Log success
            elapsed_ms = (time.time() - start_time) * 1000
            if hasattr(response, "usage") and response.usage:
                self.last_usage_tokens = getattr(response.usage, "total_tokens", None)

            if self.enable_logging:
                token_info = ""
                if self.last_usage_tokens is not None:
                    token_info = f", tokens={self.last_usage_tokens}"
                logger.info(
                    f"Generated {len(generated_text)} characters in {elapsed_ms:.0f}ms "
                    f"using {full_model_id}{token_info}"
                )

            return generated_text

        except Exception as e:
            # Log error with context
            if self.enable_logging:
                logger.error(
                    f"Generation failed: {e} (model={full_model_id}, "
                    f"temperature={temperature}, max_tokens={max_tokens})"
                )

            # Wrap in custom exception if not already
            if isinstance(e, (NVIDIALLMError,)):
                raise

            # Check for specific HTTP status codes to provide actionable error messages
            status_code, api_error_code = self._extract_status_and_code(e)
            error_payload = {
                "response": getattr(e, "response", None),
                "body": getattr(e, "body", None),
            }

            message_attr = getattr(e, "message", None)
            if message_attr is not None:
                error_payload["message"] = message_attr

            error_attr = getattr(e, "error", None)
            if error_attr is not None:
                error_payload["error"] = error_attr

            if status_code is not None:
                error_payload["status_code"] = status_code

            if api_error_code is not None:
                error_payload["code"] = api_error_code

            if status_code == 401:
                raise NVIDIALLMAPIError(
                    "Authentication failed. Please verify NVIDIA_API_KEY is correct.",
                    status_code=401,
                    response_data=error_payload,
                ) from e
            elif status_code == 404:
                raise NVIDIALLMAPIError(
                    f"Model '{full_model_id}' not found. Please verify the model ID is correct.",
                    status_code=404,
                    response_data=error_payload,
                ) from e
            else:
                # Preserve status code in wrapped exception for all other errors
                raise NVIDIALLMAPIError(
                    f"API request failed: {e}", status_code=status_code, response_data=error_payload
                ) from e

    def generate_simple(
        self, prompt: str, model: str = "8b", temperature: float = None, system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        """
        Generate text from a simple prompt (convenience wrapper).

        This method provides a simpler interface for single-turn generation
        without needing to construct message lists manually.

        Args:
            prompt: User prompt text
            model: Model shorthand ("8b" or "70b", default: "8b")
            temperature: Sampling temperature (default: 0.3)
            system_prompt: Optional system prompt to set context
            **kwargs: Additional parameters passed to generate()

        Returns:
            Generated text string

        Example:
            >>> response = client.generate_simple(
            ...     "What is aspirin?",
            ...     model="8b",
            ...     system_prompt="You are a medical expert."
            ... )
        """
        # Build messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call main generate method
        return self.generate(messages=messages, model=model, temperature=temperature, **kwargs)

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """
        Execute function with exponential backoff retry logic.

        This method retries transient errors (rate limits, timeouts, server errors)
        with exponential backoff, following patterns from openai_wrapper.py.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            Exception: If all retry attempts are exhausted
        """

        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                pass

                is_retryable = False
                status_code, api_error_code = self._extract_status_and_code(e)

                retryable_types = tuple(
                    exc
                    for exc in (
                        self.RateLimitError,
                        self.APIConnectionError,
                    )
                    if exc is not None
                )
                if retryable_types and isinstance(e, retryable_types):
                    is_retryable = True

                if api_error_code is not None:
                    retryable_api_codes = {
                        "rate_limit_exceeded",
                        "server_error",
                        "engine_overloaded",
                        "service_unavailable",
                    }
                    if api_error_code in retryable_api_codes:
                        is_retryable = True

                if not is_retryable and status_code:
                    # Non-retryable: Authentication/authorization errors
                    if status_code in [401, 403]:
                        if self.enable_logging:
                            logger.error(f"Authentication/authorization error (HTTP {status_code}): {e}")
                        raise

                    # Non-retryable: Model not found
                    elif status_code == 404:
                        if self.enable_logging:
                            logger.error(f"Model not found (HTTP {status_code}): {e}")
                        raise

                    # Retryable: Rate limiting
                    elif status_code == 429:
                        is_retryable = True

                    # Retryable: Server errors
                    elif 500 <= status_code < 600:
                        is_retryable = True

                if not is_retryable and not status_code:
                    # Fallback to string matching for non-HTTP exceptions (timeouts, connection errors)
                    error_str = str(e).lower()
                    is_timeout = "timeout" in error_str or "timed out" in error_str
                    is_connection = "connection" in error_str
                    is_retryable = is_timeout or is_connection

                # If not retryable or last attempt, raise immediately
                if not is_retryable or attempt == self.MAX_RETRIES - 1:
                    raise

                # Calculate backoff delay (exponential)
                delay = self.RETRY_DELAY_SECONDS * (2**attempt)

                if self.enable_logging:
                    logger.warning(f"Retry {attempt + 1}/{self.MAX_RETRIES} after {delay}s: {e}")

                time.sleep(delay)

    def _extract_status_and_code(self, exception: Exception) -> (Optional[int], Optional[str]):
        """
        Extract HTTP status code and API error code (if available) from exceptions.

        Args:
            exception: Exception raised by the OpenAI client.

        Returns:
            Tuple of (status_code, api_error_code)
        """
        status_code = getattr(exception, "status_code", None)
        if status_code is None:
            response_obj = getattr(exception, "response", None)
            status_code = getattr(response_obj, "status_code", None)

        api_error_code = getattr(exception, "code", None)

        openai_specific_types = [
            ("APIStatusError", self.APIStatusError),
            ("APIError", self.APIError),
        ]

        for _, exc_type in openai_specific_types:
            if exc_type is not None and isinstance(exception, exc_type):
                if status_code is None:
                    status_code = getattr(exception, "status_code", None) or getattr(
                        getattr(exception, "response", None), "status_code", None
                    )
                if api_error_code is None:
                    api_error_code = getattr(exception, "code", None)

        return status_code, api_error_code

    def test_connection(self, model: str = "8b") -> Dict[str, Any]:
        """
        Test API connectivity with a simple generation.

        This method validates that the API key is working and the specified
        model is accessible.

        Args:
            model: Model to test ("8b" or "70b", default: "8b")

        Returns:
            Dict with test results:
                - success: bool
                - model_tested: str
                - response_time_ms: float
                - status_code: Optional[int]
                - error: Optional[str]
                - error_type: Optional[str]
                - response_data: Optional[Any]

        Example:
            >>> result = client.test_connection(model="8b")
            >>> if result["success"]:
            ...     print(f"Connected in {result['response_time_ms']:.0f}ms")
        """
        test_prompt = "What is aspirin? Answer in one sentence."

        start_time = time.time()
        try:
            response = self.generate_simple(test_prompt, model=model)
            elapsed_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "model_tested": self.MODELS.get(model, model),
                "response_time_ms": elapsed_ms,
                "latency_ms": elapsed_ms,
                "base_url": self.base_url,
                "status_code": None,
                "error": None,
            }

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            error_info = {
                "success": False,
                "model_tested": self.MODELS.get(model, model),
                "response_time_ms": elapsed_ms,
                "latency_ms": elapsed_ms,
                "base_url": getattr(self, "base_url", self.BASE_URL),
                "status_code": getattr(e, "status_code", None),
                "error": str(e),
                "error_type": e.__class__.__name__,
            }
            if isinstance(e, NVIDIALLMAPIError):
                error_info["response_data"] = getattr(e, "response_data", None)
            elif isinstance(e, NVIDIALLMConfigError):
                error_info["response_data"] = None
            return error_info


# ============================================================================
# Module-Level Convenience Functions
# ============================================================================


def generate_with_nvidia(prompt: str, model: str = "8b", api_key: Optional[str] = None, **kwargs) -> str:
    """
    Quick generation wrapper (creates client and generates in one call).

    This convenience function is useful for one-off generations without
    managing a client instance.

    Args:
        prompt: User prompt text
        model: Model shorthand ("8b" or "70b", default: "8b")
        api_key: NVIDIA API key (or use NVIDIA_API_KEY env var)
        **kwargs: Additional parameters passed to generate_simple()

    Returns:
        Generated text string

    Raises:
        NVIDIALLMError: If generation fails

    Example:
        >>> response = generate_with_nvidia(
        ...     "What is aspirin?",
        ...     model="8b"
        ... )
        >>> print(response)
    """
    client = NVIDIALLMClient(api_key=api_key, enable_logging=False)
    return client.generate_simple(prompt=prompt, model=model, **kwargs)


def test_nvidia_llm_access(api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick test function for API access validation.

    This function tests connectivity to both 8B and 70B models and returns
    comprehensive results for debugging API access issues.

    Args:
        api_key: NVIDIA API key (or use NVIDIA_API_KEY env var)

    Returns:
        Dict with test results:
            - api_key_valid: bool indicating whether model tests detected any authentication/authorization failures
            - models_tested: List[str]
            - results: Dict[str, Dict[str, Any]]
            - overall_success: bool
            - error: Optional[str] (only present when client initialization fails)

    Example:
        >>> results = test_nvidia_llm_access()
        >>> print(f"API Valid: {results['overall_success']}")
        >>> for model, result in results['results'].items():
        ...     print(f"{model}: {result['success']}")
    """
    try:
        client = NVIDIALLMClient(api_key=api_key)
    except NVIDIALLMConfigError as e:
        return {"api_key_valid": False, "models_tested": [], "results": {}, "overall_success": False, "error": str(e)}

    # Test both models
    models_to_test = ["8b", "70b"]
    results = {}

    for model in models_to_test:
        results[model] = client.test_connection(model=model)

    auth_error_codes = {401, 403}
    api_key_valid = not any(
        (not result["success"]) and result.get("status_code") in auth_error_codes for result in results.values()
    )

    # Determine overall success
    overall_success = all(r["success"] for r in results.values())

    return {
        "api_key_valid": api_key_valid,
        "models_tested": models_to_test,
        "results": results,
        "overall_success": overall_success,
    }


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "NVIDIALLMClient",
    "NVIDIALLMError",
    "NVIDIALLMAPIError",
    "NVIDIALLMConfigError",
    "generate_with_nvidia",
    "test_nvidia_llm_access",
]


# ============================================================================
# Main Block (for quick testing)
# ============================================================================

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("NVIDIA Build API LLM Client - Connection Test")
    print("=" * 70)

    # Test API access
    results = test_nvidia_llm_access()

    # Print results in formatted JSON
    print("\nTest Results:")
    print(json.dumps(results, indent=2))

    # Print summary
    print("\n" + "=" * 70)
    if results["overall_success"]:
        print("✓ All tests passed! NVIDIA Build API is accessible.")
    else:
        print("✗ Some tests failed. Check API key and network connectivity.")
    print("=" * 70)
