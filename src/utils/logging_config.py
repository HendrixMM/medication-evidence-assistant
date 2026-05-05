"""
Centralized logging configuration for pharmaceutical RAG system.

Provides standardized logging setup with appropriate levels and formatting
for production medical applications.
"""
import logging.config
import os
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional


def env_true(var_name: str, default: bool = False) -> bool:
    """
    Check if an environment variable is set to a truthy value.

    Args:
        var_name: Name of the environment variable to check
        default: Default value if the environment variable is not set

    Returns:
        True if the environment variable is set to a truthy value, False otherwise
    """
    value = os.getenv(var_name)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def get_log_level() -> int:
    """Get log level from environment variable."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def setup_logging(
    log_level: Optional[int] = None, log_file: Optional[str] = None, include_medical_context: bool = True
) -> None:
    """
    Setup centralized logging configuration.

    Args:
        log_level: Logging level (defaults to environment LOG_LEVEL)
        log_file: Optional log file path
        include_medical_context: Whether to include medical compliance formatting
    """
    if log_level is None:
        log_level = get_log_level()

    # Create logs directory if logging to file
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Medical compliance format includes timestamps and source tracking
    medical_format = "%(asctime)s | %(levelname)-8s | %(name)-20s | " "%(funcName)-15s:%(lineno)-4d | %(message)s"

    simple_format = "%(levelname)-8s | %(name)-20s | %(message)s"

    format_string = medical_format if include_medical_context else simple_format

    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": format_string, "datefmt": "%Y-%m-%d %H:%M:%S"},
            "detailed": {
                "format": (
                    "%(asctime)s | %(levelname)-8s | %(name)-30s | "
                    "%(pathname)s:%(lineno)d | %(funcName)s() | %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "stream": sys.stdout,
            }
        },
        "loggers": {
            # Pharmaceutical RAG system loggers
            "src": {"level": log_level, "handlers": ["console"], "propagate": False},
            "guardrails": {"level": log_level, "handlers": ["console"], "propagate": False},
            # Third-party library loggers (quieter)
            "requests": {"level": logging.WARNING, "handlers": ["console"], "propagate": False},
            "urllib3": {"level": logging.WARNING, "handlers": ["console"], "propagate": False},
            "faiss": {"level": logging.WARNING, "handlers": ["console"], "propagate": False},
            "langchain": {"level": logging.WARNING, "handlers": ["console"], "propagate": False},
        },
        "root": {"level": log_level, "handlers": ["console"]},
    }

    # Add file handler if log file specified
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "detailed",
            "filename": log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        }

        # Add file handler to all loggers
        for logger_config in config["loggers"].values():
            logger_config["handlers"].append("file")
        config["root"]["handlers"].append("file")

    logging.config.dictConfig(config)

    # Log configuration for medical compliance
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {logging.getLevelName(log_level)}")
    if include_medical_context:
        logger.info("Medical compliance logging enabled")
    if log_file:
        logger.info(f"File logging enabled: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with standardized configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Auto-setup logging when module is imported
if not logging.getLogger().handlers:
    setup_logging()


# Medical compliance utility functions
def log_medical_action(logger: logging.Logger, action: str, details: str = "") -> None:
    """Log medical action for compliance tracking."""
    message = f"MEDICAL_ACTION: {action}"
    if details:
        message += f" | {details}"
    logger.info(message)


def log_safety_check(logger: logging.Logger, check_type: str, result: str, confidence: float = 1.0) -> None:
    """Log safety check results for compliance tracking."""
    logger.info(f"SAFETY_CHECK: {check_type} | Result: {result} | Confidence: {confidence:.3f}")


def log_pharmaceutical_query(logger: logging.Logger, query: str, source: str = "unknown") -> None:
    """Log pharmaceutical queries for audit purposes."""
    sanitized_query = query[:100] + "..." if len(query) > 100 else query
    logger.info(f"PHARMA_QUERY: {sanitized_query} | Source: {source}")


def log_api_interaction(
    logger: logging.Logger, service: str, operation: str, status: str, context: Optional[Dict[str, Any]] = None
) -> None:
    """Log API interactions for monitoring."""
    context_str = f" | Context: {context}" if context else ""
    logger.info(f"API_INTERACTION: {service} | Operation: {operation} | Status: {status}{context_str}")


def log_api_timing(
    logger: logging.Logger,
    service: str,
    operation: str,
    duration_ms: float,
    success: bool = True,
    error_msg: Optional[str] = None,
) -> None:
    """Log API timing for performance monitoring."""
    status = "success" if success else "failure"
    error_str = f" | Error: {error_msg}" if error_msg else ""
    logger.info(f"API_TIMING: {service}.{operation} | Duration: {duration_ms:.2f}ms | Status: {status}{error_str}")


def log_cache_operation(
    logger: logging.Logger, operation: str, cache_key: str, hit: bool, data_size: Optional[int] = None
) -> None:
    """Log cache operations for debugging."""
    hit_str = "HIT" if hit else "MISS"
    size_str = f" | Size: {data_size}" if data_size is not None else ""
    key_preview = cache_key[:50] + "..." if len(cache_key) > 50 else cache_key
    logger.debug(f"CACHE_{operation.upper()}: {hit_str} | Key: {key_preview}{size_str}")


def log_pubmed_query_construction(
    logger: logging.Logger,
    original_query: str,
    pubmed_query: str,
    intent: str,
    entities: Optional[Dict[str, Any]] = None,
) -> None:
    """Log PubMed query construction for debugging."""
    original_preview = original_query[:80] + "..." if len(original_query) > 80 else original_query
    pubmed_preview = pubmed_query[:100] + "..." if len(pubmed_query) > 100 else pubmed_query
    entities_str = f" | Entities: {entities}" if entities else ""
    logger.info(f"PUBMED_QUERY: {original_preview} -> {pubmed_preview} | Intent: {intent}{entities_str}")


def log_pubmed_retrieval(
    logger: logging.Logger,
    query: str,
    result_count: int,
    duration_ms: float,
    cache_hit: bool = False,
    error: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log PubMed retrieval results."""
    cache_str = " (cached)" if cache_hit else ""
    error_str = f" | Error: {error}" if error else ""
    context_str = f" | Context: {context}" if context else ""
    query_preview = query[:80] + "..." if len(query) > 80 else query
    logger.info(
        f"PUBMED_RETRIEVAL{cache_str}: {result_count} articles | Duration: {duration_ms:.2f}ms | Query: {query_preview}{error_str}{context_str}"
    )


def log_source_relevance_validation(
    logger: logging.Logger,
    query: str,
    relevance_score: float,
    needs_refinement: bool,
    articles_count: int,
    matched_keywords: list[str],
    query_types: list[str],
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log source relevance validation results for monitoring."""
    refinement_str = "NEEDS_REFINEMENT" if needs_refinement else "ACCEPTABLE"
    keywords_str = ", ".join(matched_keywords[:5]) if matched_keywords else "none"
    types_str = ", ".join(query_types) if query_types else "generic"
    context_str = f" | Context: {context}" if context else ""
    query_preview = query[:80] + "..." if len(query) > 80 else query
    logger.info(
        f"SOURCE_RELEVANCE: {refinement_str} | Score: {relevance_score:.2f} | "
        f"Articles: {articles_count} | Keywords: {keywords_str} | "
        f"Types: {types_str} | Query: {query_preview}{context_str}"
    )
