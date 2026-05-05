"""
Source metadata manipulation utilities.

Provides functions for safely updating source metadata without
in-place mutations, preserving pharmaceutical safety compliance.
"""
import logging
from copy import deepcopy
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _clone_source_for_update(source: Any) -> Any:
    """Return a shallowly cloned source object to avoid in-place mutations."""
    try:
        return deepcopy(source)
    except Exception:
        if isinstance(source, dict):
            cloned = dict(source)
            if "metadata" in source and isinstance(source["metadata"], dict):
                cloned["metadata"] = dict(source["metadata"])
            return cloned
        return source


def _ensure_metadata_dict(source: Any) -> Dict[str, Any]:
    """Ensure the source exposes a mutable metadata dictionary and return it."""
    if isinstance(source, dict):
        metadata = dict(source.get("metadata", {}))
        source["metadata"] = metadata
        return metadata

    metadata = getattr(source, "metadata", None)
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        try:
            metadata = dict(metadata)
        except Exception:
            metadata = {"raw_metadata": metadata}
    setattr(source, "metadata", metadata)
    return metadata


def _get_source_value(source: Any, key: str, default: Any = None) -> Any:
    """Fetch a value from a source dict/object, falling back to metadata."""
    if isinstance(source, dict):
        if key in source:
            return source[key]
        metadata = source.get("metadata", {})
        if isinstance(metadata, dict):
            return metadata.get(key, default)
        return default

    if hasattr(source, key):
        return getattr(source, key)
    metadata = getattr(source, "metadata", {})
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return default


async def update_source_metadata(
    source: Any, updates: Optional[Dict[str, Any]] = None, key: Optional[str] = None, value: Any = None
) -> Any:
    """Return a copy of the source with metadata updated without in-place mutation."""
    try:
        update_map: Dict[str, Any] = updates.copy() if updates else {}
        if key is not None:
            update_map[key] = value

        if not update_map:
            return source

        updated_source = _clone_source_for_update(source)
        metadata = _ensure_metadata_dict(updated_source)
        metadata.update(update_map)
        return updated_source
    except Exception as exc:
        logger.error(f"Error updating source metadata: {exc}")
        return source


async def set_source_flag(source: Any, key: str, value: Any) -> Any:
    """Return a copy of the source with a top-level attribute/key set."""
    try:
        updated_source = _clone_source_for_update(source)
        if isinstance(updated_source, dict):
            updated_source[key] = value
        else:
            try:
                setattr(updated_source, key, value)
            except Exception:
                metadata = _ensure_metadata_dict(updated_source)
                metadata[key] = value
        return updated_source
    except Exception as exc:
        logger.error(f"Error setting source flag '{key}': {exc}")
        return source


async def append_source_warning(source: Any, warning: str) -> Any:
    """Add a warning message to the source metadata without duplicates."""
    if not warning:
        return source

    try:
        updated_source = _clone_source_for_update(source)

        if hasattr(updated_source, "add_warning") and callable(getattr(updated_source, "add_warning")):
            existing_warnings = getattr(updated_source, "warnings", None)
            if isinstance(existing_warnings, list) and warning in existing_warnings:
                return updated_source
            updated_source.add_warning(warning)
            return updated_source

        metadata = _ensure_metadata_dict(updated_source)
        warnings = metadata.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = [warnings]
        if warning not in warnings:
            warnings.append(warning)
        metadata["warnings"] = warnings
        return updated_source
    except Exception as exc:
        logger.error(f"Error appending source warning: {exc}")
        return source
