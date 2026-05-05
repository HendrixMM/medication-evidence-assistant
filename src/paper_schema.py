"""Canonical paper schema used by synthesis and DDI processors."""
from __future__ import annotations

import html
import re
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

# Unified regex patterns for paper identifiers
DOI_PATTERN = r"(?:doi[:\s]*)?10\.\d{4,9}/[^\s)>\]]+"
PMID_PATTERN = r"(?:\()?pmid[\s:]*[-]?(\d+)(?:\))?"

# Alternative PMID pattern for extraction with case-insensitive matching
PMID_PATTERN_EXTRACT = r"(?:\()?PMID[\s:]*[-]?(\d+)(?:\))?"


class Paper(BaseModel):
    """Normalized representation of a literature paper used by analysis engines."""

    page_content: str = Field(default="", alias="page_content")
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    title: str | None = None
    abstract: str | None = None
    pmid: str | None = None
    pk_parameters: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    def model_post_init(self, __context: Any) -> None:  # pragma: no cover - exercised indirectly
        if not isinstance(self.metadata, dict):
            self.metadata = {}

        # Normalise primary metadata fields
        for key in ("title", "abstract", "pmid"):
            value = getattr(self, key)
            if value is None:
                meta_value = self.metadata.get(key)
                if isinstance(meta_value, str):
                    setattr(self, key, meta_value)
            else:
                self.metadata.setdefault(key, value)

        if not self.page_content and self.content:
            self.page_content = self.content
        elif not self.page_content:
            self.page_content = ""

        if not self.pk_parameters:
            pk_meta = self.metadata.get("pk_parameters")
            if isinstance(pk_meta, dict):
                self.pk_parameters = dict(pk_meta)
            else:
                self.pk_parameters = {}
        else:
            self.pk_parameters = dict(self.pk_parameters)

        self.metadata.setdefault("pk_parameters", dict(self.pk_parameters))

    @property
    def text(self) -> str:
        """Return the best available textual content for downstream analysis."""
        return self.page_content or self.content or ""

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation with guaranteed schema fields."""
        metadata = dict(self.metadata)
        metadata.setdefault("title", self.title)
        metadata.setdefault("abstract", self.abstract)
        metadata.setdefault("pmid", self.pmid)
        metadata["pk_parameters"] = dict(self.pk_parameters)

        result: dict[str, Any] = {
            "__paper_schema_validated__": True,
            "page_content": self.text,
            "content": self.content,
            "metadata": metadata,
        }

        extra = getattr(self, "model_extra", None)
        if extra:
            result.update(extra)

        if self.title is not None:
            result.setdefault("title", self.title)
        if self.abstract is not None:
            result.setdefault("abstract", self.abstract)
        if self.pmid is not None:
            result.setdefault("pmid", self.pmid)

        return result


def coerce_paper(entry: Paper | dict[str, Any]) -> Paper:
    """Return a `Paper` instance for the supplied entry."""
    if isinstance(entry, Paper):
        return entry
    if isinstance(entry, dict):
        return Paper.model_validate(entry)
    raise TypeError("Paper entries must be mappings or Paper models.")


def coerce_papers(entries: Iterable[Paper | dict[str, Any]]) -> list[Paper]:
    """Coerce a collection of paper-like entries into `Paper` instances."""
    return [coerce_paper(entry) for entry in entries]


def clean_identifier(identifier: str) -> str:
    """Clean DOI/PMID by stripping zero-width characters and decoding HTML entities.

    Args:
        identifier: Raw identifier string

    Returns:
        Cleaned identifier string
    """
    if not identifier:
        return identifier

    # Strip zero-width characters
    cleaned = identifier.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")

    # Decode common HTML entities
    cleaned = html.unescape(cleaned)

    return cleaned


def normalize_doi(doi: str) -> str:
    """Normalize DOI by removing prefixes and converting to lowercase.

    Args:
        doi: Raw DOI string

    Returns:
        Normalized DOI string
    """
    if not doi:
        return ""

    # Convert to lowercase and strip whitespace
    normalized = doi.lower().strip()

    # Strip common prefixes (case-insensitive, including trailing slash variants)
    prefixes_to_strip = [
        "doi:",
        "https://doi.org/",
        "https://doi.org",
        "http://doi.org/",
        "http://doi.org",
        "doi.org/",
        "doi.org",
        "www.doi.org/",
        "www.doi.org",
    ]

    for prefix in prefixes_to_strip:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            # Remove leading slash if it remains after prefix removal
            if normalized.startswith("/"):
                normalized = normalized[1:]
            break

    # Strip trailing punctuation
    normalized = normalized.rstrip(".,;)")

    return normalized.strip()


def normalize_pmid(pmid: str) -> str:
    """Normalize PMID by cleaning and ensuring numeric format.

    Args:
        pmid: Raw PMID string

    Returns:
        Normalized PMID string
    """
    if not pmid:
        return ""

    # Clean and strip whitespace
    normalized = clean_identifier(pmid).strip()

    # Remove any non-numeric characters except digits
    normalized = re.sub(r"[^\d]", "", normalized)

    return normalized


def extract_doi(text: str) -> str | None:
    """Extract and normalize DOI from text.

    Args:
        text: Text to search for DOI

    Returns:
        Normalized DOI if found, None otherwise
    """
    match = re.search(DOI_PATTERN, text, re.IGNORECASE)
    if match:
        raw_doi = match.group().replace("doi:", "").replace("DOI:", "").strip()
        clean_doi = clean_identifier(raw_doi.rstrip(".,;)"))
        return normalize_doi(clean_doi)

    # Check for DOI starting with 10. directly
    lines = text.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("10."):
            clean_doi = clean_identifier(stripped.rstrip(".,;)"))
            return normalize_doi(clean_doi)

    return None


def extract_pmid(text: str) -> str | None:
    """Extract and normalize PMID from text.

    Args:
        text: Text to search for PMID

    Returns:
        Normalized PMID if found, None otherwise
    """
    match = re.search(PMID_PATTERN_EXTRACT, text, re.IGNORECASE)
    if match:
        clean_pmid = clean_identifier(match.group(1))
        return normalize_pmid(clean_pmid)

    return None


def normalize_identifier(identifier: str, identifier_type: str) -> str:
    """Normalize identifier based on type.

    Args:
        identifier: Raw identifier string
        identifier_type: Type of identifier ('doi' or 'pmid')

    Returns:
        Normalized identifier string
    """
    if identifier_type.lower() == "doi":
        return normalize_doi(identifier)
    elif identifier_type.lower() == "pmid":
        return normalize_pmid(identifier)
    else:
        return clean_identifier(identifier)


def validate_doi(doi: str) -> bool:
    """Validate DOI format.

    Args:
        doi: DOI string to validate

    Returns:
        True if DOI is valid format, False otherwise
    """
    if not doi:
        return False

    normalized = normalize_doi(doi)
    return bool(re.match(r"^10\.\d{4,9}/.+", normalized))


def validate_pmid(pmid: str) -> bool:
    """Validate PMID format.

    Args:
        pmid: PMID string to validate

    Returns:
        True if PMID is valid format, False otherwise
    """
    if not pmid:
        return False

    normalized = normalize_pmid(pmid)
    return normalized.isdigit() and len(normalized) >= 1


__all__ = [
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
