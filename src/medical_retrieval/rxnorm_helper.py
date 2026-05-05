from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, field_validator


class RxNormNormalizationRecord(BaseModel):
    raw_name: str
    canonical_name: str
    generic_name: str | None = None
    match_type: str | None = None
    rxnorm_cui: str | None = None
    confidence: float | None = None

    @field_validator("raw_name", "canonical_name")
    @classmethod
    def required_names_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("RxNorm names must not be empty")
        return value


class RxNormHelper:
    """Optional adapter around an RxNorm resolver for medication and supplement mentions."""

    def __init__(
        self,
        resolver: Any | None = None,
        resolver_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.resolver = resolver
        self.resolver_factory = resolver_factory or _build_default_resolver

    def normalize_mentions(self, mentions: list[Any]) -> list[RxNormNormalizationRecord]:
        names = [_extract_mention_name(mention) for mention in mentions]
        names = [name for name in names if name]
        if not names:
            return []

        resolver = self.resolver or self.resolver_factory()
        candidates = resolver.resolve_names_with_context(names)
        return [self._map_candidate(candidate) for candidate in candidates]

    def _map_candidate(self, candidate: Any) -> RxNormNormalizationRecord:
        raw_name = str(_get_value(candidate, "raw_name") or "")
        generic_name = _get_value(candidate, "generic_name")
        canonical_name = str(_get_value(candidate, "canonical_name") or generic_name or raw_name)
        confidence = _get_value(candidate, "confidence")
        return RxNormNormalizationRecord(
            raw_name=raw_name,
            canonical_name=canonical_name,
            generic_name=str(generic_name) if generic_name is not None else None,
            match_type=_get_value(candidate, "match_type"),
            rxnorm_cui=_get_value(candidate, "rxnorm_cui"),
            confidence=float(confidence) if confidence is not None else None,
        )


def _extract_mention_name(mention: Any) -> str:
    if isinstance(mention, str):
        return mention.strip()
    value = _get_value(mention, "raw_name") or _get_value(mention, "name")
    if value is None:
        return ""
    return str(value).strip()


def _get_value(record: Any, key: str) -> Any:
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key, None)


def _build_default_resolver() -> Any:
    from src.evidence.rxnorm_resolver import RxNormResolver

    return RxNormResolver()
