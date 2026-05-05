from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ClaimType = Literal[
    "efficacy",
    "safety",
    "contraindication",
    "interaction",
    "mechanism",
    "dosing",
    "misc",
    "other",
]
ClaimLabel = Literal["supported", "contradicted", "unclear"]


class Claim(BaseModel):
    id: str
    text: str
    type: ClaimType = "other"
    importance: int | None = None
    display_text: str | None = None
    evidence_terms: list[str] = Field(default_factory=list)


class SpanEvidence(BaseModel):
    pmid: str
    pubmed_url: str
    chunk_id: str
    span_text: str
    span_start: int
    span_end: int


class ValidatedClaim(BaseModel):
    claim: Claim
    label: ClaimLabel
    spans: list[SpanEvidence] = Field(default_factory=list)
    softened_text: str | None = None
    span_text: str | None = None
    display_text: str | None = None


__all__ = ["Claim", "ClaimLabel", "ClaimType", "SpanEvidence", "ValidatedClaim"]
