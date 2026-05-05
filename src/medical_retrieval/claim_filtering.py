from __future__ import annotations

import re

from src.medical_retrieval.claim_models import Claim, ValidatedClaim


_METHOD_OR_META_PATTERNS = (
    re.compile(r"\ba literature search was conducted\b", re.IGNORECASE),
    re.compile(r"\bwe searched\b.*\b(pubmed|embase|medline|cochrane)\b", re.IGNORECASE),
    re.compile(r"\bsearch(?:es)? (?:were|was) (?:conducted|performed)\b", re.IGNORECASE),
    re.compile(r"\ba total of\b.*\b(records?|articles?|studies)\b.*\b(identified|screened|included)\b", re.IGNORECASE),
    re.compile(r"\b(records?|articles?|studies)\b.*\bwere (?:identified|screened|included|excluded)\b", re.IGNORECASE),
    re.compile(r"\b(randomi[sz]ed|double-blind|placebo-controlled|multicentre|multicenter)\b.*\btrial aims?\b", re.IGNORECASE),
    re.compile(r"\bclinical trial aims?\b|\bthis study (?:compared|aims?|was designed)\b", re.IGNORECASE),
    re.compile(r"^\s*study selection\s*:", re.IGNORECASE),
    re.compile(
        r"^\s*(?:materials and methods|interventions?|main outcome measures?|data extraction|data synthesis)\s*:",
        re.IGNORECASE,
    ),
    re.compile(r"\bmeta-analyses on the subset\b.*\breported sufficient quantitative information\b", re.IGNORECASE),
    re.compile(r"\b(?:children|patients|participants)\b.*\bwere included\b.*\b(randomi[sz]ed|controlled|trial)\b", re.IGNORECASE),
    re.compile(r"\bwas designed to assess\b|\bdesigned to compare\b", re.IGNORECASE),
    re.compile(r"\binclusion criteria\b|\bexclusion criteria\b|\bprisma\b", re.IGNORECASE),
    re.compile(r"\baccording to consumption data\b", re.IGNORECASE),
    re.compile(r"\bpackage counts?\b|\bsales stats?\b|\bsales data\b", re.IGNORECASE),
    re.compile(r"\bfrom \d{4} to \d{4}\b.*\b(use|usage|consumption|sales)\b", re.IGNORECASE),
)

_PATIENT_RELEVANT_TERMS = re.compile(
    r"\b("
    r"effective|efficacy|reduced?|increased?|relief|pain|fever|benefit|harm|risk|safe|safety|"
    r"side effects?|adverse|bleeding|toxicity|tolerability|tolerated|liver|kidney|contraindicat|avoid|interaction|"
    r"interacts?|dose|dosing|mg|superior|inferior|better|preferred|combination|combined|together|"
    r"alternating|treats?|prevents?"
    r")\b",
    re.IGNORECASE,
)


def is_methodological_or_meta_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in _METHOD_OR_META_PATTERNS)


def is_patient_relevant_text(text: str) -> bool:
    return bool(_PATIENT_RELEVANT_TERMS.search(text)) and not is_methodological_or_meta_text(text)


def is_patient_relevant_claim(claim: ValidatedClaim | Claim) -> bool:
    text = claim.claim.text if isinstance(claim, ValidatedClaim) else claim.text
    return is_patient_relevant_text(text)


__all__ = [
    "is_methodological_or_meta_text",
    "is_patient_relevant_claim",
    "is_patient_relevant_text",
]
