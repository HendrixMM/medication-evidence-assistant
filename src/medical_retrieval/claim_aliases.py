from __future__ import annotations

import re


BRAND_ALIASES: dict[str, set[str]] = {
    "tylenol": {"acetaminophen", "paracetamol"},
    "acetaminophen": {"tylenol", "paracetamol"},
    "paracetamol": {"acetaminophen", "tylenol"},
    "advil": {"ibuprofen", "motrin"},
    "motrin": {"ibuprofen", "advil"},
    "ibuprofen": {"advil", "motrin"},
}

LAY_CLINICAL_ALIASES: dict[str, set[str]] = {
    "fever": {"febrile", "temperature", "antipyretic", "antipyretic effects", "antipyresis"},
    "febrile": {"fever", "temperature", "antipyretic", "antipyretic effects", "antipyresis"},
    "antipyretic": {"fever", "febrile", "temperature", "antipyretic effects", "antipyresis"},
    "antipyresis": {"fever", "febrile", "temperature", "antipyretic"},
    "pain": {"analgesic", "analgesia", "pain relief"},
    "analgesic": {"pain", "analgesia", "pain relief"},
    "analgesia": {"pain", "analgesic", "pain relief"},
    "inflammation": {"anti-inflammatory", "inflammatory pain"},
    "inflammatory": {"inflammation", "anti-inflammatory", "inflammatory pain"},
}

COMPARISON_USAGE_ALIASES: dict[str, set[str]] = {
    "better": {"comparative efficacy", "comparative safety", "superior"},
    "versus": {"comparative efficacy", "comparative safety", "compared"},
    "vs": {"comparative efficacy", "comparative safety", "compared"},
    "together": {"combination", "combined", "alternating"},
    "alternating": {"combination", "combined", "together"},
    "alternate": {"combination", "combined", "together"},
}

_STOP_TERMS = {
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "a",
    "an",
    "does",
    "do",
    "is",
    "are",
    "for",
    "with",
    "this",
    "that",
    "at",
    "be",
    "from",
    "by",
    "as",
}

_PARAPHRASE_FILLER = {
    "appear",
    "appears",
    "medicine",
    "medication",
    "standard",
    "bit",
    "slightly",
}

_OVERSTATED_TERMS = re.compile(
    r"\b(always|guaranteed|guarantees|best|safest|definitely|cures?|cured|no risk|risk-free)\b",
    re.IGNORECASE,
)


def content_terms(text: str) -> set[str]:
    terms = {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2 and word not in _STOP_TERMS}
    return terms | expanded_terms(terms)


def expanded_terms(terms: set[str]) -> set[str]:
    expanded: set[str] = set()
    for term in terms:
        expanded.update(BRAND_ALIASES.get(term, set()))
        expanded.update(LAY_CLINICAL_ALIASES.get(term, set()))
        expanded.update(COMPARISON_USAGE_ALIASES.get(term, set()))
    return expanded


def evidence_terms_for_text(*texts: str) -> list[str]:
    terms: set[str] = set()
    for text in texts:
        terms.update(content_terms(text))
    return sorted(terms)


def safe_paraphrase_supported_by_span(sentence: str, span_text: str) -> bool:
    if _OVERSTATED_TERMS.search(sentence):
        return False

    sentence_tokens = _canonical_match_tokens(sentence)
    span_tokens = _canonical_match_tokens(span_text)
    if not sentence_tokens or not span_tokens:
        return False

    missing = sentence_tokens - span_tokens
    if missing:
        return False

    if _medication_tokens(sentence_tokens) and not (_medication_tokens(sentence_tokens) & _medication_tokens(span_tokens)):
        return False
    if _clinical_tokens(sentence_tokens) and not (_clinical_tokens(sentence_tokens) & _clinical_tokens(span_tokens)):
        return False
    return True


def _canonical_match_tokens(text: str) -> set[str]:
    normalized = text.lower()
    replacements = (
        (r"\btylenol\b", " acetaminophen "),
        (r"\badvil\b|\bmotrin\b", " ibuprofen "),
        (r"\bfevers?\b|\bfebrile\b|\btemperature\b", " antipyretic "),
        (r"\bpain relief\b|\banalgesia\b", " analgesic "),
        (r"\binflammations?\b|\binflammatory\b", " anti inflammatory "),
        (r"\ba bit more effective\b|\bslightly more effective\b|\bmodestly more effective\b", " modestly superior "),
        (r"\bmore effective\b|\bmore efficacious\b", " superior "),
        (r"\bsimilarly effective\b|\bsimilar efficacy\b|\bwork similarly\b|\bworks similarly\b", " similar effect "),
        (r"\beffects?\b|\beffective\b|\befficacy\b|\befficacious\b", " effect "),
        (r"\bdoses?\b", " dose "),
        (r"\bover-the-counter\b", " over counter "),
        (r"\banti-inflammatory\b", " anti inflammatory "),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    raw_tokens = re.findall(r"[a-z0-9]+", normalized)
    return {
        token
        for token in raw_tokens
        if len(token) > 2 and token not in _STOP_TERMS and token not in _PARAPHRASE_FILLER
    }


def _medication_tokens(tokens: set[str]) -> set[str]:
    return tokens & {"acetaminophen", "paracetamol", "ibuprofen"}


def _clinical_tokens(tokens: set[str]) -> set[str]:
    return tokens & {"antipyretic", "analgesic", "inflammatory", "bleeding", "toxicity", "risk", "safety"}


__all__ = [
    "BRAND_ALIASES",
    "LAY_CLINICAL_ALIASES",
    "COMPARISON_USAGE_ALIASES",
    "content_terms",
    "evidence_terms_for_text",
    "expanded_terms",
    "safe_paraphrase_supported_by_span",
]
