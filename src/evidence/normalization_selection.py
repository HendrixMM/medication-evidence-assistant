from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


DISFAVORED_TTYS = {"BPCK", "GPCK", "SCD", "SBD", "SCDC", "SBDC"}
PREFERRED_TTYS = {"IN", "PIN", "MIN", "BN"}
SUSPICIOUS_MARKERS = ("pack", "kit", "mg", "unt", "oral tablet", "oral capsule", "chewable tablet")


@dataclass(frozen=True)
class NormalizationCandidate:
    raw_name: str
    canonical_name: str
    match_type: str
    rxnorm_cui: str | None
    tty: str | None
    source: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.0

    def model_dump(self) -> dict[str, Any]:
        return {
            "raw_name": self.raw_name,
            "canonical_name": self.canonical_name,
            "match_type": self.match_type,
            "rxnorm_cui": self.rxnorm_cui,
            "tty": self.tty,
            "source": self.source,
            "aliases": list(self.aliases),
            "confidence": self.confidence,
        }


def _normalize_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.lower().strip()) if token}


def build_aliases(*values: str | None) -> tuple[str, ...]:
    aliases: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        cleaned = value.strip()
        lowered = cleaned.lower()
        if lowered and lowered not in seen:
            aliases.append(cleaned)
            seen.add(lowered)
        for match in re.findall(r"\[([^\]]+)\]", cleaned):
            bracketed = match.strip()
            lowered_bracketed = bracketed.lower()
            if lowered_bracketed and lowered_bracketed not in seen:
                aliases.append(bracketed)
                seen.add(lowered_bracketed)
    return tuple(aliases)


def _lexical_score(raw_name: str, candidate_name: str) -> float:
    raw = raw_name.lower().strip()
    candidate = candidate_name.lower().strip()
    if not raw or not candidate:
        return 0.0
    if raw == candidate:
        return 1.0
    if raw in candidate:
        return 0.55
    if candidate in raw:
        return 0.35
    overlap = len(_normalize_tokens(raw) & _normalize_tokens(candidate))
    return 0.15 * overlap


def _best_lexical_score(raw_name: str, candidate_name: str, aliases: tuple[str, ...] = ()) -> float:
    return max([_lexical_score(raw_name, candidate_name), *(_lexical_score(raw_name, alias) for alias in aliases)])


def score_candidate(
    raw_name: str,
    candidate_name: str,
    tty: str | None,
    source: str,
    aliases: tuple[str, ...] = (),
) -> float:
    candidate = candidate_name.lower().strip()
    lexical_score = _best_lexical_score(raw_name, candidate_name, aliases)
    score = lexical_score

    if tty in PREFERRED_TTYS:
        score += 0.2
    if tty in DISFAVORED_TTYS:
        score -= 0.35
    if any(marker in candidate for marker in SUSPICIOUS_MARKERS):
        score -= 0.25
    if "/" in candidate:
        score -= 0.25
    if source == "related":
        score += 0.15

    return max(0.0, min(score, 1.0))


def select_best_candidate(raw_name: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None

    scored: list[tuple[float, dict[str, Any]]] = []
    for item in candidates:
        name = str(item.get("name") or "")
        aliases = tuple(str(alias) for alias in item.get("aliases", ()) if alias)
        lexical_score = _best_lexical_score(raw_name, name, aliases)
        if str(item.get("source") or "") == "approximate" and lexical_score <= 0.0:
            continue
        score = score_candidate(
            raw_name,
            name,
            item.get("tty"),
            str(item.get("source") or ""),
            aliases,
        )
        enriched = dict(item)
        enriched["confidence"] = score
        scored.append((score, enriched))

    if not scored:
        return None

    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best = scored[0]
    if best_score < 0.45:
        return None
    return best
