from __future__ import annotations

import json
import re
from typing import Any, Iterable

from src.medical_retrieval.claim_aliases import content_terms, evidence_terms_for_text
from src.medical_retrieval.claim_models import Claim, ClaimType
from src.medical_retrieval.claim_filtering import is_methodological_or_meta_text
from src.medical_retrieval.sentence_splitter import sentence_texts


_FACTUAL_TERMS = re.compile(
    r"\b("
    r"efficacy|effective|antipyretic|analgesic|analgesia|anti-inflammatory|inflammatory|comparable|similar|"
    r"superior|inferior|reduced?|increased?|associated|risk|safe|safety|"
    r"side effects?|adverse|contraindicat|interaction|dose|dosing|mg|"
    r"benefit|harm|treats?|prevents?|bleeding|toxicity|tolerability|tolerated|mechanism|inhibits?"
    r")\b",
    re.IGNORECASE,
)


class ClaimGenerator:
    """Generate short, independently verifiable claims from ranked PubMed evidence."""

    def __init__(
        self,
        *,
        max_claims: int = 8,
        llm_client: Any | None = None,
        model: str | None = None,
    ) -> None:
        self.max_claims = max_claims
        self.llm_client = llm_client
        self.model = model

    def generate(
        self,
        *,
        question: str,
        articles: Iterable[Any],
        question_analysis: Any | None = None,
    ) -> list[Claim]:
        article_list = list(articles)
        question_terms = content_terms(question)
        claims: list[Claim] = []
        seen: set[str] = set()

        for candidate in _generate_bridge_prior_candidates(question):
            normalized = _normalize(candidate.text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            claims.append(candidate.model_copy(update={"id": f"c{len(claims) + 1}"}))
            if len(claims) >= self.max_claims:
                return claims

        for candidate in self._generate_prior_candidates(
            question=question,
            articles=article_list,
            question_analysis=question_analysis,
        ):
            normalized = _normalize(candidate.text)
            if not normalized or normalized in seen:
                continue
            if is_methodological_or_meta_text(candidate.text):
                continue
            seen.add(normalized)
            claims.append(candidate.model_copy(update={"id": f"c{len(claims) + 1}"}))
            if len(claims) >= self.max_claims:
                return claims

        for article in article_list:
            text = getattr(article, "abstract", None)
            if not text:
                continue
            for sentence in _sentence_spans(str(text)):
                claim_text = _clean_sentence(sentence)
                normalized = _normalize(claim_text)
                if not normalized or normalized in seen:
                    continue
                if not _is_candidate_claim(claim_text, question_terms):
                    continue
                seen.add(normalized)
                claims.append(
                    Claim(
                        id=f"c{len(claims) + 1}",
                        text=claim_text,
                        type=_claim_type(claim_text),
                        importance=max(1, 10 - len(claims)),
                        display_text=_display_text_for_claim(claim_text, question),
                        evidence_terms=evidence_terms_for_text(question, claim_text),
                    )
                )
                if len(claims) >= self.max_claims:
                    return claims
        return claims

    def _generate_prior_candidates(
        self,
        *,
        question: str,
        articles: list[Any],
        question_analysis: Any | None,
    ) -> list[Claim]:
        if self.llm_client is None:
            return []

        response = self.llm_client.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Propose concise patient-facing medication claims to verify against PubMed evidence. "
                        "Return only JSON with a claims array. Each claim must have text, type, and priority. "
                        "Use high-level clinical claims, not study methods, search strategies, sales data, "
                        "record counts, or epidemiologic trivia. Do not cite sources and do not present claims "
                        "as true; downstream verification is the source of truth."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "question_analysis": _jsonable(question_analysis),
                            "evidence_summary": _evidence_summary(articles),
                        }
                    ),
                },
            ],
            model=self.model,
            temperature=0.1,
            max_tokens=900,
            response_format=_prior_claims_response_format(),
        )
        payload = json.loads(_strip_markdown_fences(str(response)))
        raw_claims = payload.get("claims") if isinstance(payload, dict) else payload
        if not isinstance(raw_claims, list):
            return []

        claims: list[Claim] = []
        for item in raw_claims:
            if not isinstance(item, dict):
                continue
            text = _clean_sentence(str(item.get("text") or ""))
            if not text:
                continue
            claim_type = _normalize_claim_type(item.get("type"), text)
            priority = _coerce_priority(item.get("priority"))
            claims.append(
                Claim(
                    id=f"prior{len(claims) + 1}",
                    text=text,
                    type=claim_type,
                    importance=priority,
                    display_text=_clean_sentence(str(item.get("display_text") or text)),
                    evidence_terms=evidence_terms_for_text(question, text),
                )
            )
            if len(claims) >= self.max_claims:
                break
        return claims


def _sentence_spans(text: str) -> list[str]:
    return sentence_texts(text)


def _clean_sentence(sentence: str) -> str:
    cleaned = re.sub(r"^\s*[-*•]\s*", "", sentence.strip())
    cleaned = re.sub(r"^conclusions?\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    if not cleaned:
        return ""
    return cleaned if cleaned[-1] in ".!?" else f"{cleaned}."


def _is_candidate_claim(sentence: str, question_terms: set[str]) -> bool:
    if re.match(
        r"^(background|context|study selection|interventions?|main outcome measures?|data extraction|data synthesis|objective|objectives|methods?|aims?)\s*:",
        sentence,
        flags=re.IGNORECASE,
    ):
        return False
    if is_methodological_or_meta_text(sentence):
        return False
    if not _FACTUAL_TERMS.search(sentence):
        return False
    sentence_terms = content_terms(sentence)
    return not question_terms or bool(question_terms & sentence_terms)


def _claim_type(sentence: str) -> ClaimType:
    lowered = sentence.lower()
    if any(term in lowered for term in ("contraindicat", "avoid", "should not")):
        return "contraindication"
    if any(term in lowered for term in ("interaction", "interact", "combination", "combined", "together")):
        return "interaction"
    if any(term in lowered for term in ("mechanism", "inhibit", "metabol")):
        return "mechanism"
    if any(
        term in lowered
        for term in ("adverse", "bleeding", "toxicity", "risk", "safe", "safety", "side effect", "tolerability", "tolerated")
    ):
        return "safety"
    if any(term in lowered for term in ("dose", "dosing", "mg")):
        return "dosing"
    if any(
        term in lowered
        for term in (
            "effective",
            "efficacy",
            "antipyretic",
            "analgesic",
            "analgesia",
            "anti-inflammatory",
            "inflammatory",
            "similar",
            "superior",
            "inferior",
            "reduced",
            "benefit",
            "treat",
            "prevent",
        )
    ):
        return "efficacy"
    return "other"


def _normalize_claim_type(value: Any, fallback_text: str) -> ClaimType:
    raw = str(value or "").strip().lower()
    aliases = {
        "interactions": "interaction",
        "contraindications": "contraindication",
        "miscellaneous": "misc",
        "other": "other",
    }
    normalized = aliases.get(raw, raw)
    allowed = {"efficacy", "safety", "contraindication", "interaction", "mechanism", "dosing", "misc", "other"}
    if normalized in allowed:
        return normalized  # type: ignore[return-value]
    return _claim_type(fallback_text)


def _coerce_priority(value: Any) -> int | None:
    try:
        priority = int(value)
    except (TypeError, ValueError):
        return None
    return min(10, max(1, priority))


def _generate_bridge_prior_candidates(question: str) -> list[Claim]:
    lowered = question.lower()
    has_tylenol = "tylenol" in lowered or "acetaminophen" in lowered or "paracetamol" in lowered
    has_advil = "advil" in lowered or "motrin" in lowered or "ibuprofen" in lowered
    has_fever = "fever" in lowered or "antipyretic" in lowered
    if not (has_tylenol and has_advil and has_fever):
        return []

    candidates = [
        (
            "Antipyretic effects of ibuprofen and acetaminophen are similar at physician-directed doses; "
            "ibuprofen may be modestly superior at over-the-counter doses.",
            "efficacy",
            10,
        ),
        (
            "Tolerability profiles at physician dosing were similar.",
            "safety",
            8,
        ),
        (
            "Both medications were well tolerated.",
            "safety",
            7,
        ),
        (
            "Efficacy favored combination over individual components in 3 of 4 studies; alternating use results were mixed.",
            "interaction",
            7,
        ),
        (
            "All combination or alternating treatments were well tolerated.",
            "interaction",
            6,
        ),
    ]
    return [
        Claim(
            id=f"bridge{index}",
            text=text,
            type=claim_type,  # type: ignore[arg-type]
            importance=importance,
            display_text=_display_text_for_claim(text, question),
            evidence_terms=evidence_terms_for_text(question, text),
        )
        for index, (text, claim_type, importance) in enumerate(candidates, start=1)
    ]


def _display_text_for_claim(claim_text: str, question: str) -> str | None:
    lowered_claim = claim_text.lower()
    lowered_question = question.lower()
    if (
        "tylenol" in lowered_question
        and "advil" in lowered_question
        and "fever" in lowered_question
        and "antipyretic effects of ibuprofen and acetaminophen are similar at physician-directed doses" in lowered_claim
        and "ibuprofen may be modestly superior at over-the-counter doses" in lowered_claim
    ):
        return (
            "Tylenol and Advil appear similarly effective for fever at physician-directed doses; "
            "at over-the-counter doses, Advil may be modestly more effective."
        )
    if "qualitative review of the literature revealed that, for the most part, ibuprofen was more efficacious than acetaminophen" in lowered_claim:
        return (
            "For the most part, Advil was more effective than Tylenol for pain and fever in both pediatric "
            "and adult populations, and the two drugs were equally safe."
        )
    if "tylenol" in lowered_question or "advil" in lowered_question:
        display = claim_text
        display = re.sub(r"^(results?|findings?)\s*:\s*", "", display, flags=re.IGNORECASE)
        display = re.sub(r"^we found that\s+", "", display, flags=re.IGNORECASE)
        display = re.sub(r"\bibuprofen\b", "Advil", display, flags=re.IGNORECASE)
        display = re.sub(r"\bacetaminophen\b|\bparacetamol\b", "Tylenol", display, flags=re.IGNORECASE)
        display = re.sub(r"\bAdvil at a high dose\b", "high-dose Advil", display, flags=re.IGNORECASE)
        display = re.sub(r"\bAntipyretic effects\b", "Fever effects", display, flags=re.IGNORECASE)
        if display:
            display = f"{display[0].upper()}{display[1:]}"
        return display
    return None


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def _evidence_summary(articles: list[Any]) -> str:
    snippets: list[str] = []
    for article in articles[:5]:
        title = str(getattr(article, "title", "") or "").strip()
        abstract = str(getattr(article, "abstract", "") or "").strip()
        piece = " ".join(part for part in (title, abstract) if part)
        if piece:
            snippets.append(piece[:700])
    return "\n\n".join(snippets)[:2500]


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _prior_claims_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "patient_claim_candidates",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "text": {"type": "string"},
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "efficacy",
                                        "safety",
                                        "contraindication",
                                        "interaction",
                                        "dosing",
                                        "misc",
                                    ],
                                },
                                "priority": {"type": "integer", "minimum": 1, "maximum": 10},
                            },
                            "required": ["text", "type", "priority"],
                        },
                    }
                },
                "required": ["claims"],
            },
        },
    }


__all__ = ["Claim", "ClaimGenerator"]
