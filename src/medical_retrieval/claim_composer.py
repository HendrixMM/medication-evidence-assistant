from __future__ import annotations

from collections import defaultdict
import re

from src.medical_retrieval.claim_filtering import is_patient_relevant_claim
from src.medical_retrieval.claim_models import ValidatedClaim


MEDICAL_DISCLAIMER = (
    "This information is from published research and is not medical advice.\n"
    "Please consult a healthcare provider for personal medical decisions."
)

_SECTION_TITLES = {
    "safety": "Safety",
    "efficacy": "Efficacy",
    "contraindication": "Contraindications",
    "interaction": "Interactions",
    "mechanism": "Mechanism",
    "dosing": "Dosing",
    "misc": "Evidence",
    "other": "Evidence",
}


def compose_answer_from_claims(validated_claims: list[ValidatedClaim]) -> str:
    supported = [claim for claim in validated_claims if claim.label == "supported" and claim.spans]
    if not supported:
        return ""

    grouped: dict[str, list[ValidatedClaim]] = defaultdict(list)
    for claim in supported:
        grouped[claim.claim.type].append(claim)

    sections: list[str] = []
    for claim_type in ("safety", "efficacy", "contraindication", "interaction", "mechanism", "dosing", "misc", "other"):
        claims = grouped.get(claim_type)
        if not claims:
            continue
        lines = [_SECTION_TITLES[claim_type]]
        for claim in claims:
            text = _claim_text(claim)
            citations = _citations(claim)
            lines.append(f"{text} {citations}".strip())
        sections.append("\n".join(lines))

    if not sections:
        return ""
    return "\n\n".join(sections + [MEDICAL_DISCLAIMER])


def compose_patient_medication_guide(validated_claims: list[ValidatedClaim], question: str) -> str:
    supported = _patient_supported_claims(validated_claims)
    if not supported:
        return ""

    used_ids: set[str] = set()
    sections: list[str] = [_intro(question, supported)]

    comparison_claims = _select_quick_comparison_claims(supported, question)
    if comparison_claims:
        sections.append(_section("Quick comparison", comparison_claims, question=question, medication_prefix=True))
        used_ids.update(_claim_key(claim) for claim in comparison_claims)

    combination_claims = [claim for claim in supported if _is_combination_claim(_claim_text(claim))]
    combination_claims = _unused_claims(combination_claims, used_ids)
    if combination_claims:
        sections.append(_section("Can you take them together?", combination_claims[:3], question=question))
        used_ids.update(_claim_key(claim) for claim in combination_claims[:3])

    all_better_claims = [claim for claim in supported if _is_comparative_claim(_claim_text(claim))]
    better_claims = _unused_claims(all_better_claims, used_ids)
    if _looks_like_comparison_question(question) or better_claims:
        better_section_claims = better_claims[:3] or all_better_claims[:2] or _unused_claims(supported, used_ids)[:2]
        if better_section_claims:
            sections.append(_section("Which is better?", better_section_claims, question=question))
            used_ids.update(_claim_key(claim) for claim in better_section_claims)

    sections.append(_section("Bottom line", supported[:2], question=question))
    sections.append(MEDICAL_DISCLAIMER)
    return "\n\n".join(section for section in sections if section.strip())


def _patient_supported_claims(validated_claims: list[ValidatedClaim]) -> list[ValidatedClaim]:
    supported = [
        claim
        for claim in validated_claims
        if claim.label == "supported" and claim.spans and is_patient_relevant_claim(claim)
    ]
    return sorted(supported, key=lambda claim: claim.claim.importance or 0, reverse=True)


def _select_quick_comparison_claims(
    supported: list[ValidatedClaim],
    question: str,
) -> list[ValidatedClaim]:
    if not _looks_like_comparison_question(question):
        return supported[:5]

    selected: list[ValidatedClaim] = []
    for medication in ("tylenol", "advil"):
        for claim in supported:
            if medication in _claim_text(claim).lower() and claim not in selected:
                selected.append(claim)
                break
    for claim_type in ("efficacy", "safety", "contraindication", "interaction", "dosing", "misc", "other"):
        for claim in supported:
            if claim.claim.type == claim_type and claim not in selected:
                selected.append(claim)
                break
    return selected[:5] or supported[:5]


def _intro(question: str, supported: list[ValidatedClaim]) -> str:
    if _mentions_both_common_brands(question):
        intro = "Tylenol and Advil are not the same medicine."
    else:
        intro = "This guide summarizes the medication evidence found for your question."
    if len(supported) <= 2:
        scope = (
            "The points below include only claims verified against PubMed abstract spans; "
            "topics without a bullet did not have enough retrieved support for this answer."
        )
    else:
        scope = "The points below include only claims verified against PubMed abstract spans."
    return f"{intro}\n{scope}"


def _section(
    title: str,
    claims: list[ValidatedClaim],
    *,
    question: str,
    medication_prefix: bool = False,
) -> str:
    lines = [title]
    for claim in claims:
        text = _claim_text(claim)
        if medication_prefix and _looks_like_comparison_question(question):
            text = _with_medication_prefix(text)
        citations = _citations(claim, max_citations=2)
        lines.append(f"- {text} {citations}".strip())
    return "\n".join(lines)


def _claim_text(claim: ValidatedClaim) -> str:
    return claim.display_text or claim.softened_text or claim.claim.display_text or claim.claim.text


def _claim_key(claim: ValidatedClaim) -> str:
    return claim.claim.id or _claim_text(claim)


def _unused_claims(claims: list[ValidatedClaim], used_ids: set[str]) -> list[ValidatedClaim]:
    return [claim for claim in claims if _claim_key(claim) not in used_ids]


def _looks_like_comparison_question(question: str) -> bool:
    return bool(re.search(r"\b(vs|versus|compare|comparison|better|which)\b", question, re.IGNORECASE))


def _mentions_both_common_brands(question: str) -> bool:
    lowered = question.lower()
    return "tylenol" in lowered and "advil" in lowered


def _is_comparative_claim(text: str) -> bool:
    return bool(
        re.search(
            r"\b(superior|inferior|better|preferred|compared|more effective|less effective|higher|more frequently|lower)\b",
            text,
            re.IGNORECASE,
        )
    )


def _is_combination_claim(text: str) -> bool:
    return bool(re.search(r"\b(combination|combined|together|same time|alternating|alternate)\b", text, re.IGNORECASE))


def _citations(claim: ValidatedClaim, *, max_citations: int | None = None) -> str:
    seen: set[str] = set()
    citations: list[str] = []
    for span in claim.spans:
        if span.pmid in seen:
            continue
        seen.add(span.pmid)
        citations.append(f"[PMID {span.pmid}]({span.pubmed_url})")
        if max_citations is not None and len(citations) >= max_citations:
            break
    if not citations:
        return ""
    return f"(Supported by {'; '.join(citations)})"


def _with_medication_prefix(text: str) -> str:
    lowered = text.lower()
    mentions_advil = "advil" in lowered or "ibuprofen" in lowered
    mentions_tylenol = "tylenol" in lowered or "acetaminophen" in lowered or "paracetamol" in lowered
    if mentions_advil and mentions_tylenol:
        return text
    if "advil" in lowered and not lowered.startswith("advil:"):
        return f"Advil: {text}"
    if "tylenol" in lowered and not lowered.startswith("tylenol:"):
        return f"Tylenol: {text}"
    return text


__all__ = ["MEDICAL_DISCLAIMER", "compose_answer_from_claims", "compose_patient_medication_guide"]
