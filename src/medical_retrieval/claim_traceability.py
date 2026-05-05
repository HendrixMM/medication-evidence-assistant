from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from src.medical_retrieval.claim_aliases import safe_paraphrase_supported_by_span
from src.medical_retrieval.claim_models import Claim, ClaimLabel, SpanEvidence, ValidatedClaim
from src.medical_retrieval.schemas import MedicalArticleRecord
from src.medical_retrieval.sentence_splitter import sentence_spans as split_sentence_spans


SentenceType = Literal["factual", "non_factual"]


class ClaimTraceRecord(BaseModel):
    sentence_text: str
    sentence_type: SentenceType
    supported: bool
    pmid: str | None = None
    pubmed_url: str | None = None
    chunk_id: str | None = None
    span_text: str | None = None
    span_start: int | None = None
    span_end: int | None = None
    verifier_decision: str
    rewrite_required: bool = False


class ClaimTraceabilityResult(BaseModel):
    publishable: bool
    answer_text: str
    sentences: list[ClaimTraceRecord]
    validated_claims: list[ValidatedClaim] = Field(default_factory=list)


class ClaimTraceabilityGate:
    """Validate that factual answer sentences are traceable to retrieved PubMed spans."""

    def validate(
        self,
        answer_text: str | list[Claim],
        articles: list[MedicalArticleRecord],
    ) -> ClaimTraceabilityResult:
        if isinstance(answer_text, list):
            return self.validate_claims(answer_text, articles)

        source_spans = _source_spans(articles)
        records: list[ClaimTraceRecord] = []
        kept_sentences: list[str] = []

        for sentence in _atomic_sentences(answer_text):
            if not _is_factual_sentence(sentence):
                records.append(
                    ClaimTraceRecord(
                        sentence_text=sentence,
                        sentence_type="non_factual",
                        supported=True,
                        verifier_decision="non_factual",
                    )
                )
                kept_sentences.append(sentence)
                continue

            support = _find_support(sentence, source_spans)
            if support is None:
                records.append(
                    ClaimTraceRecord(
                        sentence_text=sentence,
                        sentence_type="factual",
                        supported=False,
                        verifier_decision="no_supporting_span",
                        rewrite_required=True,
                    )
                )
                continue

            record = ClaimTraceRecord(
                sentence_text=sentence,
                sentence_type="factual",
                supported=True,
                pmid=support["pmid"],
                pubmed_url=support["pubmed_url"],
                chunk_id=support["chunk_id"],
                span_text=support["span_text"],
                span_start=support["span_start"],
                span_end=support["span_end"],
                verifier_decision="supported_by_exact_span",
            )
            records.append(record)
            kept_sentences.append(_with_citation(sentence, record))

        supported_factual = sum(1 for record in records if record.sentence_type == "factual" and record.supported)
        unsupported_factual = sum(
            1 for record in records if record.sentence_type == "factual" and not record.supported
        )
        publishable = supported_factual > 0 and unsupported_factual <= supported_factual
        if not publishable:
            kept_sentences = []

        return ClaimTraceabilityResult(
            publishable=publishable,
            answer_text="\n".join(kept_sentences),
            sentences=records,
        )

    def validate_claims(
        self,
        claims: list[Claim],
        articles: list[MedicalArticleRecord],
    ) -> ClaimTraceabilityResult:
        source_spans = _source_spans(articles)
        validated: list[ValidatedClaim] = []
        records: list[ClaimTraceRecord] = []
        kept_sentences: list[str] = []

        for claim in claims:
            support = _find_support(claim.text, source_spans)
            contradiction = None if support is not None else _find_contradiction(claim.text, source_spans)
            if support is not None:
                span = _span_evidence(support)
                display_text = _safe_display_text(claim, span.span_text)
                validated_claim = ValidatedClaim(
                    claim=claim,
                    label="supported",
                    spans=[span],
                    span_text=span.span_text,
                    display_text=display_text,
                )
                validated.append(validated_claim)
                record = ClaimTraceRecord(
                    sentence_text=claim.text,
                    sentence_type="factual",
                    supported=True,
                    pmid=span.pmid,
                    pubmed_url=span.pubmed_url,
                    chunk_id=span.chunk_id,
                    span_text=span.span_text,
                    span_start=span.span_start,
                    span_end=span.span_end,
                    verifier_decision="supported_by_exact_span",
                )
                records.append(record)
                kept_sentences.append(_with_citation(claim.text, record))
                continue

            if contradiction is not None:
                validated.append(
                    ValidatedClaim(claim=claim, label="contradicted", spans=[_span_evidence(contradiction)])
                )
                records.append(
                    ClaimTraceRecord(
                        sentence_text=claim.text,
                        sentence_type="factual",
                        supported=False,
                        verifier_decision="contradicted_by_exact_span",
                        rewrite_required=True,
                    )
                )
                continue

            validated.append(ValidatedClaim(claim=claim, label="unclear", spans=[]))
            records.append(
                ClaimTraceRecord(
                    sentence_text=claim.text,
                    sentence_type="factual",
                    supported=False,
                    verifier_decision="no_supporting_span",
                    rewrite_required=True,
                )
            )

        return ClaimTraceabilityResult(
            publishable=any(claim.label == "supported" for claim in validated),
            answer_text="\n".join(kept_sentences),
            sentences=records,
            validated_claims=validated,
        )


def _source_spans(articles: list[MedicalArticleRecord]) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    for article in articles:
        pmid = (article.pmid or "").strip()
        url = (article.url or "").strip()
        if not pmid or "pubmed.ncbi.nlm.nih.gov" not in url.lower():
            continue
        for chunk_id, text in (("title", article.title), ("abstract", article.abstract)):
            if not text:
                continue
            for index, sentence in enumerate(_sentence_spans(text)):
                span_text, start, end = sentence
                spans.append(
                    {
                        "pmid": pmid,
                        "pubmed_url": url,
                        "chunk_id": f"{chunk_id}:{index}",
                        "span_text": span_text,
                        "span_start": start,
                        "span_end": end,
                        "normalized": _normalize(span_text),
                    }
                )
    return spans


def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    return split_sentence_spans(text)


def _atomic_sentences(text: str) -> list[str]:
    text = _strip_citations(text)
    sentences: list[str] = []
    for raw in _sentence_spans(text):
        sentence = _clean_sentence(raw[0])
        if not sentence:
            continue
        sentences.extend(_split_multi_claim(sentence))
    return sentences


def _split_multi_claim(sentence: str) -> list[str]:
    parts = re.split(r"\s+(?:and|but|whereas|while)\s+", sentence, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return [sentence]
    left, right = (_ensure_terminal(parts[0].strip()), _ensure_terminal(parts[1].strip()))
    if _has_predicate(left) and _has_predicate(right) and _has_explicit_subject(left) and _has_explicit_subject(right):
        return [left, right]
    return [sentence]


def _clean_sentence(sentence: str) -> str:
    cleaned = sentence.strip()
    if not cleaned:
        return ""
    if _is_section_heading(cleaned):
        return cleaned.rstrip(".!?")
    return _ensure_terminal(cleaned)


def _ensure_terminal(sentence: str) -> str:
    if not sentence:
        return sentence
    return sentence if sentence[-1] in ".!?" else f"{sentence}."


def _is_section_heading(sentence: str) -> bool:
    heading = re.sub(r"[^a-z]+", " ", _strip_citations(sentence).lower()).strip()
    return heading in _SECTION_HEADINGS


_FACTUAL_TERMS = re.compile(
    r"\b("
    r"efficacy|effective|reduced?|increased?|associated|caused?|causes|cause|risk|safe|safety|"
    r"side effects?|adverse|contraindicat|interaction|interacts?|dose|dosing|mg|superior|inferior|"
    r"benefit|harm|treats?|prevents?|analgesia|fever|pain|bleeding|toxicity|liver|kidney|"
    r"mechanism|inhibits?|improves?|worsens?|compared|combination|substitution|switch"
    r")\b",
    re.IGNORECASE,
)

_NON_FACTUAL_PHRASES = (
    "not medical advice",
    "consult a healthcare provider",
    "consult your healthcare provider",
    "contact a clinician",
    "contact a pharmacist",
    "seek medical care",
)

_SECTION_HEADINGS = {
    "safety",
    "efficacy",
    "contraindication",
    "contraindications",
    "mechanism",
    "dosing",
    "evidence",
}


def _is_factual_sentence(sentence: str) -> bool:
    if _is_section_heading(sentence):
        return False
    lowered = sentence.lower()
    if any(phrase in lowered for phrase in _NON_FACTUAL_PHRASES):
        return False
    return bool(_FACTUAL_TERMS.search(sentence))


def _has_predicate(sentence: str) -> bool:
    return bool(_FACTUAL_TERMS.search(sentence))


def _has_explicit_subject(sentence: str) -> bool:
    return bool(
        re.match(
            r"^(?:-?\s*)?(?:acetaminophen|ibuprofen|aspirin|tylenol|advil|asa|patients?|children|adults?|participants|it|they|this|that)\b",
            sentence,
            flags=re.IGNORECASE,
        )
    )


def _find_support(sentence: str, spans: list[dict[str, object]]) -> dict[str, object] | None:
    normalized_sentence = _normalize(_strip_citations(sentence))
    if not normalized_sentence:
        return None
    for span in spans:
        normalized_span = str(span["normalized"])
        if normalized_sentence == normalized_span or normalized_sentence in normalized_span:
            return span
        if safe_paraphrase_supported_by_span(sentence, str(span["span_text"])):
            return span
    return None


def _safe_display_text(claim: Claim, span_text: str) -> str | None:
    display_text = (claim.display_text or "").strip()
    if not display_text or display_text == claim.text:
        return None
    if safe_paraphrase_supported_by_span(display_text, span_text):
        return display_text
    return None


def _find_contradiction(sentence: str, spans: list[dict[str, object]]) -> dict[str, object] | None:
    normalized_sentence = _normalize(_strip_citations(sentence))
    if not normalized_sentence:
        return None
    sentence_base = _canonical_for_contradiction(normalized_sentence)
    for span in spans:
        normalized_span = str(span["normalized"])
        if not _contains_negation(normalized_span):
            continue
        span_base = _canonical_for_contradiction(normalized_span)
        if _token_overlap(sentence_base, span_base) >= 0.6:
            return span
    return None


def _span_evidence(span: dict[str, object]) -> SpanEvidence:
    return SpanEvidence(
        pmid=str(span["pmid"]),
        pubmed_url=str(span["pubmed_url"]),
        chunk_id=str(span["chunk_id"]),
        span_text=str(span["span_text"]),
        span_start=int(span["span_start"]),
        span_end=int(span["span_end"]),
    )


def _strip_citations(sentence: str) -> str:
    sentence = re.sub(
        r"\(\s*Supported by\s+\[PMID\s+\d+\]\([^)]+\)(?:\s*;\s*\[PMID\s+\d+\]\([^)]+\))*\s*\)",
        "",
        sentence,
        flags=re.IGNORECASE,
    )
    sentence = re.sub(r"\[PMID\s+\d+\]\([^)]+\)", "", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"https://pubmed\.ncbi\.nlm\.nih\.gov/\d+/?", "", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bPMID\s*:?\s*\d+\b", "", sentence, flags=re.IGNORECASE)
    return sentence.strip()


def _with_citation(sentence: str, record: ClaimTraceRecord) -> str:
    citation = f"[PMID {record.pmid}]({record.pubmed_url})"
    stripped = _strip_citations(sentence).strip()
    if stripped.startswith("- "):
        return f"{stripped} (Supported by {citation})"
    return f"{stripped} {citation}"


def _normalize(text: str) -> str:
    normalized = _strip_citations(text).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _contains_negation(text: str) -> bool:
    return bool(re.search(r"\b(no|not|did not|does not|without|neither)\b", text, flags=re.IGNORECASE))


def _canonical_for_contradiction(text: str) -> set[str]:
    text = re.sub(r"\b(did|does|do|not|no|without|neither|this|trial|study|in)\b", " ", text)
    replacements = {
        "increased": "increase",
        "increases": "increase",
        "reduced": "reduce",
        "reduces": "reduce",
    }
    return {replacements.get(token, token) for token in text.split() if len(token) > 2}


def _token_overlap(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


__all__ = [
    "ClaimTraceRecord",
    "ClaimTraceabilityGate",
    "ClaimTraceabilityResult",
    "SpanEvidence",
    "ValidatedClaim",
]
