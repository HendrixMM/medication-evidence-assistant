from __future__ import annotations

from src.medical_retrieval.claim_generation import Claim
from src.medical_retrieval.claim_traceability import ClaimTraceabilityGate
from src.medical_retrieval.schemas import MedicalArticleRecord


def _article(*, abstract: str, pmid: str = "123") -> MedicalArticleRecord:
    return MedicalArticleRecord(
        pmid=pmid,
        title="Acetaminophen and ibuprofen evidence",
        abstract=abstract,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )


def test_sentence_with_exact_span_support_is_allowed() -> None:
    gate = ClaimTraceabilityGate()

    result = gate.validate(
        "Acetaminophen and ibuprofen produced similar analgesia.",
        [_article(abstract="Acetaminophen and ibuprofen produced similar analgesia.")],
    )

    assert result.publishable is True
    assert result.answer_text == (
        "Acetaminophen and ibuprofen produced similar analgesia. "
        "[PMID 123](https://pubmed.ncbi.nlm.nih.gov/123/)"
    )
    assert result.sentences[0].supported is True
    assert result.sentences[0].span_text == "Acetaminophen and ibuprofen produced similar analgesia."


def test_document_level_relevance_without_span_support_is_rejected() -> None:
    gate = ClaimTraceabilityGate()

    result = gate.validate(
        "Acetaminophen and ibuprofen are safe to combine.",
        [_article(abstract="This review discusses acetaminophen and ibuprofen use in children.")],
    )

    assert result.publishable is False
    assert result.answer_text == ""
    assert result.sentences[0].supported is False
    assert result.sentences[0].verifier_decision == "no_supporting_span"


def test_multi_claim_sentence_splits_and_only_supported_portion_survives() -> None:
    gate = ClaimTraceabilityGate()

    result = gate.validate(
        "Ibuprofen reduced fever and acetaminophen prevents liver injury.",
        [_article(abstract="Ibuprofen reduced fever.")],
    )

    assert result.publishable is True
    assert result.answer_text == "Ibuprofen reduced fever. [PMID 123](https://pubmed.ncbi.nlm.nih.gov/123/)"
    assert [record.supported for record in result.sentences] == [True, False]
    assert result.sentences[1].sentence_text == "acetaminophen prevents liver injury."


def test_traceability_does_not_split_compound_side_effect_noun_phrase() -> None:
    gate = ClaimTraceabilityGate()

    result = gate.validate(
        "- Advil: The risk of gastrointestinal and renal side effects is significantly higher with Advil.",
        [
            _article(
                abstract="The risk of gastrointestinal and renal side effects is significantly higher with ibuprofen.",
                pmid="34503427",
            )
        ],
    )

    assert result.publishable is True
    assert len([record for record in result.sentences if record.sentence_type == "factual"]) == 1
    assert result.answer_text == (
        "- Advil: The risk of gastrointestinal and renal side effects is significantly higher with Advil. "
        "(Supported by [PMID 34503427](https://pubmed.ncbi.nlm.nih.gov/34503427/))"
    )


def test_broad_medication_safety_claim_not_found_in_span_is_blocked() -> None:
    gate = ClaimTraceabilityGate()

    result = gate.validate(
        "Tylenol and Advil can be taken together safely if dosing limits are followed.",
        [_article(abstract="The trial compared acetaminophen and ibuprofen for fever reduction.")],
    )

    assert result.publishable is False
    assert "safely" in result.sentences[0].sentence_text
    assert result.sentences[0].supported is False


def test_fully_supported_answer_publishes_with_per_sentence_citations() -> None:
    gate = ClaimTraceabilityGate()

    result = gate.validate(
        "Ibuprofen reduced fever. Acetaminophen reduced fever.",
        [
            _article(
                abstract="Ibuprofen reduced fever. Acetaminophen reduced fever.",
            )
        ],
    )

    assert result.publishable is True
    assert result.answer_text == (
        "Ibuprofen reduced fever. [PMID 123](https://pubmed.ncbi.nlm.nih.gov/123/)\n"
        "Acetaminophen reduced fever. [PMID 123](https://pubmed.ncbi.nlm.nih.gov/123/)"
    )
    assert all(record.supported for record in result.sentences)


def test_already_cited_answer_is_revalidated_without_markdown_url_fragments() -> None:
    gate = ClaimTraceabilityGate()

    result = gate.validate(
        (
            "Low doses of ASA increased the risk of gastrointestinal bleeding. "
            "[PMID 21699808](https://pubmed.ncbi.nlm.nih.gov/21699808/)"
        ),
        [
            _article(
                abstract="Low doses of ASA increased the risk of gastrointestinal bleeding.",
                pmid="21699808",
            )
        ],
    )

    assert result.publishable is True
    assert result.answer_text == (
        "Low doses of ASA increased the risk of gastrointestinal bleeding. "
        "[PMID 21699808](https://pubmed.ncbi.nlm.nih.gov/21699808/)"
    )
    assert "\n" not in result.answer_text


def test_traceability_does_not_split_statistical_decimal_values() -> None:
    gate = ClaimTraceabilityGate()

    result = gate.validate(
        "Aspirin increased the risk of major bleeding (RR=1.71, 95% CI 1.41-2.08).",
        [
            _article(
                abstract="RESULTS: Aspirin increased the risk of major bleeding (RR=1.71, 95% CI 1.41-2.08).",
                pmid="16887404",
            )
        ],
    )

    assert result.publishable is True
    assert result.answer_text == (
        "Aspirin increased the risk of major bleeding (RR=1.71, 95% CI 1.41-2.08). "
        "[PMID 16887404](https://pubmed.ncbi.nlm.nih.gov/16887404/)"
    )
    assert len(result.sentences) == 1
    assert result.sentences[0].span_text == (
        "RESULTS: Aspirin increased the risk of major bleeding (RR=1.71, 95% CI 1.41-2.08)."
    )


def test_traceability_does_not_cite_section_headings() -> None:
    gate = ClaimTraceabilityGate()

    result = gate.validate(
        "Safety\nLow doses of ASA increased the risk of gastrointestinal bleeding.",
        [
            _article(
                abstract=(
                    "Safety outcomes were reviewed. "
                    "Low doses of ASA increased the risk of gastrointestinal bleeding."
                ),
                pmid="21699808",
            )
        ],
    )

    assert result.publishable is True
    assert result.answer_text == (
        "Safety\n"
        "Low doses of ASA increased the risk of gastrointestinal bleeding. "
        "[PMID 21699808](https://pubmed.ncbi.nlm.nih.gov/21699808/)"
    )
    assert result.sentences[0].sentence_type == "non_factual"
    assert result.sentences[0].pmid is None


def test_claim_list_validation_labels_supported_claim_with_exact_span() -> None:
    gate = ClaimTraceabilityGate()

    result = gate.validate_claims(
        [
            Claim(
                id="c1",
                text="Low doses of ASA increased the risk of gastrointestinal bleeding.",
                type="safety",
            )
        ],
        [
            _article(
                abstract="Low doses of ASA increased the risk of gastrointestinal bleeding.",
                pmid="21699808",
            )
        ],
    )

    assert result.publishable is True
    assert result.validated_claims[0].label == "supported"
    assert result.validated_claims[0].spans[0].pmid == "21699808"
    assert result.validated_claims[0].spans[0].span_text == (
        "Low doses of ASA increased the risk of gastrointestinal bleeding."
    )


def test_claim_list_validation_preserves_span_and_patient_display_text_for_safe_paraphrase() -> None:
    span_text = (
        "Antipyretic effects of ibuprofen and acetaminophen are similar at physician-directed doses; "
        "ibuprofen may be modestly superior at over-the-counter doses."
    )
    display_text = (
        "Tylenol and Advil appear similarly effective for fever at physician-directed doses; "
        "at over-the-counter doses, Advil may be modestly more effective."
    )
    gate = ClaimTraceabilityGate()

    result = gate.validate_claims(
        [
            Claim(
                id="c1",
                text=span_text,
                type="efficacy",
                display_text=display_text,
            )
        ],
        [_article(abstract=span_text, pmid="33966545")],
    )

    assert result.validated_claims[0].label == "supported"
    assert result.validated_claims[0].span_text == span_text
    assert result.validated_claims[0].display_text == display_text


def test_traceability_allows_conservative_brand_lay_paraphrase_against_exact_span() -> None:
    span_text = (
        "Antipyretic effects of ibuprofen and acetaminophen are similar at physician-directed doses; "
        "ibuprofen may be modestly superior at over-the-counter doses."
    )
    display_text = (
        "Tylenol and Advil appear similarly effective for fever at physician-directed doses; "
        "at over-the-counter doses, Advil may be modestly more effective."
    )
    gate = ClaimTraceabilityGate()

    result = gate.validate(display_text, [_article(abstract=span_text, pmid="33966545")])

    assert result.publishable is True
    assert result.sentences[0].supported is True
    assert result.sentences[0].span_text == span_text


def test_traceability_rejects_overstated_brand_lay_paraphrase() -> None:
    span_text = (
        "Antipyretic effects of ibuprofen and acetaminophen are similar at physician-directed doses; "
        "ibuprofen may be modestly superior at over-the-counter doses."
    )
    gate = ClaimTraceabilityGate()

    result = gate.validate(
        "Advil is always the best fever medicine.",
        [_article(abstract=span_text, pmid="33966545")],
    )

    assert result.publishable is False
    assert result.sentences[0].supported is False


def test_traceability_keeps_supported_by_citation_style_for_bullets() -> None:
    gate = ClaimTraceabilityGate()

    result = gate.validate(
        "- Advil reduced temperature more than Tylenol at 2, 4, and 6 hours after treatment. "
        "(Supported by [PMID 15184213](https://pubmed.ncbi.nlm.nih.gov/15184213/))",
        [
            _article(
                abstract="Ibuprofen reduced temperature more than acetaminophen at 2, 4, and 6 hours after treatment.",
                pmid="15184213",
            )
        ],
    )

    assert result.publishable is True
    assert "(Supported by [PMID 15184213](https://pubmed.ncbi.nlm.nih.gov/15184213/))" in result.answer_text


def test_claim_list_validation_labels_contradicted_and_unclear_claims() -> None:
    gate = ClaimTraceabilityGate()

    result = gate.validate_claims(
        [
            Claim(id="c1", text="Aspirin increased bleeding.", type="safety"),
            Claim(id="c2", text="Aspirin improved sleep quality.", type="efficacy"),
        ],
        [
            _article(
                abstract="Aspirin did not increase bleeding in this trial.",
                pmid="1",
            )
        ],
    )

    labels = {claim.claim.id: claim.label for claim in result.validated_claims}
    assert labels == {"c1": "contradicted", "c2": "unclear"}
    assert result.publishable is False
