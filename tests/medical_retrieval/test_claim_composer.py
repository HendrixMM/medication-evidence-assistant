from __future__ import annotations

from src.medical_retrieval.claim_composer import (
    compose_answer_from_claims,
    compose_patient_medication_guide,
)
from src.medical_retrieval.claim_generation import Claim
from src.medical_retrieval.claim_traceability import ClaimTraceabilityGate, SpanEvidence, ValidatedClaim
from src.medical_retrieval.schemas import MedicalArticleRecord


def _span(pmid: str, text: str) -> SpanEvidence:
    return SpanEvidence(
        pmid=pmid,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        chunk_id="abstract:0",
        span_text=text,
        span_start=0,
        span_end=len(text),
    )


def test_compose_answer_from_claims_keeps_only_supported_claims_with_citations() -> None:
    supported_text = "Low doses of ASA increased the risk of gastrointestinal bleeding."
    unclear_text = "Aspirin improved sleep quality."
    answer = compose_answer_from_claims(
        [
            ValidatedClaim(
                claim=Claim(id="c1", text=supported_text, type="safety"),
                label="supported",
                spans=[_span("21699808", supported_text)],
            ),
            ValidatedClaim(
                claim=Claim(id="c2", text=unclear_text, type="efficacy"),
                label="unclear",
                spans=[],
            ),
        ]
    )

    assert supported_text in answer
    assert unclear_text not in answer
    assert "[PMID 21699808](https://pubmed.ncbi.nlm.nih.gov/21699808/)" in answer
    assert "This information is from published research and is not medical advice." in answer


def test_compose_answer_from_claims_returns_empty_when_no_supported_claims() -> None:
    answer = compose_answer_from_claims(
        [
            ValidatedClaim(
                claim=Claim(id="c1", text="Aspirin improved sleep quality.", type="efficacy"),
                label="unclear",
                spans=[],
            )
        ]
    )

    assert answer == ""


def test_patient_medication_guide_uses_sections_bullets_and_filters_methods() -> None:
    tylenol_text = "Acetaminophen reduced fever and pain in children."
    advil_text = "Ibuprofen reduced fever and pain in children."
    together_text = "Combined acetaminophen and ibuprofen increased fever clearance."
    methods_text = "A literature search was conducted in PubMed and Embase."
    trial_aims_text = (
        "This randomized, double-blind and placebo-controlled clinical trial aims at comparing the "
        "antipyretic effectiveness and safety of a single administration of alternating ibuprofen and "
        "acetaminophen doses to that of ibuprofen mono-therapy in febrile children."
    )
    usage_text = "According to consumption data from Italy between 2019 and 2024, pediatric ibuprofen use grew by over 60%."

    answer = compose_patient_medication_guide(
        [
            ValidatedClaim(
                claim=Claim(id="c1", text=tylenol_text, type="efficacy", importance=9),
                label="supported",
                spans=[_span("111", tylenol_text)],
            ),
            ValidatedClaim(
                claim=Claim(id="c2", text=advil_text, type="efficacy", importance=8),
                label="supported",
                spans=[_span("222", advil_text)],
            ),
            ValidatedClaim(
                claim=Claim(id="c3", text=together_text, type="efficacy", importance=7),
                label="supported",
                spans=[_span("333", together_text)],
            ),
            ValidatedClaim(
                claim=Claim(id="c4", text=methods_text, type="other", importance=10),
                label="supported",
                spans=[_span("444", methods_text)],
            ),
            ValidatedClaim(
                claim=Claim(id="c5", text=trial_aims_text, type="safety", importance=10),
                label="supported",
                spans=[_span("445", trial_aims_text)],
            ),
            ValidatedClaim(
                claim=Claim(id="c6", text=usage_text, type="other", importance=10),
                label="supported",
                spans=[_span("555", usage_text)],
            ),
        ],
        question="Tylenol vs Advil for fever",
    )

    assert "Quick comparison" in answer
    assert "Which is better?" in answer
    assert "Can you take them together?" in answer
    assert "Bottom line" in answer
    assert "- " in answer
    assert tylenol_text in answer
    assert advil_text in answer
    assert together_text in answer
    assert "A literature search was conducted" not in answer
    assert "trial aims at comparing" not in answer
    assert "consumption data from Italy" not in answer


def test_patient_medication_guide_uses_display_text_instead_of_raw_span_text() -> None:
    span_text = (
        "Antipyretic effects of ibuprofen and acetaminophen are similar at physician-directed doses; "
        "ibuprofen may be modestly superior at over-the-counter doses."
    )
    display_text = (
        "Tylenol and Advil appear similarly effective for fever at physician-directed doses; "
        "at over-the-counter doses, Advil may be modestly more effective."
    )

    answer = compose_patient_medication_guide(
        [
            ValidatedClaim(
                claim=Claim(id="c1", text=span_text, type="efficacy", importance=9),
                label="supported",
                spans=[_span("33966545", span_text)],
                span_text=span_text,
                display_text=display_text,
            )
        ],
        question="Tylenol vs Advil for fever",
    )

    assert display_text in answer
    assert span_text not in answer
    assert "- " in answer


def test_patient_medication_guide_uses_patient_structure_and_supported_by_citations() -> None:
    stomach_span = "The risk of gastrointestinal and renal side effects is significantly higher with ibuprofen."
    liver_span = "Hepatobiliary side effects are more frequently linked to acetaminophen."

    answer = compose_patient_medication_guide(
        [
            ValidatedClaim(
                claim=Claim(id="c1", text=stomach_span, type="safety", importance=9),
                label="supported",
                spans=[_span("34503427", stomach_span)],
                span_text=stomach_span,
                display_text="The risk of gastrointestinal and renal side effects is significantly higher with Advil.",
            ),
            ValidatedClaim(
                claim=Claim(id="c2", text=liver_span, type="safety", importance=8),
                label="supported",
                spans=[_span("34503427", liver_span)],
                span_text=liver_span,
                display_text="Hepatobiliary side effects are more frequently linked to Tylenol.",
            ),
        ],
        question="Tylenol vs Advil for fever",
    )

    assert answer.startswith("Tylenol and Advil are not the same medicine.")
    assert "Quick comparison" in answer
    assert "Which is better?" in answer
    assert "Bottom line" in answer
    assert "- Advil:" in answer
    assert "- Tylenol:" in answer
    assert "(Supported by [PMID 34503427]" in answer
    assert answer.count("- ") >= 4

    traceability = ClaimTraceabilityGate().validate(
        answer,
        [
            MedicalArticleRecord(
                pmid="34503427",
                title="Safety profile of acetaminophen and ibuprofen",
                abstract=f"{stomach_span} {liver_span}",
                url="https://pubmed.ncbi.nlm.nih.gov/34503427/",
                source_domain="pubmed.ncbi.nlm.nih.gov",
            )
        ],
    )
    assert traceability.publishable
    assert not any(
        record.sentence_type == "factual" and not record.supported
        for record in traceability.sentences
    )


def test_patient_medication_guide_with_patient_paraphrases_passes_traceability() -> None:
    span_text = (
        "Antipyretic effects of ibuprofen and acetaminophen are similar at physician-directed doses; "
        "ibuprofen may be modestly superior at over-the-counter doses."
    )
    display_text = (
        "Tylenol and Advil appear similarly effective for fever at physician-directed doses; "
        "at over-the-counter doses, Advil may be modestly more effective."
    )
    answer = compose_patient_medication_guide(
        [
            ValidatedClaim(
                claim=Claim(id="c1", text=span_text, type="efficacy", importance=9),
                label="supported",
                spans=[_span("33966545", span_text)],
                span_text=span_text,
                display_text=display_text,
            )
        ],
        question="Tylenol vs Advil for fever",
    )

    traceability = ClaimTraceabilityGate().validate(
        answer,
        [
            MedicalArticleRecord(
                pmid="33966545",
                title="Acetaminophen and ibuprofen in the treatment of pediatric fever: a narrative review.",
                abstract=span_text,
                url="https://pubmed.ncbi.nlm.nih.gov/33966545/",
                source_domain="pubmed.ncbi.nlm.nih.gov",
            )
        ],
    )

    assert traceability.publishable
    assert any(record.span_text == span_text for record in traceability.sentences)


def test_patient_medication_guide_limits_each_bullet_to_two_citations() -> None:
    claim_text = "Ibuprofen increased gastrointestinal bleeding risk."

    answer = compose_patient_medication_guide(
        [
            ValidatedClaim(
                claim=Claim(id="c1", text=claim_text, type="safety", importance=9),
                label="supported",
                spans=[
                    _span("111", claim_text),
                    _span("222", claim_text),
                    _span("333", claim_text),
                ],
            )
        ],
        question="Is Advil safe?",
    )

    assert "- Ibuprofen increased gastrointestinal bleeding risk." in answer
    assert "[PMID 111]" in answer
    assert "[PMID 222]" in answer
    assert "[PMID 333]" not in answer


def test_patient_medication_guide_returns_empty_without_patient_relevant_supported_claims() -> None:
    methods_text = "A total of 1,245 records were identified through database searching."

    answer = compose_patient_medication_guide(
        [
            ValidatedClaim(
                claim=Claim(id="c1", text=methods_text, type="other", importance=10),
                label="supported",
                spans=[_span("444", methods_text)],
            )
        ],
        question="Tylenol vs Advil",
    )

    assert answer == ""


def test_patient_medication_guide_renders_only_bullets_with_backing_pmids_and_passes_traceability() -> None:
    claim_text = "Ibuprofen reduced fever and pain in children."
    answer = compose_patient_medication_guide(
        [
            ValidatedClaim(
                claim=Claim(id="c1", text=claim_text, type="efficacy", importance=9),
                label="supported",
                spans=[_span("222", claim_text)],
            ),
            ValidatedClaim(
                claim=Claim(id="c2", text="Acetaminophen reduced fever.", type="efficacy", importance=8),
                label="supported",
                spans=[],
            ),
        ],
        question="Tylenol vs Advil",
    )

    factual_bullets = [line for line in answer.splitlines() if line.startswith("- ")]
    assert factual_bullets
    assert all("[PMID " in line for line in factual_bullets)
    assert "Acetaminophen reduced fever." not in answer

    traceability = ClaimTraceabilityGate().validate(
        answer,
        [
            MedicalArticleRecord(
                pmid="222",
                title="Ibuprofen for fever",
                abstract=claim_text,
                url="https://pubmed.ncbi.nlm.nih.gov/222/",
                source_domain="pubmed.ncbi.nlm.nih.gov",
            )
        ],
    )

    assert traceability.publishable
