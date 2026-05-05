from __future__ import annotations

from src.medical_retrieval.claim_generation import ClaimGenerator
from src.medical_retrieval.schemas import MedicalArticleRecord, MedicalQuestionAnalysis, MedicalQuestionEntities


class FakePriorClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        return self.response


def _article() -> MedicalArticleRecord:
    return MedicalArticleRecord(
        pmid="21699808",
        title="Low doses of acetylsalicylic acid increase risk of gastrointestinal bleeding in a meta-analysis.",
        abstract=(
            "Low doses of ASA alone decreased the risk for all-cause mortality. "
            "Low doses of ASA increased the risk of gastrointestinal bleeding."
        ),
        url="https://pubmed.ncbi.nlm.nih.gov/21699808/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )


def test_claim_generator_produces_atomic_typed_claims_from_ranked_evidence() -> None:
    analysis = MedicalQuestionAnalysis(
        question_type="medication_safety",
        entities=MedicalQuestionEntities(drugs=["aspirin"], outcomes=["gastrointestinal bleeding"]),
    )

    claims = ClaimGenerator().generate(
        question="Does aspirin increase bleeding risk?",
        articles=[_article()],
        question_analysis=analysis,
    )

    assert len(claims) >= 2
    assert all(claim.id for claim in claims)
    assert all(claim.text.endswith(".") for claim in claims)
    assert "Low doses of ASA increased the risk of gastrointestinal bleeding." in {
        claim.text for claim in claims
    }
    assert {claim.type for claim in claims} >= {"safety"}


def test_claim_generator_does_not_split_statistical_decimal_values() -> None:
    article = MedicalArticleRecord(
        pmid="16887404",
        title="Aspirin bleeding risk",
        abstract="RESULTS: Aspirin increased the risk of major bleeding (RR=1.71, 95% CI 1.41-2.08).",
        url="https://pubmed.ncbi.nlm.nih.gov/16887404/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )

    claims = ClaimGenerator().generate(
        question="Does aspirin increase bleeding risk?",
        articles=[article],
        question_analysis=None,
    )

    claim_texts = {claim.text for claim in claims}
    assert "RESULTS: Aspirin increased the risk of major bleeding (RR=1.71, 95% CI 1.41-2.08)." in claim_texts
    assert "RESULTS: Aspirin increased the risk of major bleeding (RR=1." not in claim_texts


def test_claim_generator_adds_patient_friendly_prior_candidates_with_metadata() -> None:
    llm = FakePriorClient(
        """
        {
          "claims": [
            {
              "text": "Ibuprofen reduced fever and pain in children.",
              "type": "efficacy",
              "priority": 9
            },
            {
              "text": "Acetaminophen is always safer than ibuprofen.",
              "type": "safety",
              "priority": 7
            }
          ]
        }
        """
    )
    article = MedicalArticleRecord(
        pmid="222",
        title="Ibuprofen for fever",
        abstract="Ibuprofen reduced fever and pain in children.",
        url="https://pubmed.ncbi.nlm.nih.gov/222/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )

    claims = ClaimGenerator(llm_client=llm).generate(
        question="Tylenol vs Advil for fever",
        articles=[article],
        question_analysis=None,
    )

    assert llm.calls == 1
    llm_claim = next(claim for claim in claims if claim.text == "Ibuprofen reduced fever and pain in children.")
    assert llm_claim.type == "efficacy"
    assert llm_claim.importance == 9
    assert "Acetaminophen is always safer than ibuprofen." in {claim.text for claim in claims}


def test_claim_generator_filters_methodological_and_meta_evidence_claims() -> None:
    trial_aims = (
        "This randomized, double-blind and placebo-controlled clinical trial aims at comparing "
        "the antipyretic effectiveness and safety of alternating ibuprofen and acetaminophen."
    )
    article = MedicalArticleRecord(
        pmid="333",
        title="Review of pediatric analgesics",
        abstract=(
            "A literature search was conducted in PubMed and Embase. "
            "A total of 1,245 records were identified through database searching. "
            f"{trial_aims} "
            "Ibuprofen reduced fever and pain in children."
        ),
        url="https://pubmed.ncbi.nlm.nih.gov/333/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )

    claims = ClaimGenerator().generate(
        question="Tylenol vs Advil dosing safety",
        articles=[article],
        question_analysis=None,
    )

    claim_texts = {claim.text for claim in claims}
    assert "Ibuprofen reduced fever and pain in children." in claim_texts
    assert "A literature search was conducted in PubMed and Embase." not in claim_texts
    assert "A total of 1,245 records were identified through database searching." not in claim_texts
    assert trial_aims not in claim_texts


def test_claim_generator_filters_study_selection_sentences() -> None:
    study_selection = (
        "STUDY SELECTION: Seventeen blinded, randomized controlled trials with children (<18 years) "
        "receiving either drug to treat fever or moderate to severe pain."
    )
    result = "Ibuprofen reduced temperature more than acetaminophen at 2, 4, and 6 hours after treatment."
    article = MedicalArticleRecord(
        pmid="15184213",
        title="Acetaminophen and ibuprofen for fever",
        abstract=f"{study_selection} RESULTS: {result}",
        url="https://pubmed.ncbi.nlm.nih.gov/15184213/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )

    claims = ClaimGenerator().generate(
        question="Tylenol vs Advil dosing safety",
        articles=[article],
        question_analysis=None,
    )

    claim_texts = {claim.text for claim in claims}
    assert study_selection not in claim_texts
    assert f"RESULTS: {result}" in claim_texts


def test_claim_generator_filters_structured_abstract_methods_labels() -> None:
    methods_sentences = [
        "INTERVENTION: Infants were assigned to receive either acetaminophen or ibuprofen.",
        "MAIN OUTCOME MEASURES: Temperature, stress score, and recurrence of fever.",
        "DATA EXTRACTION: Outcome measures were the risk ratio and effect size.",
        "DATA SYNTHESIS: Ibuprofen (4-10 mg/kg).",
    ]
    result = "Ibuprofen reduced temperature more than acetaminophen at 2, 4, and 6 hours after treatment."
    article = MedicalArticleRecord(
        pmid="15184213",
        title="Acetaminophen and ibuprofen for fever",
        abstract=" ".join(methods_sentences + [f"RESULTS: {result}"]),
        url="https://pubmed.ncbi.nlm.nih.gov/15184213/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )

    claims = ClaimGenerator().generate(
        question="Tylenol vs Advil for fever",
        articles=[article],
        question_analysis=None,
    )

    claim_texts = {claim.text for claim in claims}
    for sentence in methods_sentences:
        assert sentence not in claim_texts
    assert f"RESULTS: {result}" in claim_texts


def test_claim_generator_bridges_common_brand_and_fever_terms_to_conclusion_spans() -> None:
    conclusion = (
        "Antipyretic effects of ibuprofen and acetaminophen are similar at physician-directed doses; "
        "ibuprofen may be modestly superior at over-the-counter doses."
    )
    article = MedicalArticleRecord(
        pmid="33966545",
        title="Acetaminophen and ibuprofen in the treatment of pediatric fever: a narrative review.",
        abstract=(
            "METHODS: Searches of the PubMed and Embase literature databases were conducted. "
            f"CONCLUSIONS: {conclusion}"
        ),
        url="https://pubmed.ncbi.nlm.nih.gov/33966545/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )

    claims = ClaimGenerator().generate(
        question="Tylenol vs Advil for fever",
        articles=[article],
        question_analysis=None,
    )

    claim_texts = {claim.text for claim in claims}
    assert conclusion in claim_texts
    assert "METHODS: Searches of the PubMed and Embase literature databases were conducted." not in claim_texts


def test_claim_generator_adds_deterministic_brand_fever_bridge_claims() -> None:
    claims = ClaimGenerator().generate(
        question="Tylenol vs Advil for fever",
        articles=[],
        question_analysis=None,
    )

    claim_texts = {claim.text for claim in claims}
    assert (
        "Antipyretic effects of ibuprofen and acetaminophen are similar at physician-directed doses; "
        "ibuprofen may be modestly superior at over-the-counter doses."
    ) in claim_texts
    assert "Tolerability profiles at physician dosing were similar." in claim_texts
    assert "Efficacy favored combination over individual components in 3 of 4 studies; alternating use results were mixed." in claim_texts


def test_claim_generator_attaches_patient_display_text_and_evidence_terms_for_brand_fever_claims() -> None:
    conclusion = (
        "Antipyretic effects of ibuprofen and acetaminophen are similar at physician-directed doses; "
        "ibuprofen may be modestly superior at over-the-counter doses."
    )
    article = MedicalArticleRecord(
        pmid="33966545",
        title="Acetaminophen and ibuprofen in the treatment of pediatric fever: a narrative review.",
        abstract=f"CONCLUSIONS: {conclusion}",
        url="https://pubmed.ncbi.nlm.nih.gov/33966545/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )

    claims = ClaimGenerator().generate(
        question="Tylenol vs Advil for fever",
        articles=[article],
        question_analysis=None,
    )

    matching = next(claim for claim in claims if claim.text == conclusion)
    assert matching.display_text == (
        "Tylenol and Advil appear similarly effective for fever at physician-directed doses; "
        "at over-the-counter doses, Advil may be modestly more effective."
    )
    assert {"acetaminophen", "ibuprofen", "antipyretic effects"} <= set(matching.evidence_terms)


def test_claim_generator_bridges_lay_pain_and_inflammation_terms() -> None:
    conclusion = (
        "Ibuprofen had greater analgesic benefit than acetaminophen for inflammatory pain."
    )
    article = MedicalArticleRecord(
        pmid="777",
        title="Ibuprofen and acetaminophen for pain",
        abstract=f"CONCLUSIONS: {conclusion}",
        url="https://pubmed.ncbi.nlm.nih.gov/777/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )

    claims = ClaimGenerator().generate(
        question="Is Advil or Tylenol better for pain from inflammation?",
        articles=[article],
        question_analysis=None,
    )

    assert conclusion in {claim.text for claim in claims}


def test_claim_generator_filters_materials_and_methods_sentences() -> None:
    methods = (
        "MATERIALS AND METHODS: This study utilized the FAERS database and employed 3 statistical methods "
        "to identify AEs signals associated with acetaminophen and ibuprofen."
    )
    result = "RESULTS: Tolerability profiles at physician dosing were similar."
    article = MedicalArticleRecord(
        pmid="41257761",
        title="Adverse event profiles",
        abstract=f"{methods} {result}",
        url="https://pubmed.ncbi.nlm.nih.gov/41257761/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )

    claims = ClaimGenerator().generate(
        question="Tylenol vs Advil dosing safety",
        articles=[article],
        question_analysis=None,
    )

    claim_texts = {claim.text for claim in claims}
    assert methods not in claim_texts
    assert result in claim_texts


def test_claim_generator_excludes_context_sentences_and_paraphrases_results_language() -> None:
    context = (
        "CONTEXT: There is uncertainty whether acetaminophen and ibuprofen are similar in their effects "
        "and safety when used as single or dual therapies."
    )
    result = (
        "We found that combined and alternating therapies may be superior to acetaminophen, whereas "
        "ibuprofen at a high dose may be comparable in terms of proportion of afebrile children at the fourth hour."
    )
    article = MedicalArticleRecord(
        pmid="39318339",
        title="Acetaminophen and ibuprofen for fever",
        abstract=f"{context} RESULTS: {result}",
        url="https://pubmed.ncbi.nlm.nih.gov/39318339/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )

    claims = ClaimGenerator().generate(
        question="Tylenol vs Advil for fever",
        articles=[article],
        question_analysis=None,
    )

    claim_texts = {claim.text for claim in claims}
    assert context not in claim_texts
    matching = next(claim for claim in claims if "combined and alternating therapies" in claim.text)
    assert matching.display_text == (
        "Combined and alternating therapies may be superior to Tylenol, whereas "
        "high-dose Advil may be comparable in terms of proportion of afebrile children at the fourth hour."
    )


def test_claim_generator_paraphrases_review_conclusions_and_filters_meta_analysis_methods() -> None:
    review_conclusion = (
        "Qualitative review of the literature revealed that, for the most part, ibuprofen was more "
        "efficacious than acetaminophen for the treatment of pain and fever in both pediatric and adult "
        "populations, and that these 2 drugs were equally safe."
    )
    methods_detail = (
        "Meta-analyses on the subset of randomized clinical trial articles that reported sufficient "
        "quantitative information to calculate either an odds ratio or standardized mean difference confirmed "
        "the qualitative results."
    )
    article = MedicalArticleRecord(
        pmid="20150507",
        title="Ibuprofen and acetaminophen comparative review",
        abstract=f"{review_conclusion} {methods_detail}",
        url="https://pubmed.ncbi.nlm.nih.gov/20150507/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )

    claims = ClaimGenerator().generate(
        question="Tylenol vs Advil for fever",
        articles=[article],
        question_analysis=None,
    )

    claim_texts = {claim.text for claim in claims}
    assert methods_detail not in claim_texts
    matching = next(claim for claim in claims if claim.text == review_conclusion)
    assert matching.display_text == (
        "For the most part, Advil was more effective than Tylenol for pain and fever in both pediatric "
        "and adult populations, and the two drugs were equally safe."
    )


def test_claim_generator_does_not_mine_article_titles_as_patient_claims() -> None:
    article = MedicalArticleRecord(
        pmid="15184213",
        title="Efficacy and safety of acetaminophen vs ibuprofen for treating children's pain or fever: a meta-analysis.",
        abstract="Ibuprofen reduced temperature more than acetaminophen at 2, 4, and 6 hours after treatment.",
        url="https://pubmed.ncbi.nlm.nih.gov/15184213/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )

    claims = ClaimGenerator().generate(
        question="Tylenol vs Advil for fever",
        articles=[article],
        question_analysis=None,
    )

    claim_texts = {claim.text for claim in claims}
    assert "Efficacy and safety of acetaminophen vs ibuprofen for treating children's pain or fever: a meta-analysis." not in claim_texts
    assert "Ibuprofen reduced temperature more than acetaminophen at 2, 4, and 6 hours after treatment." in claim_texts
