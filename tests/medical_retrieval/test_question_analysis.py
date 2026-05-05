from __future__ import annotations

import importlib.util
import json
from typing import Any

import pytest

from src.medical_retrieval.question_analysis import QuestionAnalysisError, QuestionAnalyzer
from src.medical_retrieval.rxnorm_helper import RxNormHelper


class FakeLLM:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeResolver:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def resolve_names_with_context(self, names: list[str]):
        self.calls.append(names)
        return [
            type(
                "Resolved",
                (),
                {
                    "raw_name": "Tylenol",
                    "canonical_name": "acetaminophen",
                    "generic_name": "acetaminophen",
                    "match_type": "brand",
                    "rxnorm_cui": "161",
                    "confidence": 0.97,
                },
            )()
        ]


class ExplodingResolverFactory:
    def __call__(self):  # pragma: no cover - should never be reached
        raise AssertionError("resolver should not be constructed for empty mentions")


class FakeProviderTransportError(RuntimeError):
    is_expected_llm_error = True


def test_analyzer_parses_fenced_json_for_medication_safety_question_and_preserves_object_lists() -> None:
    llm = FakeLLM(
        """```json
        {
          "question_type": "medication_safety",
          "entities": {
            "drugs": [{"raw_name": "Tylenol", "role": "exposure"}],
            "supplements": [],
            "normalized_ingredients": [{"raw_name": "Tylenol", "canonical_name": "acetaminophen"}],
            "drug_classes": [{"name": "analgesics"}],
            "conditions": [{"name": "chronic liver disease"}],
            "populations": [{"name": "adults"}],
            "outcomes": [{"name": "hepatotoxicity"}],
            "mechanisms": [{"name": "glutathione depletion"}]
          },
          "rxnorm_candidates": [
            {
              "raw_name": "Tylenol",
              "canonical_name": "acetaminophen",
              "match_type": "brand",
              "rxnorm_cui": "161",
              "confidence": 0.97
            }
          ],
          "retrieval_constraints": ["human studies", "safety outcomes"],
          "rationale": "The question asks about acetaminophen safety in liver disease."
        }
        ```"""
    )
    analyzer = QuestionAnalyzer(llm_client=llm)

    analysis = analyzer.analyze(
        "Can I take Tylenol if I have chronic liver disease?",
        conversation_context=[{"role": "user", "content": "Adult patient context."}],
    )

    assert analysis.question_type == "medication_safety"
    assert analysis.entities.drugs[0].raw_name == "Tylenol"
    assert analysis.entities.drugs[0].role == "exposure"
    assert analysis.entities.normalized_ingredients[0].canonical_name == "acetaminophen"
    assert analysis.entities.drug_classes[0].name == "analgesics"
    assert analysis.entities.conditions[0].name == "chronic liver disease"
    assert analysis.entities.populations[0].name == "adults"
    assert analysis.entities.outcomes[0].name == "hepatotoxicity"
    assert analysis.entities.mechanisms[0].name == "glutathione depletion"
    assert analysis.rxnorm_candidates[0].rxnorm_cui == "161"
    assert analysis.retrieval_constraints == ["human studies", "safety outcomes"]
    assert "Perplexity Agent" in llm.calls[0]["messages"][0]["content"]
    assert "conversation_context" in llm.calls[0]["messages"][1]["content"]


def test_analyzer_handles_supplement_question_without_forcing_rxnorm_dependency() -> None:
    llm = FakeLLM(
        json.dumps(
            {
                "question_type": "supplement_safety",
                "entities": {
                    "drugs": [],
                    "supplements": [{"raw_name": "St. John's wort", "role": "exposure"}],
                    "normalized_ingredients": [{"raw_name": "St. John's wort", "canonical_name": "Hypericum perforatum"}],
                    "drug_classes": [],
                    "conditions": [{"name": "depression"}],
                    "populations": [],
                    "outcomes": [{"name": "serotonin syndrome"}],
                    "mechanisms": [{"name": "CYP3A4 induction"}],
                },
                "rxnorm_candidates": [],
                "retrieval_constraints": ["human studies"],
                "rationale": "The question names a supplement, not a drug product requiring RxNorm.",
            }
        )
    )
    analyzer = QuestionAnalyzer(llm_client=llm)

    analysis = analyzer.analyze("Is St. John's wort safe for depression?")

    assert analysis.entities.drugs == []
    assert analysis.entities.supplements[0].raw_name == "St. John's wort"
    assert analysis.rxnorm_candidates == []


def test_analyzer_normalizes_known_structured_output_variants_before_validation() -> None:
    llm = FakeLLM(
        json.dumps(
            {
                "question_type": "medication_interaction",
                "entities": {
                    "drugs": [{"name": "Tylenol"}, {"name": "Advil"}],
                    "supplements": [],
                    "normalized_ingredients": [{"name": "acetaminophen"}, {"name": "ibuprofen"}],
                    "drug_classes": [{"name": "NSAID"}],
                    "conditions": [],
                    "populations": [],
                    "outcomes": [{"name": "drug interaction safety"}],
                    "mechanisms": [{"name": "combined analgesic use"}],
                },
                "rxnorm_candidates": [
                    {"mention": "Tylenol", "candidates": [{"name": "acetaminophen", "rxnorm_id": "161"}]},
                    {"mention": "Advil", "candidates": [{"name": "ibuprofen", "rxnorm_id": "5640"}]},
                ],
                "retrieval_constraints": {
                    "must_include": ["coadministration", "concurrent use"],
                    "exclude": [],
                    "time_sensitivity": "none",
                    "evidence_preferences": ["clinical guidance", "drug labeling"],
                },
            }
        )
    )
    analyzer = QuestionAnalyzer(llm_client=llm)

    analysis = analyzer.analyze("Can I take Tylenol and Advil together?")

    assert [drug.raw_name for drug in analysis.entities.drugs] == ["Tylenol", "Advil"]
    assert [
        (ingredient.raw_name, ingredient.canonical_name)
        for ingredient in analysis.entities.normalized_ingredients
    ] == [("acetaminophen", "acetaminophen"), ("ibuprofen", "ibuprofen")]
    assert analysis.rxnorm_candidates[0].raw_name == "Tylenol"
    assert analysis.rxnorm_candidates[0].canonical_name == "acetaminophen"
    assert analysis.rxnorm_candidates[0].rxnorm_cui == "161"
    assert analysis.rxnorm_candidates[1].raw_name == "Advil"
    assert analysis.rxnorm_candidates[1].canonical_name == "ibuprofen"
    assert analysis.rxnorm_candidates[1].rxnorm_cui == "5640"
    assert analysis.retrieval_constraints == [
        "must_include: coadministration",
        "must_include: concurrent use",
        "time_sensitivity: none",
        "evidence_preferences: clinical guidance",
        "evidence_preferences: drug labeling",
    ]


def test_analyzer_sends_strict_structured_output_response_format() -> None:
    llm = FakeLLM(
        json.dumps(
            {
                "question_type": "medication_safety",
                "entities": {
                    "drugs": [{"raw_name": "Tylenol", "role": None}],
                    "supplements": [],
                    "normalized_ingredients": [{"raw_name": "Tylenol", "canonical_name": "acetaminophen"}],
                    "drug_classes": [],
                    "conditions": [],
                    "populations": [],
                    "outcomes": [],
                    "mechanisms": [],
                },
                "rxnorm_candidates": [],
                "retrieval_constraints": ["PubMed evidence"],
                "rationale": None,
                "reasoning": None,
            }
        )
    )
    analyzer = QuestionAnalyzer(llm_client=llm)

    analyzer.analyze("Is Tylenol safe?")

    response_format = llm.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["entities"]["additionalProperties"] is False


def test_analyzer_normalizes_known_string_bucket_variants_before_validation() -> None:
    llm = FakeLLM(
        json.dumps(
            {
                "question_type": "medication_safety",
                "entities": {
                    "drugs": ["Tylenol"],
                    "supplements": [],
                    "normalized_ingredients": ["acetaminophen"],
                    "drug_classes": ["analgesic"],
                    "conditions": [],
                    "populations": ["adults"],
                    "outcomes": ["liver injury"],
                    "mechanisms": ["glutathione depletion"],
                },
                "rxnorm_candidates": [],
                "retrieval_constraints": ["human PubMed evidence"],
            }
        )
    )
    analyzer = QuestionAnalyzer(llm_client=llm)

    analysis = analyzer.analyze("Is Tylenol safe for adults?")

    assert analysis.entities.drugs[0].raw_name == "Tylenol"
    assert analysis.entities.normalized_ingredients[0].canonical_name == "acetaminophen"
    assert analysis.entities.drug_classes[0].name == "analgesic"
    assert analysis.entities.populations[0].name == "adults"


@pytest.mark.parametrize("response", ["not json", "{\"question_type\": \"medication_safety\", \"entities\": []}"])
def test_analyzer_raises_question_analysis_error_on_invalid_json_and_invalid_schema(response: str) -> None:
    analyzer = QuestionAnalyzer(llm_client=FakeLLM(response))

    with pytest.raises(QuestionAnalysisError):
        analyzer.analyze("Can I take ibuprofen with kidney disease?")


def test_analyzer_wraps_expected_provider_transport_failure() -> None:
    analyzer = QuestionAnalyzer(llm_client=FakeLLM(FakeProviderTransportError("provider unavailable")))

    with pytest.raises(QuestionAnalysisError) as exc_info:
        analyzer.analyze("Can I take ibuprofen with kidney disease?")

    assert isinstance(exc_info.value.__cause__, FakeProviderTransportError)


def test_analyzer_does_not_wrap_unexpected_programming_errors_from_llm_client() -> None:
    analyzer = QuestionAnalyzer(llm_client=FakeLLM(AttributeError("misconfigured client")))

    with pytest.raises(AttributeError, match="misconfigured client"):
        analyzer.analyze("Can I take ibuprofen with kidney disease?")


def test_analyzer_raises_question_analysis_error_when_entities_missing_required_bucket() -> None:
    analyzer = QuestionAnalyzer(
        llm_client=FakeLLM(
            json.dumps(
                {
                    "question_type": "medication_safety",
                    "entities": {
                        "drugs": [{"raw_name": "ibuprofen"}],
                        "supplements": [],
                        "normalized_ingredients": [],
                        "drug_classes": [],
                        "conditions": [{"name": "kidney disease"}],
                        "populations": [],
                        "outcomes": [],
                    },
                    "retrieval_constraints": [],
                }
            )
        )
    )

    with pytest.raises(QuestionAnalysisError):
        analyzer.analyze("Can I take ibuprofen with kidney disease?")


def test_analyzer_raises_question_analysis_error_when_retrieval_constraints_missing() -> None:
    analyzer = QuestionAnalyzer(
        llm_client=FakeLLM(
            json.dumps(
                {
                    "question_type": "medication_safety",
                    "entities": {
                        "drugs": [{"raw_name": "ibuprofen"}],
                        "supplements": [],
                        "normalized_ingredients": [],
                        "drug_classes": [],
                        "conditions": [{"name": "kidney disease"}],
                        "populations": [],
                        "outcomes": [],
                        "mechanisms": [],
                    },
                }
            )
        )
    )

    with pytest.raises(QuestionAnalysisError):
        analyzer.analyze("Can I take ibuprofen with kidney disease?")


@pytest.mark.parametrize("retrieval_constraints", [None, "human studies"])
def test_analyzer_raises_question_analysis_error_when_retrieval_constraints_is_not_a_list(
    retrieval_constraints: object,
) -> None:
    analyzer = QuestionAnalyzer(
        llm_client=FakeLLM(
            json.dumps(
                {
                    "question_type": "medication_safety",
                    "entities": {
                        "drugs": [{"raw_name": "ibuprofen"}],
                        "supplements": [],
                        "normalized_ingredients": [],
                        "drug_classes": [],
                        "conditions": [{"name": "kidney disease"}],
                        "populations": [],
                        "outcomes": [],
                        "mechanisms": [],
                    },
                    "retrieval_constraints": retrieval_constraints,
                }
            )
        )
    )

    with pytest.raises(QuestionAnalysisError):
        analyzer.analyze("Can I take ibuprofen with kidney disease?")


def test_analyzer_prompt_is_medication_focused_without_broad_arbitrary_medical_goal() -> None:
    analyzer = QuestionAnalyzer(llm_client=FakeLLM("{}"))

    prompt = analyzer._system_prompt()

    assert "medication" in prompt.lower()
    assert "supplement" in prompt.lower()
    assert "Perplexity Agent" in prompt
    assert "broad medical" not in prompt.lower()
    assert "arbitrary medical" not in prompt.lower()


def test_rxnorm_helper_returns_empty_without_constructing_or_calling_resolver_for_empty_mentions() -> None:
    helper = RxNormHelper(resolver_factory=ExplodingResolverFactory())

    assert helper.normalize_mentions([]) == []
    assert helper.normalize_mentions(["", "   "]) == []


def test_rxnorm_helper_delegates_to_fake_resolver_for_drug_mentions_and_maps_records() -> None:
    resolver = FakeResolver()
    helper = RxNormHelper(resolver=resolver)

    records = helper.normalize_mentions([{"raw_name": " Tylenol "}, ""])

    assert resolver.calls == [["Tylenol"]]
    assert len(records) == 1
    assert records[0].raw_name == "Tylenol"
    assert records[0].canonical_name == "acetaminophen"
    assert records[0].generic_name == "acetaminophen"
    assert records[0].match_type == "brand"
    assert records[0].rxnorm_cui == "161"
    assert records[0].confidence == 0.97


def test_stale_strategy_planner_module_is_absent() -> None:
    assert importlib.util.find_spec("src.medical_retrieval.strategy_planner") is None
