from __future__ import annotations

from src.evidence.query_builder import QueryBuilder
from src.evidence.schemas import NormalizedDrug


class StubLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate_simple(self, *_args, **_kwargs) -> str:
        return self.response


def _context(*, raw_name: str, canonical_name: str, confidence: float, aliases: list[str] | None = None) -> dict[str, object]:
    return {
        "raw_name": raw_name,
        "canonical_name": canonical_name,
        "confidence": confidence,
        "aliases": aliases or [raw_name, canonical_name],
    }


def test_query_builder_parses_json_queries() -> None:
    builder = QueryBuilder(
        llm_client=StubLLMClient(
            '{"queries":[{"query":"acetaminophen liver injury","strategy_label":"broad"},'
            '{"query":"acetaminophen hepatotoxicity adults","strategy_label":"narrow"}]}'
        ),
        model="gpt-5.4-mini",
    )

    result = builder.build_queries(
        question="Is Tylenol safe for the liver?",
        intent="safety",
        normalized_drugs=[
            NormalizedDrug(raw_name="Tylenol", generic_name="acetaminophen", match_type="exact", rxnorm_cui="123")
        ],
        drug_contexts=[_context(raw_name="Tylenol", canonical_name="acetaminophen", confidence=0.92)],
    )

    assert [item.query for item in result] == [
        "acetaminophen liver injury",
        "acetaminophen hepatotoxicity adults",
    ]


def test_query_builder_sanitizes_bad_query_text_and_prefers_generic_terms() -> None:
    builder = QueryBuilder(
        llm_client=StubLLMClient(
            '{"queries":[{"query":"Tylenol acetaminophen safety adults PubMed","strategy_label":"narrow"}]}'
        ),
        model="gpt-5.4-mini",
    )

    result = builder.build_queries(
        question="Is Tylenol safe for adults?",
        intent="safety",
        normalized_drugs=[
            NormalizedDrug(raw_name="Tylenol", generic_name="acetaminophen", match_type="exact", rxnorm_cui="123")
        ],
        drug_contexts=[_context(raw_name="Tylenol", canonical_name="acetaminophen", confidence=0.92)],
    )

    assert [item.query for item in result] == ["acetaminophen safety adults"]


def test_query_builder_compacts_overly_long_safety_queries_for_pubmed() -> None:
    builder = QueryBuilder(
        llm_client=StubLLMClient(
            '{"queries":[{"query":"acetaminophen safety adverse effects toxicity overdose contraindications liver injury pregnancy","strategy_label":"broad"}]}'
        ),
        model="gpt-5.4-mini",
    )

    result = builder.build_queries(
        question="Is Tylenol safe to take?",
        intent="safety",
        normalized_drugs=[
            NormalizedDrug(raw_name="Tylenol", generic_name="acetaminophen", match_type="exact", rxnorm_cui="123")
        ],
        drug_contexts=[_context(raw_name="Tylenol", canonical_name="acetaminophen", confidence=0.92)],
    )

    assert [item.query for item in result] == ["acetaminophen safety adverse effects toxicity"]


def test_query_builder_falls_back_to_emergency_query_when_json_is_invalid() -> None:
    builder = QueryBuilder(llm_client=StubLLMClient("not-json"), model="gpt-5.4-mini")

    result = builder.build_queries(
        question="Is Tylenol safe to take?",
        intent="safety",
        normalized_drugs=[
            NormalizedDrug(raw_name="Tylenol", generic_name="acetaminophen", match_type="exact", rxnorm_cui="123")
        ],
        drug_contexts=[_context(raw_name="Tylenol", canonical_name="acetaminophen", confidence=0.92)],
    )

    assert len(result) == 1
    assert result[0].strategy_label == "fallback"
    assert "acetaminophen" in result[0].query


def test_query_builder_fallback_for_safety_uses_direct_safety_terms() -> None:
    builder = QueryBuilder(llm_client=StubLLMClient("not-json"), model="gpt-5.4-mini")

    result = builder.build_queries(
        question="Is Tylenol safe to take?",
        intent="safety",
        normalized_drugs=[
            NormalizedDrug(raw_name="Tylenol", generic_name="acetaminophen", match_type="exact", rxnorm_cui="123")
        ],
        drug_contexts=[_context(raw_name="Tylenol", canonical_name="acetaminophen", confidence=0.92)],
    )

    assert len(result) == 1
    assert result[0].strategy_label == "fallback"
    assert result[0].query.startswith("acetaminophen")
    assert "adverse" in result[0].query
    assert "toxicity" in result[0].query
    assert "contraindications" in result[0].query


def test_query_builder_uses_raw_anchor_when_normalization_confidence_is_low() -> None:
    builder = QueryBuilder(llm_client=StubLLMClient("not-json"), model="gpt-5.4-mini")

    result = builder.build_queries(
        question="Does creatine help with muscle building?",
        intent="benefits",
        normalized_drugs=[
            NormalizedDrug(raw_name="creatine monohydrate", generic_name="creatine", match_type="none", rxnorm_cui=None)
        ],
        drug_contexts=[
            _context(
                raw_name="creatine monohydrate",
                canonical_name="creatine",
                confidence=0.2,
                aliases=["creatine monohydrate", "creatine"],
            )
        ],
    )

    assert result[0].query.startswith("creatine monohydrate")


def test_query_builder_uses_canonical_anchor_when_normalization_confidence_is_high() -> None:
    builder = QueryBuilder(llm_client=StubLLMClient("not-json"), model="gpt-5.4-mini")

    result = builder.build_queries(
        question="Is Tylenol safe to take?",
        intent="safety",
        normalized_drugs=[
            NormalizedDrug(raw_name="Tylenol", generic_name="acetaminophen", match_type="exact", rxnorm_cui="123")
        ],
        drug_contexts=[_context(raw_name="Tylenol", canonical_name="acetaminophen", confidence=0.92)],
    )

    assert result[0].query.startswith("acetaminophen")


def test_query_builder_adds_one_broad_query_for_vague_benefit_questions() -> None:
    builder = QueryBuilder(
        llm_client=StubLLMClient(
            '{"queries":[{"query":"vitamin D benefits","strategy_label":"broad"},'
            '{"query":"vitamin D immune function bone health adults","strategy_label":"focused"}]}'
        ),
        model="gpt-5.4-mini",
    )

    result = builder.build_queries(
        question="What are the benefits of vitamin D?",
        intent="benefits",
        normalized_drugs=[
            NormalizedDrug(raw_name="vitamin D", generic_name="vitamin D", match_type="approximate", rxnorm_cui="11253")
        ],
        drug_contexts=[
            _context(
                raw_name="vitamin D",
                canonical_name="vitamin D",
                confidence=0.7,
                aliases=["vitamin D", "cholecalciferol"],
            )
        ],
    )

    assert [item.query for item in result] == [
        "vitamin D benefits",
        "vitamin D immune function bone health adults",
    ]
