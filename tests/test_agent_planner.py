"""Unit tests for the agent planner."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from src.agent.config import AgentConfig
from src.agent.models import EvaluationVerdict
from src.agent.models import PlannedQuery
from src.agent.models import QueryIntent
from src.agent.models import QueryPlan
from src.agent.models import RetrievedArticle
from src.agent.models import SubQuestion
from src.agent.models import SubQuestionCoverage
from src.agent.planner import QueryPlanner, _normalize_planner_model, _normalize_str_list
from src.llm_query_generator import validate_pubmed_query
from src.nvidia_llm_client import NVIDIALLMAPIError, NVIDIALLMClient

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def _verdict() -> EvaluationVerdict:
    return EvaluationVerdict(
        is_sufficient=False,
        relevant_articles=[],
        study_type_distribution={},
        coverage_gaps=["monitoring"],
        recommendation="broaden",
        flags=[],
        failure_modes=["insufficient_direct_evidence", "subquestion_gap"],
        subquestion_coverages=[
            SubQuestionCoverage(
                sub_question_index=0,
                sub_question_text="Does alcohol change exposure?",
                coverage_score=0.1,
                matched_pmids=[],
                covered=False,
            )
        ],
    )


class TestQueryPlanner:
    def test_planner_uses_configured_model_for_generate_calls(self) -> None:
        llm_client = Mock()
        llm_client.generate.side_effect = [
            _load_fixture_text("agent_planner_response_wrapped.txt"),
            _load_fixture_text("agent_refine_response_wrapped.txt"),
        ]
        planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())

        plan = planner.plan("Is warfarin safe with fluconazole?")
        planner.refine_plan(plan, _verdict())

        assert llm_client.generate.call_args_list[0].kwargs["model"] == AgentConfig.planner_model
        assert llm_client.generate.call_args_list[1].kwargs["model"] == AgentConfig.planner_model

    def test_planner_passes_through_explicit_model_id_without_coercion(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = _load_fixture_text("agent_planner_response_wrapped.txt")
        planner = QueryPlanner(llm_client=llm_client, config=AgentConfig(planner_model="meta/llama-3.3-70b-instruct"))

        planner.plan("Is warfarin safe with fluconazole?")

        assert llm_client.generate.call_args.kwargs["model"] == "meta/llama-3.3-70b-instruct"

    def test_plan_parses_valid_llm_json(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = _load_fixture_text("agent_planner_response_wrapped.txt")
        planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())

        plan = planner.plan("Is warfarin safe with fluconazole?")

        assert isinstance(plan, QueryPlan)
        assert plan.intent == QueryIntent.drug_interaction
        assert plan.drugs_identified == ["warfarin", "fluconazole"]

    def test_plan_falls_back_on_invalid_json(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = "not json"
        planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())

        plan = planner.plan("What are aspirin side effects?")

        assert plan.planned_queries[0].strategy_label == "fallback"

    def test_plan_falls_back_on_llm_exception(self) -> None:
        llm_client = Mock()
        llm_client.generate.side_effect = NVIDIALLMAPIError("failure")
        planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())

        plan = planner.plan("What are aspirin side effects?")

        assert plan.planned_queries[0].strategy_label == "fallback"

    def test_plan_discards_queries_failing_validation(self) -> None:
        llm_client = Mock()
        payload = json.loads(_load_fixture_text("agent_planner_response.json"))
        payload["intent"] = "unknown"
        payload["drugs_identified"] = ["warfarin"]
        payload["planned_queries"] = [
            {
                "pubmed_query": "(warfarin[tiab]) AND English[Language] AND Humans[Mesh]",
                "sub_question_index": 0,
                "strategy_label": "valid",
            },
            {
                "pubmed_query": "(warfarin[tiab] AND English[Language]",
                "sub_question_index": 0,
                "strategy_label": "invalid",
            },
        ]
        llm_client.generate.return_value = json.dumps(payload)
        planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())

        plan = planner.plan("q")

        assert len(plan.planned_queries) == 1
        assert plan.planned_queries[0].strategy_label == "valid"

    def test_refine_plan_returns_additional_queries(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = _load_fixture_text("agent_refine_response_wrapped.txt")
        planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())
        original_plan = QueryPlan(
            normalized_question="q",
            intent=QueryIntent.unknown,
            sub_questions=[],
            planned_queries=[],
            drugs_identified=[],
            reasoning="r",
        )

        queries = planner.refine_plan(original_plan, _verdict())

        assert len(queries) == 1
        assert queries[0].strategy_label == "refined"

    def test_refine_plan_returns_empty_on_failure(self) -> None:
        llm_client = Mock()
        llm_client.generate.side_effect = NVIDIALLMAPIError("failure")
        planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())
        original_plan = QueryPlan(
            normalized_question="q",
            intent=QueryIntent.unknown,
            sub_questions=[],
            planned_queries=[],
            drugs_identified=[],
            reasoning="r",
        )

        assert planner.refine_plan(original_plan, _verdict()) == []

    def test_refine_plan_uses_failure_modes_and_uncovered_subquestions(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = _load_fixture_text("agent_refine_response_wrapped.txt")
        planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())
        original_plan = QueryPlan(
            normalized_question="q",
            intent=QueryIntent.unknown,
            sub_questions=[],
            planned_queries=[],
            drugs_identified=[],
            reasoning="r",
        )

        planner.refine_plan(original_plan, _verdict())

        call = llm_client.generate.call_args
        system_prompt = call.kwargs["messages"][0]["content"]
        user_prompt = call.kwargs["messages"][1]["content"]
        assert "failure_modes" in system_prompt
        assert "subquestion_coverages" in system_prompt
        assert "sub_question_text" in system_prompt
        assert "failure_modes" in user_prompt
        assert "insufficient_direct_evidence" in user_prompt
        assert "Does alcohol change exposure?" in user_prompt

    def test_refine_plan_defaults_missing_strategy_label_to_targeted_refine(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = json.dumps(
            [
                {
                    "pubmed_query": "(warfarin[tiab]) AND English[Language] AND Humans[Mesh]",
                    "sub_question_index": 0,
                }
            ]
        )
        planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())
        original_plan = QueryPlan(
            normalized_question="q",
            intent=QueryIntent.unknown,
            sub_questions=[],
            planned_queries=[],
            drugs_identified=[],
            reasoning="r",
        )

        queries = planner.refine_plan(original_plan, _verdict())

        assert queries == [
            PlannedQuery(
                pubmed_query="(warfarin[tiab]) AND English[Language] AND Humans[Mesh]",
                sub_question_index=0,
                strategy_label="targeted_refine",
            )
        ]

    def test_refine_plan_defaults_malformed_sub_question_index_per_entry(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = json.dumps(
            [
                {
                    "pubmed_query": "(warfarin[tiab]) AND English[Language] AND Humans[Mesh]",
                    "sub_question_index": "",
                    "strategy_label": "bad-index",
                },
                {
                    "pubmed_query": "(aspirin[tiab]) AND English[Language] AND Humans[Mesh]",
                    "sub_question_index": 2,
                    "strategy_label": "good-index",
                },
            ]
        )
        planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())
        original_plan = QueryPlan(
            normalized_question="q",
            intent=QueryIntent.unknown,
            sub_questions=[],
            planned_queries=[],
            drugs_identified=[],
            reasoning="r",
        )

        queries = planner.refine_plan(original_plan, _verdict())

        assert queries == [
            PlannedQuery(
                pubmed_query="(warfarin[tiab]) AND English[Language] AND Humans[Mesh]",
                sub_question_index=0,
                strategy_label="bad-index",
            ),
            PlannedQuery(
                pubmed_query="(aspirin[tiab]) AND English[Language] AND Humans[Mesh]",
                sub_question_index=2,
                strategy_label="good-index",
            ),
        ]


def test_fallback_query_injects_interaction_term_for_drug_interaction_intent() -> None:
    llm_client = Mock()
    planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())

    queries = planner._fallback_keyword_query("Can I take warfarin with aspirin?", ["warfarin", "aspirin"], intent="drug_interaction")

    assert len(queries) == 1
    assert "interaction" in queries[0].pubmed_query.lower()


def test_fallback_query_prefers_drug_over_generic_token_for_side_effects_intent() -> None:
    """Regression: 'side effects of metformin' must not search for 'side'."""
    llm_client = Mock()
    planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())

    queries = planner._fallback_keyword_query(
        "what are the side effects of metformin",
        ["metformin"],
        intent="side_effects",
    )

    assert len(queries) == 1
    query = queries[0].pubmed_query.lower()
    assert "metformin[tiab]" in query
    # The lead keyword clause must not be driven by the generic token "side".
    assert "side[tiab]" not in query
    assert "adverse effects[tiab]" in query


def test_fallback_query_skips_generic_tokens_when_no_drugs_identified() -> None:
    """Regression: without drugs, lead term should be a specific noun, not 'side'."""
    llm_client = Mock()
    planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())

    queries = planner._fallback_keyword_query(
        "side effects of statins in elderly",
        [],
        intent="side_effects",
    )

    assert len(queries) == 1
    query = queries[0].pubmed_query.lower()
    assert "side[tiab]" not in query
    # "statins" or "elderly" (both specific) should anchor the drugs_part.
    assert "statins[tiab]" in query or "elderly[tiab]" in query


def test_fallback_query_passes_validation_for_drug_interaction_intent() -> None:
    llm_client = Mock()
    planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())

    queries = planner._fallback_keyword_query("Can I take warfarin with aspirin?", ["warfarin", "aspirin"], intent="drug_interaction")

    is_valid, error = validate_pubmed_query(queries[0].pubmed_query, "drug_interaction", ["warfarin", "aspirin"], "Can I take warfarin with aspirin?")
    assert is_valid, f"Fallback query failed validation: {error}"


def test_plan_uses_intent_in_fallback_when_all_llm_queries_rejected() -> None:
    llm_client = Mock()
    payload = {
        "normalized_question": "warfarin and aspirin interaction",
        "intent": "drug_interaction",
        "drugs_identified": ["warfarin", "aspirin"],
        "sub_questions": [],
        "planned_queries": [
            {"pubmed_query": "bad query no tags", "sub_question_index": 0, "strategy_label": "invalid"}
        ],
        "reasoning": "test",
    }
    llm_client.generate.return_value = json.dumps(payload)
    planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())

    plan = planner.plan("warfarin and aspirin interaction")

    assert plan.planned_queries[0].strategy_label == "fallback"
    assert "interaction" in plan.planned_queries[0].pubmed_query.lower()


def test_validate_pubmed_query_balanced_parens() -> None:
    assert validate_pubmed_query("(warfarin[tiab]", "unknown", [], "q") == (False, "Unbalanced parentheses")


def test_validate_pubmed_query_missing_english_filter() -> None:
    valid, message = validate_pubmed_query("(warfarin[tiab])", "unknown", [], "q")
    assert valid is False
    assert "English[Language]" in str(message)


def test_validate_pubmed_query_valid_query() -> None:
    assert validate_pubmed_query(
        "(warfarin[tiab]) AND English[Language] AND Humans[Mesh]",
        "unknown",
        [],
        "q",
    ) == (True, None)


def test_validate_pubmed_query_empty_query() -> None:
    assert validate_pubmed_query("", "unknown", [], "q") == (False, "Query is empty")


# --- Model compatibility tests (Comment 2) ---


def test_normalize_planner_model_default_config_resolves_to_supported_70b() -> None:
    """Default AgentConfig.planner_model ('70b') resolves to the supported 70B full ID."""
    result = _normalize_planner_model(AgentConfig.planner_model)
    supported_values = set(NVIDIALLMClient.MODELS) | set(NVIDIALLMClient.MODELS.values())
    assert result in supported_values


def test_normalize_planner_model_whitespace_only_falls_back_to_70b() -> None:
    assert _normalize_planner_model("   ") == "70b"


def test_normalize_planner_model_none_falls_back_to_70b() -> None:
    assert _normalize_planner_model(None) == "70b"


def test_normalize_planner_model_legacy_alias_maps_to_70b() -> None:
    """Legacy full model ID not in MODELS keys or values maps to '70b'."""
    assert _normalize_planner_model("nvidia/llama-3.1-nemotron-70b-instruct") == "70b"


def test_normalize_planner_model_supported_shorthand_passes_through() -> None:
    assert _normalize_planner_model("8b") == "8b"
    assert _normalize_planner_model("70b") == "70b"


def test_normalize_planner_model_supported_full_id_passes_through() -> None:
    assert _normalize_planner_model("meta/llama-3.3-70b-instruct") == "meta/llama-3.3-70b-instruct"
    assert _normalize_planner_model("meta/llama-3.1-8b-instruct") == "meta/llama-3.1-8b-instruct"


def test_planner_resolves_default_config_to_supported_model_before_generate() -> None:
    """QueryPlanner with default AgentConfig sends a client-supported model to generate()."""
    llm_client = Mock()
    llm_client.generate.return_value = _load_fixture_text("agent_planner_response_wrapped.txt")
    planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())

    planner.plan("What are aspirin side effects?")

    model_sent = llm_client.generate.call_args.kwargs["model"]
    supported = set(NVIDIALLMClient.MODELS) | set(NVIDIALLMClient.MODELS.values())
    assert model_sent in supported, f"generate() received unsupported model: {model_sent!r}"


# --- _normalize_str_list unit tests ---


def test_normalize_str_list_bare_string_yields_single_item_list() -> None:
    assert _normalize_str_list("warfarin") == ["warfarin"]


def test_normalize_str_list_bare_string_whitespace_only_yields_empty() -> None:
    assert _normalize_str_list("   ") == []


def test_normalize_str_list_empty_string_yields_empty() -> None:
    assert _normalize_str_list("") == []


def test_normalize_str_list_list_of_strings_filters_empty() -> None:
    assert _normalize_str_list(["warfarin", "", "  ", "aspirin"]) == ["warfarin", "aspirin"]


def test_normalize_str_list_tuple_accepted_like_list() -> None:
    assert _normalize_str_list(("warfarin", "fluconazole")) == ["warfarin", "fluconazole"]


def test_normalize_str_list_none_yields_empty() -> None:
    assert _normalize_str_list(None) == []


def test_normalize_str_list_integer_yields_empty() -> None:
    assert _normalize_str_list(42) == []


# --- drugs_identified normalization integration tests ---


def test_plan_normalizes_string_drugs_identified_to_single_item_list() -> None:
    """LLM returning drugs_identified as a bare string must not be split into characters."""
    llm_client = Mock()
    payload = {
        "normalized_question": "warfarin question",
        "intent": "drug_interaction",
        "drugs_identified": "warfarin",
        "sub_questions": [],
        "planned_queries": [
            {
                "pubmed_query": "(warfarin[tiab]) AND English[Language] AND Humans[Mesh]",
                "sub_question_index": 0,
                "strategy_label": "broad",
            }
        ],
        "reasoning": "r",
    }
    llm_client.generate.return_value = json.dumps(payload)
    planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())

    plan = planner.plan("warfarin question")

    assert plan.drugs_identified == ["warfarin"], (
        "String drugs_identified must become a single-item list, not a character list"
    )


def test_plan_normalizes_list_drugs_identified_filters_empty_entries() -> None:
    """List-valued drugs_identified should filter out empty strings."""
    llm_client = Mock()
    payload = {
        "normalized_question": "interaction question",
        "intent": "drug_interaction",
        "drugs_identified": ["warfarin", "", "aspirin"],
        "sub_questions": [],
        "planned_queries": [
            {
                "pubmed_query": "(warfarin[tiab]) AND English[Language] AND Humans[Mesh]",
                "sub_question_index": 0,
                "strategy_label": "broad",
            }
        ],
        "reasoning": "r",
    }
    llm_client.generate.return_value = json.dumps(payload)
    planner = QueryPlanner(llm_client=llm_client, config=AgentConfig())

    plan = planner.plan("warfarin and aspirin interaction")

    assert plan.drugs_identified == ["warfarin", "aspirin"]
