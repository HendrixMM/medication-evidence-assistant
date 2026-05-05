"""Unit tests for the agent loop."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.agent.config import AgentConfig
from src.agent.loop import AgentLoop
from src.agent.models import AgentStopReason
from src.agent.models import ConsumerAnswer
from src.agent.models import EvaluationVerdict
from src.agent.models import EvidenceLevel
from src.agent.models import EvidenceSummary
from src.agent.models import PlannedQuery
from src.agent.models import QueryIntent
from src.agent.models import QueryPlan
from src.agent.models import RetrievalResult
from src.agent.models import RetrievedArticle
from src.agent.models import Source
from src.agent.models import SubQuestion
from src.agent.models import SubQuestionCoverage
from src.agent.session import SessionManager

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_json_fixture(name: str):
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _article(pmid: str) -> RetrievedArticle:
    base = RetrievedArticle.model_validate(_load_json_fixture("agent_pubmed_articles.json")[0])
    return base.model_copy(update={"pmid": pmid})


def _plan() -> QueryPlan:
    return QueryPlan.model_validate(_load_json_fixture("agent_planner_response.json"))


def _verdict(*, sufficient: bool) -> EvaluationVerdict:
    return EvaluationVerdict(
        is_sufficient=sufficient,
        relevant_articles=[_article("1")] if sufficient else [],
        study_type_distribution={"Review": 1},
        coverage_gaps=[] if sufficient else ["gap"],
        recommendation="sufficient" if sufficient else "broaden",
        flags=[],
        failure_modes=[] if sufficient else ["subquestion_gap"],
        subquestion_coverages=[] if sufficient else [
            SubQuestionCoverage(
                sub_question_index=0,
                sub_question_text="gap",
                coverage_score=0.0,
                matched_pmids=[],
                covered=False,
            )
        ],
    )


def _answer() -> ConsumerAnswer:
    return ConsumerAnswer.model_validate(_load_json_fixture("agent_consumer_answer.json"))


def _evidence(articles: list[RetrievedArticle] | None = None) -> EvidenceSummary:
    payload = _load_json_fixture("agent_evidence_summary.json")
    if articles is not None:
        payload["ranked_articles"] = [article.model_dump() for article in articles]
    return EvidenceSummary.model_validate(payload)


def _loop(tmp_path, *, max_turns: int = 3, api_key: str | None = None) -> tuple[AgentLoop, Mock, Mock, Mock, Mock, Mock, Mock]:
    config = AgentConfig(session_dir=str(tmp_path), max_agent_turns=max_turns, pubmed_api_key=api_key)
    planner = Mock()
    planner.plan.return_value = _plan()
    planner.refine_plan.return_value = _plan().planned_queries
    planner.llm_client = Mock()
    planner.llm_client.last_usage_tokens = 42
    evaluator = Mock()
    evaluator.evaluate.return_value = _verdict(sufficient=True)
    pubmed_tool = Mock()
    pubmed_tool.execute.return_value = RetrievalResult(
        tool_name="pubmed_search",
        query_used=_plan().planned_queries[0].pubmed_query,
        articles=[_article("1")],
        execution_time_ms=10.0,
        error=None,
    )
    summarize_tool = Mock()
    summarize_tool.execute.return_value = _evidence()
    translate_tool = Mock()
    translate_tool.execute.return_value = _answer()
    translate_tool.build_sources.return_value = _answer().sources
    guardrails = Mock()
    guardrails.validate_input.return_value = (True, None)
    guardrails.validate_output.side_effect = lambda answer: answer
    loop = AgentLoop(
        planner=planner,
        evaluator=evaluator,
        pubmed_tool=pubmed_tool,
        summarize_tool=summarize_tool,
        translate_tool=translate_tool,
        guardrails=guardrails,
        session_manager=SessionManager(config),
        config=config,
    )
    return loop, planner, evaluator, pubmed_tool, summarize_tool, translate_tool, guardrails


def test_answer_returns_consumer_answer(tmp_path) -> None:
    loop, *_ = _loop(tmp_path)

    answer = loop.answer("Is warfarin safe?")

    assert answer.answer
    assert answer.sources


def test_answer_ignores_nonterminal_error_when_stream_recovers(tmp_path) -> None:
    loop, planner, *_ = _loop(tmp_path)
    planner.plan.side_effect = RuntimeError("boom")
    planner._fallback_keyword_query = Mock(
        return_value=[
            PlannedQuery(
                pubmed_query="(warfarin[tiab]) AND English[Language] AND Humans[Mesh]",
                sub_question_index=0,
                strategy_label="fallback",
            )
        ]
    )

    answer = loop.answer("Is warfarin safe?")

    assert answer.answer == _answer().answer
    assert answer.sources == _answer().sources


def test_answer_returns_fallback_on_terminal_error_without_answer(tmp_path) -> None:
    loop, *_rest = _loop(tmp_path)
    pubmed_tool = _rest[2]
    pubmed_tool.execute.side_effect = RuntimeError("boom")

    answer = loop.answer("Is warfarin safe?")

    assert answer.answer.startswith("I found relevant studies")
    assert answer.sources == []


def test_answer_returns_guardrail_block_message_instead_of_fallback(tmp_path) -> None:
    loop, *_rest = _loop(tmp_path)
    guardrails = _rest[-1]
    block_message = "⚠️ We cannot provide information on this topic. Please contact your healthcare provider."
    guardrails.validate_input.return_value = (False, block_message)

    answer = loop.answer("Can I self-medicate?")

    assert answer.answer == block_message
    assert answer.safety_warnings == [block_message]
    assert answer.verdict == AgentStopReason.blocked.value


@pytest.mark.asyncio
async def test_answer_streaming_emits_started_and_finished(tmp_path) -> None:
    loop, *_ = _loop(tmp_path)

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    assert any("event: started" in event for event in events)
    assert any("event: finished" in event for event in events)


@pytest.mark.asyncio
async def test_answer_streaming_emits_sources_event(tmp_path) -> None:
    loop, *_ = _loop(tmp_path)

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    sources_event = next(event for event in events if "event: sources" in event)
    payload = json.loads(sources_event.splitlines()[1].split(": ", 1)[1])

    assert isinstance(payload, list)
    assert payload


@pytest.mark.asyncio
async def test_answer_streaming_blocks_on_safety_check(tmp_path) -> None:
    loop, *_rest = _loop(tmp_path)
    guardrails = _rest[-1]
    guardrails.validate_input.return_value = (False, "blocked")

    events = [event async for event in loop.answer_streaming("emergency question")]

    assert any("event: error" in event for event in events)
    assert any(AgentStopReason.blocked.value in event for event in events if "event: finished" in event)


@pytest.mark.asyncio
async def test_answer_streaming_deduplicates_articles_across_turns(tmp_path) -> None:
    loop, planner, evaluator, pubmed_tool, summarize_tool, *_ = _loop(tmp_path, max_turns=2)
    evaluator.evaluate.side_effect = [_verdict(sufficient=False), _verdict(sufficient=True)]
    pubmed_tool.execute.side_effect = [
        RetrievalResult(tool_name="pubmed_search", query_used="q1", articles=[_article("1")], execution_time_ms=5.0),
        RetrievalResult(
            tool_name="pubmed_search",
            query_used="q2",
            articles=[_article("1"), _article("2")],
            execution_time_ms=5.0,
        ),
    ]

    [event async for event in loop.answer_streaming("Is warfarin safe?")]

    articles = summarize_tool.execute.call_args.args[0]
    assert sorted(article.pmid for article in articles) == ["1", "2"]


@pytest.mark.asyncio
async def test_answer_streaming_relevant_first_reordering_keeps_distinct_empty_pmid_articles(tmp_path) -> None:
    loop, _planner, evaluator, pubmed_tool, summarize_tool, *_ = _loop(tmp_path)
    article_a = _article("").model_copy(update={"title": "Article A", "publication_date": "2023-01-01"})
    article_b = _article("").model_copy(update={"title": "Article B", "publication_date": "2024-01-01"})
    article_c = _article("").model_copy(update={"title": "Article C", "publication_date": "2025-01-01"})
    evaluator.evaluate.return_value = EvaluationVerdict(
        is_sufficient=True,
        relevant_articles=[article_b],
        study_type_distribution={"Review": 3},
        coverage_gaps=[],
        recommendation="sufficient",
        flags=[],
        failure_modes=[],
        subquestion_coverages=[],
    )
    pubmed_tool.execute.return_value = RetrievalResult(
        tool_name="pubmed_search",
        query_used=_plan().planned_queries[0].pubmed_query,
        articles=[article_a, article_b, article_c],
        execution_time_ms=10.0,
        error=None,
    )

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    articles = summarize_tool.execute.call_args.args[0]
    assert [(article.title, article.publication_date) for article in articles] == [
        ("Article B", "2024-01-01"),
        ("Article A", "2023-01-01"),
        ("Article C", "2025-01-01"),
    ]
    finished = next(event for event in events if "event: finished" in event)
    session_id = json.loads(finished.splitlines()[1].split(": ", 1)[1])["session_id"]
    session = loop.session_manager.load_session(session_id)
    assert session is not None
    assert [(article.title, article.publication_date) for article in session.all_articles] == [
        ("Article B", "2024-01-01"),
        ("Article A", "2023-01-01"),
        ("Article C", "2025-01-01"),
    ]


@pytest.mark.asyncio
async def test_answer_streaming_stops_at_max_turns(tmp_path) -> None:
    loop, planner, evaluator, pubmed_tool, *_ = _loop(tmp_path, max_turns=2)
    evaluator.evaluate.return_value = _verdict(sufficient=False)
    planner.refine_plan.return_value = _plan().planned_queries

    [event async for event in loop.answer_streaming("Is warfarin safe?")]

    assert pubmed_tool.execute.call_count == 2


@pytest.mark.asyncio
async def test_answer_streaming_stops_early_when_sufficient(tmp_path) -> None:
    loop, *_rest = _loop(tmp_path)
    pubmed_tool = _rest[2]

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    assert pubmed_tool.execute.call_count == 1
    finished = next(event for event in events if "event: finished" in event)
    session_id = json.loads(finished.splitlines()[1].split(": ", 1)[1])["session_id"]
    session = loop.session_manager.load_session(session_id)
    assert session is not None
    assert session.usage.llm_calls == 2


@pytest.mark.asyncio
async def test_agent_loop_records_failure_modes_in_turn_record(tmp_path) -> None:
    loop, _planner, evaluator, *_ = _loop(tmp_path, max_turns=2)
    evaluator.evaluate.side_effect = [_verdict(sufficient=False), _verdict(sufficient=True)]

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    finished = next(event for event in events if "event: finished" in event)
    session_id = json.loads(finished.splitlines()[1].split(": ", 1)[1])["session_id"]
    session = loop.session_manager.load_session(session_id)

    assert session is not None
    assert session.turns[0].evaluation is not None
    assert session.turns[0].evaluation.failure_modes == ["subquestion_gap"]


@pytest.mark.asyncio
async def test_answer_streaming_handles_planner_failure(tmp_path) -> None:
    loop, planner, *_ = _loop(tmp_path)
    planner.plan.side_effect = RuntimeError("boom")
    planner._fallback_keyword_query = Mock(
        return_value=[PlannedQuery(pubmed_query="(medication[tiab]) AND English[Language] AND Humans[Mesh]", sub_question_index=0, strategy_label="fallback")]
    )

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    assert any("event: error" in event and "planner_failure" in event for event in events)
    assert any("event: finished" in event for event in events)


@pytest.mark.asyncio
async def test_answer_streaming_handles_pubmed_failure(tmp_path) -> None:
    loop, *_rest = _loop(tmp_path)
    pubmed_tool = _rest[2]
    pubmed_tool.execute.side_effect = RuntimeError("boom")

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    assert any("event: error" in event for event in events)
    assert any(AgentStopReason.error.value in event for event in events if "event: finished" in event)
    finished = next(event for event in events if "event: finished" in event)
    session_id = json.loads(finished.splitlines()[1].split(": ", 1)[1])["session_id"]
    session = loop.session_manager.load_session(session_id)
    assert session is not None
    assert session.stop_reason == AgentStopReason.error


@pytest.mark.asyncio
async def test_answer_streaming_handles_pubmed_result_error(tmp_path) -> None:
    loop, *_rest = _loop(tmp_path)
    pubmed_tool = _rest[2]
    pubmed_tool.execute.return_value = RetrievalResult(
        tool_name="pubmed_search",
        query_used="q1",
        articles=[],
        execution_time_ms=5.0,
        error="backend unavailable",
    )

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    assert any("event: error" in event for event in events)
    assert any(AgentStopReason.error.value in event for event in events if "event: finished" in event)


@pytest.mark.asyncio
async def test_answer_streaming_caps_pubmed_batch_to_remaining_budget(tmp_path) -> None:
    loop, planner, evaluator, pubmed_tool, *_ = _loop(tmp_path, max_turns=2)
    loop.config.max_pubmed_calls = 1
    planner.plan.return_value = _plan().model_copy(
        update={
            "planned_queries": [
                PlannedQuery(pubmed_query="query one", sub_question_index=0, strategy_label="primary"),
                PlannedQuery(pubmed_query="query two", sub_question_index=1, strategy_label="secondary"),
            ]
        }
    )
    evaluator.evaluate.return_value = _verdict(sufficient=False)
    planner.refine_plan.return_value = [
        PlannedQuery(pubmed_query="refined query", sub_question_index=0, strategy_label="refine")
    ]

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    assert pubmed_tool.execute.call_count == 1
    finished = next(event for event in events if "event: finished" in event)
    payload = json.loads(finished.splitlines()[1].split(": ", 1)[1])
    assert payload["stop_reason"] == AgentStopReason.budget_exhausted.value
    session = loop.session_manager.load_session(payload["session_id"])
    assert session is not None
    assert session.usage.pubmed_calls == 1
    assert session.stop_reason == AgentStopReason.budget_exhausted


@pytest.mark.asyncio
async def test_answer_streaming_handles_translator_failure(tmp_path) -> None:
    loop, *_rest = _loop(tmp_path)
    translate_tool = _rest[4]
    translate_tool.execute.side_effect = RuntimeError("boom")

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    assert any("event: error" in event and "translator_failure" in event for event in events)
    assert any("I found relevant studies" in event for event in events if "event: answer_delta" in event)
    finished = next(event for event in events if "event: finished" in event)
    payload = json.loads(finished.splitlines()[1].split(": ", 1)[1])
    assert payload["stop_reason"] == AgentStopReason.error.value
    session = loop.session_manager.load_session(payload["session_id"])
    assert session is not None
    assert session.stop_reason == AgentStopReason.error


@pytest.mark.asyncio
async def test_answer_streaming_finishes_when_output_validation_raises(tmp_path) -> None:
    loop, *_rest = _loop(tmp_path)
    guardrails = _rest[-1]
    guardrails.validate_output.side_effect = RuntimeError("boom")

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    assert any("event: finished" in event for event in events)


def test_answer_parses_raw_sources_payload(tmp_path) -> None:
    loop, *_ = _loop(tmp_path)

    answer = loop.answer("Is warfarin safe?")

    assert answer.sources


def test_ncbi_semaphore_capacity_with_api_key(tmp_path) -> None:
    loop, *_ = _loop(tmp_path, api_key="key")

    assert loop._ncbi_semaphore_capacity == 10


def test_ncbi_semaphore_capacity_without_api_key(tmp_path) -> None:
    loop, *_ = _loop(tmp_path, api_key=None)

    assert loop._ncbi_semaphore_capacity == 3


def test_answer_can_be_called_twice_on_same_instance(tmp_path) -> None:
    loop, *_ = _loop(tmp_path)

    first = loop.answer("Is warfarin safe?")
    second = loop.answer("Is ibuprofen safe?")

    assert first.answer
    assert second.answer


@pytest.mark.asyncio
async def test_answer_streaming_does_not_emit_consumer_answer_event(tmp_path) -> None:
    loop, *_ = _loop(tmp_path)

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    assert not any("event: consumer_answer" in event for event in events)


@pytest.mark.asyncio
async def test_answer_streaming_routes_sources_through_streaming_adapter(tmp_path) -> None:
    loop, *_ = _loop(tmp_path)

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    sources_events = [event for event in events if "event: sources" in event]
    assert len(sources_events) == 1
    payload = json.loads(sources_events[0].splitlines()[1].split(": ", 1)[1])
    assert isinstance(payload, list)
    assert payload


@pytest.mark.asyncio
async def test_failed_planner_does_not_inflate_session_usage(tmp_path) -> None:
    loop, planner, *_ = _loop(tmp_path)
    planner.llm_client.last_usage_tokens = 100  # pre-populated; must not leak on failure
    planner.plan.side_effect = RuntimeError("boom")
    planner._fallback_keyword_query = Mock(
        return_value=[
            PlannedQuery(
                pubmed_query="(medication[tiab]) AND English[Language] AND Humans[Mesh]",
                sub_question_index=0,
                strategy_label="fallback",
            )
        ]
    )

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    finished = next(event for event in events if "event: finished" in event)
    payload = json.loads(finished.splitlines()[1].split(": ", 1)[1])
    session = loop.session_manager.load_session(payload["session_id"])
    assert session is not None
    assert session.usage.llm_calls == 1, "only translator succeeded; planner failure must not count"
    assert session.usage.output_tokens == 0, "stale last_usage_tokens must not leak into session on planner failure"


@pytest.mark.asyncio
async def test_failed_translator_does_not_inflate_session_usage(tmp_path) -> None:
    loop, *_rest = _loop(tmp_path)
    translate_tool = _rest[4]
    translate_tool.llm_client = Mock()
    translate_tool.llm_client.last_usage_tokens = 100  # pre-populated; must not leak on failure
    translate_tool.execute.side_effect = RuntimeError("boom")

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    finished = next(event for event in events if "event: finished" in event)
    payload = json.loads(finished.splitlines()[1].split(": ", 1)[1])
    session = loop.session_manager.load_session(payload["session_id"])
    assert session is not None
    assert session.usage.llm_calls == 1, "only planner succeeded; translator failure must not count"
    assert session.usage.output_tokens == 42, "planner tokens (42) are counted; translator stale 100 must not leak on failure"


@pytest.mark.asyncio
async def test_planner_tokens_attributed_per_call_not_delta(tmp_path) -> None:
    """Per-call attribution: successful planner tokens are added directly, not as a delta."""
    loop, planner, *_ = _loop(tmp_path)
    planner.llm_client.last_usage_tokens = 300

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    finished = next(event for event in events if "event: finished" in event)
    payload = json.loads(finished.splitlines()[1].split(": ", 1)[1])
    session = loop.session_manager.load_session(payload["session_id"])
    assert session is not None
    assert session.usage.output_tokens == 300, "planner per-call tokens must be attributed directly"


@pytest.mark.asyncio
async def test_evaluator_llm_usage_is_attributed_to_session_usage(tmp_path) -> None:
    loop, _planner, evaluator, *_ = _loop(tmp_path)

    def _evaluate_with_usage(*args, **kwargs):
        evaluator.last_assessment_output_tokens = 55
        evaluator.last_assessment_llm_calls = 1
        return _verdict(sufficient=True)

    evaluator.evaluate.side_effect = _evaluate_with_usage

    events = [event async for event in loop.answer_streaming("Is warfarin safe?")]

    finished = next(event for event in events if "event: finished" in event)
    payload = json.loads(finished.splitlines()[1].split(": ", 1)[1])
    session = loop.session_manager.load_session(payload["session_id"])
    assert session is not None
    assert session.usage.llm_calls == 3, "planner, evaluator, and translator calls must all be counted"
    assert session.usage.output_tokens == 97, "planner (42) plus evaluator (55) tokens must be counted"


def test_agent_mode_enabled_flag_defaults_true(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MODE_ENABLED", raising=False)
    module = importlib.import_module("src.agent.loop")
    module = importlib.reload(module)

    assert module.AGENT_MODE_ENABLED is True


def test_agent_mode_enabled_flag_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_MODE_ENABLED", "false")
    module = importlib.import_module("src.agent.loop")
    module = importlib.reload(module)

    assert module.AGENT_MODE_ENABLED is False


def test_agent_mode_enabled_flag_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_MODE_ENABLED", "true")
    module = importlib.import_module("src.agent.loop")
    module = importlib.reload(module)

    assert module.AGENT_MODE_ENABLED is True
