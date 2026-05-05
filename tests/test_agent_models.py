"""Unit tests for agent models."""
import pytest
from pydantic import ValidationError

from src.agent.models import ArticleAssessment
from src.agent.models import EvaluationVerdict
from src.agent.models import PlannedQuery
from src.agent.models import QueryIntent
from src.agent.models import QueryPlan
from src.agent.models import SessionState
from src.agent.models import Source
from src.agent.models import StreamEvent
from src.agent.models import SubQuestionCoverage
from src.agent.models import SubQuestion
from src.agent.models import TurnRecord
from src.agent.models import UsageStats


def test_query_intent_serializes_as_string_and_round_trips() -> None:
    plan = QueryPlan(
        normalized_question="What interacts with warfarin?",
        intent=QueryIntent.drug_interaction,
        sub_questions=[],
        planned_queries=[],
        drugs_identified=["warfarin"],
        reasoning="Check known interaction evidence.",
    )

    dumped = plan.model_dump()
    restored = QueryPlan.model_validate(dumped)

    assert dumped["intent"] == "drug_interaction"
    assert restored.intent == QueryIntent.drug_interaction


def test_source_is_immutable() -> None:
    source = Source(
        pmid="123456",
        title="Interaction Study",
        authors=["A. Author"],
        journal=None,
        year=None,
        url=None,
        relevance_score=4.2,
        study_type="rct",
        snippet=None,
    )

    with pytest.raises(ValidationError):
        source.title = "Updated Title"


def test_source_authors_use_list_contract() -> None:
    source = Source(
        pmid="123456",
        title="Interaction Study",
        authors=["A. Author"],
        journal=None,
        year=None,
        url=None,
        relevance_score=4.2,
        study_type="rct",
        snippet=None,
    )

    assert isinstance(source.authors, list)
    assert source.authors == ["A. Author"]
    assert source.model_dump()["authors"] == ["A. Author"]

    with pytest.raises(ValidationError):
        source.authors = ["Replacement Author"]


def test_session_state_defaults() -> None:
    session = SessionState(session_id="x", query="q")

    assert session.turns == []
    assert session.plan is None
    assert session.usage == UsageStats()


def test_turn_record_round_trip() -> None:
    record = TurnRecord(
        turn_number=1,
        tool_invocations=["pubmed.search"],
        evaluation=None,
        tokens_used=128,
        decision="continue",
        num_articles=10,
        num_relevant=4,
        study_type_counts={"rct": 2, "cohort": 2},
        latency_breakdown_ms={"pubmed": 125.0},
    )

    restored = TurnRecord.model_validate(record.model_dump())

    assert restored == record


def test_stream_event_timestamp_defaults() -> None:
    event = StreamEvent(event_type="status", data={"step": "plan"})

    assert isinstance(event.timestamp, float)
    assert event.timestamp > 0


def test_stream_event_rejects_non_dict_payloads() -> None:
    with pytest.raises(ValidationError):
        StreamEvent(event_type="sources", data=["not", "a", "dict"])


def test_evaluation_verdict_recommendation_literals() -> None:
    valid_recommendations = ("sufficient", "broaden", "narrow", "pivot")
    for recommendation in valid_recommendations:
        verdict = EvaluationVerdict(
            is_sufficient=recommendation == "sufficient",
            relevant_articles=[],
            study_type_distribution={},
            coverage_gaps=[],
            recommendation=recommendation,
            flags=[],
        )
        assert verdict.recommendation == recommendation

    with pytest.raises(ValidationError):
        EvaluationVerdict(
            is_sufficient=False,
            relevant_articles=[],
            study_type_distribution={},
            coverage_gaps=[],
            recommendation="expand",
            flags=[],
        )


def test_evaluation_verdict_includes_retrieval_quality_fields() -> None:
    verdict = EvaluationVerdict(
        is_sufficient=False,
        relevant_articles=[],
        study_type_distribution={"rct": 1},
        coverage_gaps=["Dose-response evidence"],
        recommendation="broaden",
        flags=["needs_more_evidence"],
        article_assessments=[
            ArticleAssessment(
                pmid="123456",
                direct_match_score=0.9,
                semantic_relevance_score=4.1,
                quality_score=0.8,
                covered_subquestions=[0],
                assessment_label="direct",
            )
        ],
        subquestion_coverages=[
            SubQuestionCoverage(
                sub_question_index=0,
                sub_question_text="What is the exposure change?",
                coverage_score=0.75,
                matched_pmids=["123456"],
                covered=True,
            )
        ],
        failure_modes=["insufficient_direct_evidence"],
        retrieval_score=0.42,
    )

    dumped = verdict.model_dump()
    restored = EvaluationVerdict.model_validate(dumped)

    assert dumped["article_assessments"][0]["pmid"] == "123456"
    assert dumped["article_assessments"][0]["assessment_label"] == "direct"
    assert dumped["subquestion_coverages"][0]["matched_pmids"] == ["123456"]
    assert dumped["failure_modes"] == ["insufficient_direct_evidence"]
    assert dumped["retrieval_score"] == 0.42
    assert restored == verdict


def test_evaluation_verdict_defaults_new_fields_for_backward_compat() -> None:
    verdict = EvaluationVerdict(
        is_sufficient=True,
        relevant_articles=[],
        study_type_distribution={},
        coverage_gaps=[],
        recommendation="sufficient",
        flags=[],
    )

    assert verdict.article_assessments == []
    assert verdict.subquestion_coverages == []
    assert verdict.failure_modes == []
    assert verdict.retrieval_score == 0.0


def test_article_assessment_is_frozen() -> None:
    assessment = ArticleAssessment(
        pmid="123456",
        direct_match_score=0.9,
        semantic_relevance_score=4.1,
        quality_score=0.8,
        covered_subquestions=[0],
        assessment_label="direct",
    )

    with pytest.raises(ValidationError):
        assessment.pmid = "654321"


def test_subquestion_coverage_is_frozen() -> None:
    coverage = SubQuestionCoverage(
        sub_question_index=0,
        sub_question_text="What is the exposure change?",
        coverage_score=0.75,
        matched_pmids=["123456"],
        covered=True,
    )

    with pytest.raises(ValidationError):
        coverage.covered = False


def test_query_plan_with_sub_questions_is_accessible_and_serializable() -> None:
    plan = QueryPlan(
        normalized_question="Can ritonavir affect tacrolimus exposure?",
        intent=QueryIntent.drug_interaction,
        sub_questions=[
            SubQuestion(
                text="What is the exposure change?",
                rationale="Assess PK magnitude.",
                priority=1,
            )
        ],
        planned_queries=[
            PlannedQuery(
                pubmed_query="ritonavir tacrolimus pharmacokinetics",
                sub_question_index=0,
                strategy_label="pk-focused",
            )
        ],
        drugs_identified=["ritonavir", "tacrolimus"],
        reasoning="Focus on interaction magnitude and mechanism.",
    )

    dumped = plan.model_dump()

    assert plan.drugs_identified == ["ritonavir", "tacrolimus"]
    assert plan.sub_questions[0].text == "What is the exposure change?"
    assert dumped["drugs_identified"] == ["ritonavir", "tacrolimus"]
