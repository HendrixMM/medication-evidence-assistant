"""Unit tests for agent session utilities."""
import json
import os
import uuid

import pytest

from src.agent.config import AgentConfig
from src.agent.models import AgentStopReason
from src.agent.models import ConsumerAnswer
from src.agent.models import EvidenceLevel
from src.agent.models import SessionState
from src.agent.models import Source
from src.agent.models import TurnRecord
from src.agent.session import SessionManager
from src.agent.session import TranscriptStore


def _turn_record(turn_number: int) -> TurnRecord:
    return TurnRecord(
        turn_number=turn_number,
        tool_invocations=[f"tool-{turn_number}"],
        evaluation=None,
        tokens_used=turn_number * 10,
        decision="continue",
        num_articles=turn_number,
        num_relevant=max(turn_number - 1, 0),
        study_type_counts={"rct": turn_number},
        latency_breakdown_ms={"pubmed": float(turn_number)},
    )


def _consumer_answer() -> ConsumerAnswer:
    return ConsumerAnswer(
        answer="Use caution.",
        sources=[
            Source(
                pmid="123456",
                title="Study",
                authors=["A. Author"],
                journal=None,
                year=None,
                url=None,
                relevance_score=4.0,
                study_type="rct",
                snippet=None,
            )
        ],
        evidence_level=EvidenceLevel.rct,
        uncertainties=["Small sample size"],
        safety_warnings=["Consult a clinician"],
        disclaimer="Educational information only.",
    )


def test_transcript_store_append_and_replay() -> None:
    store = TranscriptStore()
    first = _turn_record(1)
    second = _turn_record(2)

    store.append(first)
    store.append(second)
    replayed = store.replay()

    assert isinstance(replayed, tuple)
    assert replayed == (first, second)


def test_transcript_store_isolates_records_from_input_and_replay_mutation() -> None:
    store = TranscriptStore()
    record = _turn_record(1)

    store.append(record)
    record.tool_invocations.append("mutated-after-append")

    replayed = store.replay()
    replayed[0].tool_invocations.append("mutated-after-replay")

    stored = store.replay()[0]

    assert stored.tool_invocations == ["tool-1"]


def test_transcript_store_compact_keeps_last_records() -> None:
    store = TranscriptStore()
    for index in range(1, 26):
        store.append(_turn_record(index))

    store.compact(keep_last=20)

    assert len(store) == 20
    assert store.replay()[-1].turn_number == 25


def test_transcript_store_compact_zero_clears_records() -> None:
    store = TranscriptStore()
    store.append(_turn_record(1))
    store.append(_turn_record(2))

    store.compact(keep_last=0)

    assert len(store) == 0
    assert store.replay() == ()


def test_transcript_store_compact_negative_raises_value_error() -> None:
    store = TranscriptStore()
    store.append(_turn_record(1))

    with pytest.raises(ValueError, match="keep_last must be non-negative"):
        store.compact(keep_last=-1)


def test_transcript_store_flush_returns_records_and_clears_store() -> None:
    store = TranscriptStore()
    store.append(_turn_record(1))
    store.append(_turn_record(2))

    flushed = store.flush()

    assert len(flushed) == 2
    assert len(store) == 0


def test_session_manager_create_session_creates_json_file(tmp_path) -> None:
    manager = SessionManager(AgentConfig(session_dir=str(tmp_path)))

    session = manager.create_session("What are the side effects of warfarin?")
    session_path = tmp_path / f"{session.session_id}.json"

    uuid.UUID(session.session_id)
    assert session_path.exists()
    assert json.loads(session_path.read_text(encoding="utf-8"))["query"] == session.query


def test_session_manager_load_session_round_trips_query(tmp_path) -> None:
    manager = SessionManager(AgentConfig(session_dir=str(tmp_path)))
    created = manager.create_session("Can tacrolimus interact with ritonavir?")

    loaded = manager.load_session(created.session_id)

    assert loaded is not None
    assert loaded.query == created.query


def test_session_manager_load_session_supports_non_uuid_ids(tmp_path) -> None:
    manager = SessionManager(AgentConfig(session_dir=str(tmp_path)))
    session = SessionState(session_id="session-alpha_1.2", query="q")

    manager.save_session(session)
    loaded = manager.load_session("session-alpha_1.2")

    assert loaded is not None
    assert loaded.session_id == "session-alpha_1.2"


def test_session_manager_load_session_missing_returns_none(tmp_path) -> None:
    manager = SessionManager(AgentConfig(session_dir=str(tmp_path)))

    assert manager.load_session("missing-session-id") is None


def test_session_manager_load_session_does_not_expire_stale_files_by_default(tmp_path) -> None:
    manager = SessionManager(AgentConfig(session_dir=str(tmp_path), session_ttl_hours=1))
    created = manager.create_session("persistent query")
    session_path = tmp_path / f"{created.session_id}.json"
    stale_timestamp = session_path.stat().st_mtime - (2 * 3600)
    os.utime(session_path, (stale_timestamp, stale_timestamp))

    loaded = manager.load_session(created.session_id)

    assert loaded is not None
    assert session_path.exists()


def test_session_manager_load_session_expires_stale_files_when_enabled(tmp_path) -> None:
    manager = SessionManager(
        AgentConfig(session_dir=str(tmp_path), session_ttl_hours=1, enable_session_expiry=True)
    )
    created = manager.create_session("persistent query")
    session_path = tmp_path / f"{created.session_id}.json"
    stale_timestamp = session_path.stat().st_mtime - (2 * 3600)
    os.utime(session_path, (stale_timestamp, stale_timestamp))

    loaded = manager.load_session(created.session_id)

    assert loaded is None
    assert not session_path.exists()


def test_session_manager_load_session_keeps_recent_files_within_enabled_ttl(tmp_path) -> None:
    manager = SessionManager(
        AgentConfig(session_dir=str(tmp_path), session_ttl_hours=24, enable_session_expiry=True)
    )
    created = manager.create_session("recent query")

    loaded = manager.load_session(created.session_id)

    assert loaded is not None
    assert loaded.query == created.query


def test_session_manager_load_session_rejects_path_traversal_ids(tmp_path) -> None:
    manager = SessionManager(AgentConfig(session_dir=str(tmp_path)))

    assert manager.load_session("../outside") is None


def test_session_manager_resolves_relative_session_dir_from_explicit_base_dir(
    tmp_path, monkeypatch
) -> None:
    session_base_dir = tmp_path / "session-base"
    launch_dir_one = tmp_path / "launch-one"
    launch_dir_two = tmp_path / "launch-two"
    session_base_dir.mkdir()
    launch_dir_one.mkdir()
    launch_dir_two.mkdir()

    monkeypatch.chdir(launch_dir_one)
    first_manager = SessionManager(
        AgentConfig(session_dir=".sessions", session_base_dir=str(session_base_dir))
    )
    session = first_manager.create_session("persistent query")

    first_expected_path = session_base_dir / ".sessions" / f"{session.session_id}.json"
    assert first_expected_path.exists()
    assert first_manager._session_dir == first_expected_path.parent.resolve()

    monkeypatch.chdir(launch_dir_two)
    second_manager = SessionManager(
        AgentConfig(session_dir=".sessions", session_base_dir=str(session_base_dir))
    )

    assert second_manager._session_dir == (session_base_dir / ".sessions").resolve()
    loaded = second_manager.load_session(session.session_id)
    assert loaded is not None
    assert loaded.session_id == session.session_id


def test_session_manager_keeps_absolute_session_dir_unchanged(tmp_path, monkeypatch) -> None:
    launch_dir = tmp_path / "launch-dir"
    absolute_session_dir = tmp_path / "explicit-sessions"
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)

    manager = SessionManager(AgentConfig(session_dir=str(absolute_session_dir)))
    session = manager.create_session("absolute path query")

    assert (absolute_session_dir / f"{session.session_id}.json").exists()
    assert manager._session_dir == absolute_session_dir.resolve()


def test_session_manager_relative_session_dir_does_not_use_package_install_path(tmp_path, monkeypatch) -> None:
    package_like_dir = tmp_path / "readonly-site-packages" / "src" / "agent"
    session_base_dir = tmp_path / "session-base"
    runtime_dir = tmp_path / "runtime"
    package_like_dir.mkdir(parents=True)
    session_base_dir.mkdir()
    runtime_dir.mkdir()
    monkeypatch.chdir(runtime_dir)

    manager = SessionManager(
        AgentConfig(session_dir=".sessions", session_base_dir=str(session_base_dir))
    )

    assert manager._session_dir == (session_base_dir / ".sessions").resolve()
    assert package_like_dir / ".sessions" != manager._session_dir


def test_session_manager_relative_session_dir_falls_back_to_runtime_working_directory(
    tmp_path, monkeypatch
) -> None:
    launch_dir = tmp_path / "launch-dir"
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)

    manager = SessionManager(AgentConfig(session_dir=".sessions"))
    session = manager.create_session("runtime relative query")

    expected_path = launch_dir / ".sessions" / f"{session.session_id}.json"
    assert expected_path.exists()
    assert manager._session_dir == expected_path.parent.resolve()


def test_session_manager_save_session_does_not_leave_tmp_file(tmp_path) -> None:
    manager = SessionManager(AgentConfig(session_dir=str(tmp_path)))
    session = SessionState(session_id=str(uuid.uuid4()), query="q")

    manager.save_session(session)

    assert not (tmp_path / f"{session.session_id}.json.tmp").exists()


def test_session_manager_save_session_overwrites_existing_file(tmp_path) -> None:
    manager = SessionManager(AgentConfig(session_dir=str(tmp_path)))
    session = manager.create_session("initial")
    session.query = "updated"

    manager.save_session(session)
    loaded = manager.load_session(session.session_id)

    assert loaded is not None
    assert loaded.query == "updated"


def test_session_manager_save_session_rejects_invalid_session_id(tmp_path) -> None:
    manager = SessionManager(AgentConfig(session_dir=str(tmp_path)))
    session = SessionState(session_id="../outside", query="q")

    with pytest.raises(ValueError):
        manager.save_session(session)


def test_session_manager_save_session_rejects_unsafe_non_uuid_session_id(tmp_path) -> None:
    manager = SessionManager(AgentConfig(session_dir=str(tmp_path)))
    session = SessionState(session_id="session/unsafe", query="q")

    with pytest.raises(ValueError):
        manager.save_session(session)


def test_session_manager_is_within_budget_enforces_limits(tmp_path) -> None:
    manager = SessionManager(
        AgentConfig(session_dir=str(tmp_path), max_pubmed_calls=5, max_total_tokens=100)
    )
    session = SessionState(session_id=str(uuid.uuid4()), query="q")

    session.usage.pubmed_calls = 5
    assert manager.is_within_budget(session) is False

    session.usage.pubmed_calls = 0
    session.usage.input_tokens = 60
    session.usage.output_tokens = 40
    assert manager.is_within_budget(session) is False

    session.usage.output_tokens = 39
    assert manager.is_within_budget(session) is True


def test_session_manager_is_within_budget_supports_pending_deltas(tmp_path) -> None:
    manager = SessionManager(
        AgentConfig(session_dir=str(tmp_path), max_pubmed_calls=5, max_total_tokens=100)
    )
    session = SessionState(session_id=str(uuid.uuid4()), query="q")
    session.usage.pubmed_calls = 4
    session.usage.input_tokens = 60
    session.usage.output_tokens = 39

    assert manager.is_within_budget(session, pending_pubmed_calls=1, pending_tokens=1) is False
    assert manager.is_within_budget(session, pending_pubmed_calls=1) is False
    assert manager.is_within_budget(session, pending_tokens=1) is False
    assert manager.is_within_budget(session, pending_pubmed_calls=2) is False
    assert manager.is_within_budget(session, pending_tokens=2) is False


def test_session_manager_record_turn_increments_usage_counters(tmp_path) -> None:
    manager = SessionManager(AgentConfig(session_dir=str(tmp_path)))
    session = manager.create_session("Does this interaction need monitoring?")
    record = TurnRecord(
        turn_number=1,
        tool_invocations=["pubmed.search", "llm.generate", "translator.generate"],
        evaluation=None,
        tokens_used=25,
        decision="continue",
        num_articles=3,
        num_relevant=2,
        study_type_counts={"rct": 1},
        latency_breakdown_ms={"pubmed": 10.0, "llm": 40.0},
    )

    manager.record_turn(session, record)

    assert session.usage.output_tokens == 25
    assert session.usage.pubmed_calls == 1
    assert session.usage.llm_calls == 2
    assert session.usage.total_latency_ms == 50.0


def test_session_manager_record_turn_prefers_explicit_llm_call_counts(tmp_path) -> None:
    manager = SessionManager(AgentConfig(session_dir=str(tmp_path)))
    session = manager.create_session("Does the planner call count get tracked?")
    record = TurnRecord(
        turn_number=1,
        tool_invocations=["pubmed.search"],
        evaluation=None,
        tokens_used=42,
        decision="continue",
        num_articles=2,
        num_relevant=1,
        study_type_counts={"review": 1},
        latency_breakdown_ms={"pubmed": 5.0},
    )

    manager.record_turn(session, record, pubmed_calls=1, llm_calls=1)

    assert session.usage.output_tokens == 42
    assert session.usage.pubmed_calls == 1
    assert session.usage.llm_calls == 1


def test_session_manager_record_turn_can_exhaust_pubmed_budget(tmp_path) -> None:
    manager = SessionManager(AgentConfig(session_dir=str(tmp_path), max_pubmed_calls=2))
    session = manager.create_session("Is this combination safe?")

    manager.record_turn(
        session,
        TurnRecord(
            turn_number=1,
            tool_invocations=["pubmed.search"],
            evaluation=None,
            tokens_used=10,
            decision="continue",
            num_articles=1,
            num_relevant=1,
            study_type_counts={"rct": 1},
            latency_breakdown_ms={"pubmed": 5.0},
        ),
    )
    assert manager.is_within_budget(session) is True

    manager.record_turn(
        session,
        TurnRecord(
            turn_number=2,
            tool_invocations=["pubmed.fetch"],
            evaluation=None,
            tokens_used=10,
            decision="continue",
            num_articles=1,
            num_relevant=1,
            study_type_counts={"rct": 1},
            latency_breakdown_ms={"pubmed": 5.0},
        ),
    )

    assert session.usage.pubmed_calls == 2
    assert manager.is_within_budget(session) is False


def test_session_manager_finalize_session_sets_and_persists_state(tmp_path) -> None:
    manager = SessionManager(AgentConfig(session_dir=str(tmp_path)))
    session = manager.create_session("Is dose adjustment needed?")

    manager.finalize_session(
        session=session,
        answer=_consumer_answer(),
        stop_reason=AgentStopReason.completed,
    )
    loaded = manager.load_session(session.session_id)

    assert loaded is not None
    assert loaded.stop_reason == AgentStopReason.completed
    assert loaded.answer is not None
    assert loaded.answer.answer == "Use caution."
    assert loaded.answer.sources[0].authors == ["A. Author"]
