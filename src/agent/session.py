"""Session persistence utilities for the agent workflow."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import time
import uuid

from pydantic import ValidationError

from .config import AgentConfig
from .models import AgentStopReason, ConsumerAnswer, SessionState, TurnRecord

logger = logging.getLogger(__name__)


class TranscriptStore:
    """In-memory ordered turn transcript store."""

    def __init__(self) -> None:
        self._records: list[TurnRecord] = []

    def append(self, record: TurnRecord) -> None:
        self._records.append(record.model_copy(deep=True))

    def compact(self, keep_last: int = 20) -> None:
        if keep_last < 0:
            raise ValueError("keep_last must be non-negative")
        if keep_last == 0:
            self._records.clear()
            return
        if keep_last < len(self._records):
            self._records = self._records[-keep_last:]

    def replay(self) -> tuple[TurnRecord, ...]:
        return tuple(record.model_copy(deep=True) for record in self._records)

    def flush(self) -> list[TurnRecord]:
        records = [record.model_copy(deep=True) for record in self._records]
        self._records.clear()
        return records

    def __len__(self) -> int:
        return len(self._records)


class SessionManager:
    """Manage session lifecycle with JSON persistence.

    Session expiry is optional. When `AgentConfig.enable_session_expiry` is
    enabled, session age is evaluated from the backing file's modification
    time using `AgentConfig.session_ttl_hours`, and expired files are deleted
    lazily when encountered during `load_session`.
    """

    _SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._session_dir = self._resolve_session_dir(
            session_dir=config.session_dir,
            session_base_dir=config.session_base_dir,
        )
        self._session_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, query: str) -> SessionState:
        session = SessionState(session_id=str(uuid.uuid4()), query=query)
        self.save_session(session)
        return session

    def load_session(self, session_id: str) -> SessionState | None:
        try:
            path = self._session_path(session_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        if self.config.enable_session_expiry and self._is_expired(path):
            self._delete_expired_session(path, session_id)
            return None
        try:
            payload = path.read_text(encoding="utf-8")
            return SessionState.model_validate_json(payload)
        except OSError as exc:
            logger.warning("Failed to read session file for %s at %s: %s", session_id, path, exc)
            return None
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.warning("Failed to parse session file for %s at %s: %s", session_id, path, exc)
            return None

    def save_session(self, session: SessionState) -> None:
        path = self._session_path(session.session_id)
        temp_path = path.with_name(f"{path.name}.tmp")
        payload = session.model_dump_json(indent=2)
        temp_path.write_text(payload, encoding="utf-8")
        json.loads(payload)
        temp_path.replace(path)

    def record_turn(
        self,
        session: SessionState,
        record: TurnRecord,
        *,
        pubmed_calls: int | None = None,
        llm_calls: int | None = None,
    ) -> None:
        """Persist a turn and aggregate its usage counters.

        `record.tokens_used` is expected to include all LLM token usage attributed
        to this turn so budget checks and telemetry stay aligned with loop behavior.
        """
        session.turns.append(record)
        session.usage.output_tokens += record.tokens_used
        session.usage.pubmed_calls += self._resolve_call_count(
            explicit_count=pubmed_calls,
            tool_invocations=record.tool_invocations,
            tool_markers=("pubmed",),
        )
        session.usage.llm_calls += self._resolve_call_count(
            explicit_count=llm_calls,
            tool_invocations=record.tool_invocations,
            tool_markers=("llm", "planner", "translator", "evaluator"),
        )
        session.usage.total_latency_ms += sum(record.latency_breakdown_ms.values())
        self.save_session(session)

    def finalize_session(
        self,
        session: SessionState,
        answer: ConsumerAnswer,
        stop_reason: AgentStopReason,
    ) -> None:
        session.answer = answer
        session.stop_reason = stop_reason
        self.save_session(session)

    def is_within_budget(
        self,
        session: SessionState,
        *,
        pending_pubmed_calls: int = 0,
        pending_tokens: int = 0,
    ) -> bool:
        if pending_pubmed_calls < 0 or pending_tokens < 0:
            raise ValueError("pending budget deltas must be non-negative")
        total_pubmed_calls = session.usage.pubmed_calls + pending_pubmed_calls
        total_tokens = session.usage.input_tokens + session.usage.output_tokens + pending_tokens
        return (
            total_pubmed_calls < self.config.max_pubmed_calls
            and total_tokens < self.config.max_total_tokens
        )

    def _session_path(self, session_id: str) -> Path:
        validated_id = self._validate_session_id(session_id)
        path = (self._session_dir / f"{validated_id}.json").resolve()
        path.relative_to(self._session_dir)
        return path

    def _validate_session_id(self, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not self._SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError("session_id must contain only alphanumeric characters, '.', '_', or '-'")
        return session_id

    def _is_expired(self, path: Path) -> bool:
        ttl_seconds = self.config.session_ttl_hours * 3600
        if ttl_seconds <= 0:
            return False
        try:
            age_seconds = max(0.0, time.time() - path.stat().st_mtime)
        except OSError as exc:
            logger.warning("Failed to inspect session age at %s: %s", path, exc)
            return False
        return age_seconds > ttl_seconds

    @staticmethod
    def _delete_expired_session(path: Path, session_id: str) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to delete expired session file for %s at %s: %s", session_id, path, exc)
        else:
            logger.info("Deleted expired session file for %s at %s", session_id, path)

    @staticmethod
    def _resolve_session_dir(session_dir: str, session_base_dir: str | None = None) -> Path:
        path = Path(session_dir).expanduser()
        if not path.is_absolute():
            base_dir = Path(session_base_dir).expanduser() if session_base_dir else Path.cwd()
            path = base_dir / path
        return path.resolve()

    @staticmethod
    def _resolve_call_count(
        *,
        explicit_count: int | None,
        tool_invocations: list[str],
        tool_markers: tuple[str, ...],
    ) -> int:
        if explicit_count is not None:
            if explicit_count < 0:
                raise ValueError("usage counters must be non-negative")
            return explicit_count

        return sum(
            1
            for invocation in tool_invocations
            if any(marker in invocation.lower() for marker in tool_markers)
        )
