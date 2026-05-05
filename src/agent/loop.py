"""Core orchestration loop for agent-mode retrieval."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator

from .config import AgentConfig
from .evaluator import RetrievalEvaluator
from .guardrails import AgentGuardrails
from .models import (
    AgentStopReason,
    ConsumerAnswer,
    EvidenceLevel,
    EvidenceSummary,
    PlannedQuery,
    QueryIntent,
    RetrievalResult,
    SessionState,
    Source,
    SubQuestion,
    TurnRecord,
)
from .planner import QueryPlanner
from .session import SessionManager
from .streaming import StreamingAdapter
from .tools.pubmed_search import PubMedSearchTool
from .tools.summarize_evidence import SummarizeEvidenceTool
from .tools.translate_for_consumer import TranslateForConsumerTool

logger = logging.getLogger(__name__)

AGENT_MODE_ENABLED: bool = os.getenv("AGENT_MODE_ENABLED", "true").lower() in ("1", "true", "yes", "on")

# Public SSE event vocabulary for answer_streaming().
_PUBLIC_STREAM_EVENTS: frozenset[str] = frozenset({"started", "status", "answer_delta", "sources", "error", "finished"})


def _article_identity_key(article: object) -> str:
    pmid = getattr(article, "pmid", "")
    if pmid:
        return str(pmid)
    title = getattr(article, "title", "") or ""
    publication_date = getattr(article, "publication_date", "") or ""
    return f"_no_pmid:{title}|{publication_date}"


class AgentLoop:
    """Coordinate planning, retrieval, evaluation, synthesis, and translation."""

    def __init__(
        self,
        planner: QueryPlanner,
        evaluator: RetrievalEvaluator,
        pubmed_tool: PubMedSearchTool,
        summarize_tool: SummarizeEvidenceTool,
        translate_tool: TranslateForConsumerTool,
        guardrails: AgentGuardrails,
        session_manager: SessionManager,
        config: AgentConfig,
    ) -> None:
        self.planner = planner
        self.evaluator = evaluator
        self.pubmed_tool = pubmed_tool
        self.summarize_tool = summarize_tool
        self.translate_tool = translate_tool
        self.guardrails = guardrails
        self.session_manager = session_manager
        self.config = config
        self._ncbi_semaphore_capacity = 10 if config.pubmed_api_key else 3
        self._ncbi_min_interval = 0.1 if config.pubmed_api_key else (1 / 3)

    def answer(self, question: str, session_id: str | None = None) -> ConsumerAnswer:
        return asyncio.run(self._collect_consumer_answer(question, session_id=session_id))

    async def _collect_consumer_answer(self, question: str, session_id: str | None = None) -> ConsumerAnswer:
        """Internal non-SSE path: iterates _iter_pipeline_events and assembles ConsumerAnswer directly."""
        final_answer: ConsumerAnswer | None = None
        answer_text = ""
        sources: list[Source] = []
        evidence_level = EvidenceLevel.insufficient
        disclaimer = ""
        verdict = None
        uncertainties: list[str] = []
        safety_warnings: list[str] = []
        stop_reason: AgentStopReason | None = None
        saw_error = False
        last_error_code: str | None = None
        last_error_message: str | None = None

        async for event_type, data in self._iter_pipeline_events(question, session_id=session_id):
            if event_type == "consumer_answer":
                final_answer = ConsumerAnswer.model_validate(data)
            elif event_type == "answer_delta":
                answer_text += str(data.get("text") or "")
            elif event_type == "sources":
                sources = [Source.model_validate(s) for s in data.get("sources", [])]
            elif event_type == "finished":
                raw_stop_reason = data.get("stop_reason")
                if raw_stop_reason:
                    try:
                        stop_reason = AgentStopReason(raw_stop_reason)
                    except ValueError:
                        logger.warning("Unknown agent stop reason from stream: %s", raw_stop_reason)
            elif event_type == "error":
                saw_error = True
                raw_code = data.get("code")
                raw_message = data.get("message")
                if raw_code:
                    last_error_code = str(raw_code)
                if raw_message:
                    last_error_message = str(raw_message)

        if final_answer is not None:
            return final_answer

        if answer_text:
            return ConsumerAnswer(
                answer=answer_text,
                sources=sources,
                evidence_level=evidence_level,
                uncertainties=uncertainties,
                safety_warnings=safety_warnings,
                disclaimer=disclaimer,
                verdict=verdict,
            )

        if stop_reason in {AgentStopReason.error, AgentStopReason.blocked} or saw_error:
            return self._terminal_consumer_answer(
                stop_reason=stop_reason or AgentStopReason.error,
                code=last_error_code,
                message=last_error_message,
                sources=sources,
                evidence_level=evidence_level,
            )

        return ConsumerAnswer(
            answer=answer_text,
            sources=sources,
            evidence_level=evidence_level,
            uncertainties=uncertainties,
            safety_warnings=safety_warnings,
            disclaimer=disclaimer,
            verdict=verdict,
        )

    async def answer_streaming(self, question: str, session_id: str | None = None) -> AsyncIterator[str]:
        """Public SSE stream. Emits only _PUBLIC_STREAM_EVENTS; all formatting via StreamingAdapter."""
        async for event_type, data in self._iter_pipeline_events(question, session_id=session_id):
            if event_type not in _PUBLIC_STREAM_EVENTS:
                continue
            if event_type == "sources":
                yield StreamingAdapter.sources_event(data.get("sources", []))
            else:
                yield StreamingAdapter.to_sse(StreamingAdapter.make_event(event_type, data))

    async def _iter_pipeline_events(
        self, question: str, session_id: str | None = None
    ) -> AsyncIterator[tuple[str, dict]]:
        """Internal pipeline: yields (event_type, data) tuples including the internal consumer_answer event."""
        ncbi_semaphore = asyncio.Semaphore(self._ncbi_semaphore_capacity)
        ncbi_rate_lock = asyncio.Lock()
        next_ncbi_request_at: list[float] = [0.0]

        session = self.session_manager.load_session(session_id) if session_id else None
        if session is None:
            session = self.session_manager.create_session(question)

        yield ("started", {"session_id": session.session_id, "question": question})

        ok, message = self.guardrails.validate_input(question)
        if not ok:
            yield ("status", {"step": "blocked", "detail": "safety_check"})
            error_code = "safety_check"
            error_message = message or "Safety check failed."
            yield ("error", {"code": error_code, "message": error_message})
            self.session_manager.finalize_session(
                session,
                self._terminal_consumer_answer(
                    stop_reason=AgentStopReason.blocked,
                    code=error_code,
                    message=error_message,
                ),
                AgentStopReason.blocked,
            )
            yield (
                "finished",
                {
                    "session_id": session.session_id,
                    "usage": session.usage.model_dump(),
                    "stop_reason": AgentStopReason.blocked.value,
                },
            )
            return

        yield ("status", {"step": "planning"})
        try:
            plan = await asyncio.to_thread(self.planner.plan, question)
            initial_planning_tokens = self._read_usage_tokens(getattr(self.planner, "llm_client", None))
            initial_planning_calls = 1
        except Exception:
            yield ("status", {"step": "error", "detail": "planner_failure"})
            yield ("error", {"code": "planner_failure", "message": "Planning failed. Continuing with a fallback query plan."})
            fallback_queries = self.planner._fallback_keyword_query(question, [])
            from .models import QueryPlan

            plan = QueryPlan(
                normalized_question=question,
                intent=QueryIntent.unknown,
                sub_questions=[
                    SubQuestion(
                        text=question,
                        rationale="Fallback plan created after planner failure.",
                        priority=1,
                    )
                ],
                planned_queries=fallback_queries,
                drugs_identified=[],
                reasoning="Fallback plan generated due to planner failure.",
            )
            initial_planning_tokens = 0
            initial_planning_calls = 0
        session.plan = plan

        all_articles: dict[str, object] = {}
        articles_list = []
        new_queries: list[PlannedQuery] = []
        stop_reason = AgentStopReason.max_turns

        for turn in range(self.config.max_agent_turns):
            if not self.session_manager.is_within_budget(session):
                stop_reason = AgentStopReason.budget_exhausted
                break

            queries_to_run = plan.planned_queries if turn == 0 else new_queries
            remaining_pubmed_calls = self.config.max_pubmed_calls - session.usage.pubmed_calls
            if remaining_pubmed_calls <= 0:
                stop_reason = AgentStopReason.budget_exhausted
                break
            if len(queries_to_run) > remaining_pubmed_calls:
                queries_to_run = queries_to_run[:remaining_pubmed_calls]
            if not queries_to_run:
                stop_reason = AgentStopReason.budget_exhausted
                break

            yield ("status", {"step": "searching", "turn": turn + 1, "queries": len(queries_to_run)})

            turn_start = time.perf_counter()
            try:
                results = await self._run_parallel_pubmed(queries_to_run, ncbi_semaphore, ncbi_rate_lock, next_ncbi_request_at)
            except Exception:
                yield ("status", {"step": "error", "detail": "pubmed_failure"})
                error_code = "pubmed_failure"
                error_message = "PubMed retrieval failed."
                yield ("error", {"code": error_code, "message": error_message})
                stop_reason = AgentStopReason.error
                fallback_answer = self._terminal_consumer_answer(
                    stop_reason=stop_reason,
                    code=error_code,
                    message=error_message,
                )
                if turn == 0:
                    session.usage.output_tokens += initial_planning_tokens
                    session.usage.llm_calls += initial_planning_calls
                self.session_manager.finalize_session(session, fallback_answer, stop_reason)
                yield (
                    "finished",
                    {
                        "session_id": session.session_id,
                        "usage": session.usage.model_dump(),
                        "stop_reason": stop_reason.value,
                    },
                )
                return
            pubmed_ms = (time.perf_counter() - turn_start) * 1000

            failed_result = next((result for result in results if result.error), None)
            if failed_result is not None:
                yield ("status", {"step": "error", "detail": "pubmed_failure"})
                error_code = "pubmed_failure"
                error_message = "PubMed retrieval failed."
                yield ("error", {"code": error_code, "message": error_message})
                stop_reason = AgentStopReason.error
                fallback_answer = self._terminal_consumer_answer(
                    stop_reason=stop_reason,
                    code=error_code,
                    message=error_message,
                )
                if turn == 0:
                    session.usage.output_tokens += initial_planning_tokens
                    session.usage.llm_calls += initial_planning_calls
                self.session_manager.finalize_session(session, fallback_answer, stop_reason)
                yield (
                    "finished",
                    {
                        "session_id": session.session_id,
                        "usage": session.usage.model_dump(),
                        "stop_reason": stop_reason.value,
                    },
                )
                return

            for result in results:
                for article in result.articles:
                    key = _article_identity_key(article)
                    if key not in all_articles:
                        all_articles[key] = article

            articles_list = list(all_articles.values())
            session.all_articles = list(articles_list)
            yield ("status", {"step": "evaluating", "articles_found": len(articles_list)})

            eval_start = time.perf_counter()
            verdict = self.evaluator.evaluate(plan, articles_list, turn)
            eval_ms = (time.perf_counter() - eval_start) * 1000
            logger.info(
                "Agent turn %d failure_modes=%s retrieval_score=%.3f",
                turn + 1,
                verdict.failure_modes,
                verdict.retrieval_score,
            )
            ordered_articles: list[object] = []
            seen_article_keys: set[str] = set()
            for article in verdict.relevant_articles:
                key = _article_identity_key(article)
                if key not in seen_article_keys:
                    ordered_articles.append(article)
                    seen_article_keys.add(key)
            for article in articles_list:
                key = _article_identity_key(article)
                if key not in seen_article_keys:
                    ordered_articles.append(article)
                    seen_article_keys.add(key)
            articles_list = ordered_articles
            session.all_articles = list(articles_list)

            turn_llm_tokens = initial_planning_tokens if turn == 0 else 0
            turn_llm_calls = initial_planning_calls if turn == 0 else 0
            turn_llm_tokens += self._read_usage_value(getattr(self.evaluator, "last_assessment_output_tokens", 0))
            turn_llm_calls += self._read_usage_value(getattr(self.evaluator, "last_assessment_llm_calls", 0))

            if verdict.is_sufficient:
                stop_reason = AgentStopReason.completed
            elif turn < self.config.max_agent_turns - 1:
                new_queries = await asyncio.to_thread(self.planner.refine_plan, plan, verdict)
                turn_llm_tokens += self._read_usage_tokens(getattr(self.planner, "llm_client", None))
                turn_llm_calls += 1
                if not new_queries:
                    stop_reason = AgentStopReason.max_turns
            else:
                stop_reason = AgentStopReason.max_turns

            turn_record = TurnRecord(
                turn_number=turn + 1,
                tool_invocations=[f"pubmed_search:{query.pubmed_query[:40]}" for query in queries_to_run],
                evaluation=verdict,
                tokens_used=turn_llm_tokens,
                decision="sufficient" if verdict.is_sufficient else verdict.recommendation,
                num_articles=len(articles_list),
                num_relevant=len(verdict.relevant_articles),
                study_type_counts=verdict.study_type_distribution,
                latency_breakdown_ms={"pubmed_ms": pubmed_ms, "evaluator_ms": eval_ms},
            )
            self.session_manager.record_turn(
                session,
                turn_record,
                pubmed_calls=len(queries_to_run),
                llm_calls=turn_llm_calls,
            )

            initial_planning_tokens = 0
            initial_planning_calls = 0

            if verdict.is_sufficient:
                break
            if not self.session_manager.is_within_budget(session):
                stop_reason = AgentStopReason.budget_exhausted
                break
            if not new_queries:
                break
        else:
            stop_reason = AgentStopReason.max_turns

        yield ("status", {"step": "synthesizing"})
        try:
            evidence = await asyncio.to_thread(
                self.summarize_tool.execute,
                articles_list,
                question,
                plan.intent,
                plan.drugs_identified,
            )
        except Exception:
            yield ("status", {"step": "error", "detail": "synthesis_failure"})
            yield ("error", {"code": "synthesis_failure", "message": "Evidence synthesis failed."})
            evidence = EvidenceSummary(
                ranked_articles=articles_list,
                evidence_level=EvidenceLevel.insufficient,
                key_findings=[],
                convergent_findings=[],
                divergent_findings=[],
                drug_interactions=[],
                pk_parameters={},
                research_gaps=[],
            )
        session.evidence_summary = evidence

        yield ("status", {"step": "translating"})
        try:
            consumer_answer = await asyncio.to_thread(
                self.translate_tool.execute,
                question,
                plan.intent,
                evidence,
            )
        except Exception:
            yield ("status", {"step": "error", "detail": "translator_failure"})
            error_code = "translator_failure"
            error_message = "Answer generation failed. Returning retrieved sources only."
            yield ("error", {"code": error_code, "message": error_message})
            stop_reason = AgentStopReason.error
            # Generation failed — do not attribute tokens or calls
            try:
                fallback_sources = self.translate_tool.build_sources(evidence.ranked_articles)
            except Exception:
                logger.exception("Failed to build fallback sources after translator failure")
                fallback_sources = []
            if not isinstance(fallback_sources, list):
                fallback_sources = []
            consumer_answer = self._terminal_consumer_answer(
                stop_reason=stop_reason,
                code=error_code,
                message=error_message,
                sources=fallback_sources,
                evidence_level=evidence.evidence_level,
            )
            consumer_answer = self._safe_validate_output(consumer_answer)
            self.session_manager.finalize_session(session, consumer_answer, stop_reason)
            yield ("answer_delta", {"text": consumer_answer.answer})
            yield ("sources", {"sources": [source.model_dump() for source in consumer_answer.sources]})
            yield ("consumer_answer", consumer_answer.model_dump())
            yield (
                "finished",
                {
                    "session_id": session.session_id,
                    "usage": session.usage.model_dump(),
                    "stop_reason": stop_reason.value,
                },
            )
            return

        session.usage.output_tokens += self._read_usage_tokens(getattr(self.translate_tool, "llm_client", None))
        session.usage.llm_calls += 1
        consumer_answer = self._safe_validate_output(consumer_answer)
        self.session_manager.finalize_session(session, consumer_answer, stop_reason)

        yield ("answer_delta", {"text": consumer_answer.answer})
        yield ("sources", {"sources": [source.model_dump() for source in consumer_answer.sources]})
        yield ("consumer_answer", consumer_answer.model_dump())
        yield (
            "finished",
            {
                "session_id": session.session_id,
                "usage": session.usage.model_dump(),
                "stop_reason": stop_reason.value,
            },
        )

    async def _run_parallel_pubmed(
        self,
        queries: list[PlannedQuery],
        ncbi_semaphore: asyncio.Semaphore,
        ncbi_rate_lock: asyncio.Lock,
        next_ncbi_request_at: list[float],
    ) -> list[RetrievalResult]:
        async def _run_single(query: PlannedQuery) -> RetrievalResult:
            async with ncbi_semaphore:
                await self._wait_for_pubmed_rate_limit(ncbi_rate_lock, next_ncbi_request_at)
                return await asyncio.to_thread(self.pubmed_tool.execute, query)

        return await asyncio.gather(*[_run_single(query) for query in queries])

    async def _wait_for_pubmed_rate_limit(
        self,
        ncbi_rate_lock: asyncio.Lock,
        next_ncbi_request_at: list[float],
    ) -> None:
        async with ncbi_rate_lock:
            now = time.monotonic()
            if now < next_ncbi_request_at[0]:
                await asyncio.sleep(next_ncbi_request_at[0] - now)
            next_ncbi_request_at[0] = time.monotonic() + self._ncbi_min_interval

    @staticmethod
    def _fallback_consumer_answer(
        *,
        sources: list[Source] | None = None,
        evidence_level: EvidenceLevel = EvidenceLevel.insufficient,
    ) -> ConsumerAnswer:
        return ConsumerAnswer(
            answer="I found relevant studies but had trouble generating a summary. Here are the sources I found:",
            sources=sources or [],
            evidence_level=evidence_level,
            uncertainties=[],
            safety_warnings=[],
            disclaimer="Medical Disclaimer: This information is for research and educational purposes only.",
            verdict=None,
        )

    @classmethod
    def _terminal_consumer_answer(
        cls,
        *,
        stop_reason: AgentStopReason,
        code: str | None = None,
        message: str | None = None,
        sources: list[Source] | None = None,
        evidence_level: EvidenceLevel = EvidenceLevel.insufficient,
    ) -> ConsumerAnswer:
        fallback = cls._fallback_consumer_answer(sources=sources, evidence_level=evidence_level)
        normalized_message = str(message or code or "").strip()
        if stop_reason == AgentStopReason.blocked:
            block_message = normalized_message or "Safety check failed."
            return ConsumerAnswer(
                answer=block_message,
                sources=sources or [],
                evidence_level=evidence_level,
                uncertainties=[],
                safety_warnings=[block_message],
                disclaimer=fallback.disclaimer,
                verdict=AgentStopReason.blocked.value,
            )

        return fallback.model_copy(
            update={
                "safety_warnings": [normalized_message] if normalized_message else [],
                "verdict": stop_reason.value,
            }
        )

    @staticmethod
    def _read_usage_tokens(llm_client: object | None) -> int:
        raw_tokens = getattr(llm_client, "last_usage_tokens", 0)
        return raw_tokens if isinstance(raw_tokens, int) and raw_tokens >= 0 else 0

    @staticmethod
    def _read_usage_value(raw_value: object) -> int:
        return raw_value if isinstance(raw_value, int) and raw_value >= 0 else 0

    def _safe_validate_output(self, answer: ConsumerAnswer) -> ConsumerAnswer:
        try:
            return self.guardrails.validate_output(answer)
        except Exception:
            logger.exception("Agent output validation failed")
            return answer
