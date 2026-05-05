"""FastAPI application for agent-mode chat and session APIs."""
from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
import logging
import math
import threading
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.agent import AGENT_MODE_ENABLED, __version__
from src.agent import AgentLoop
from src.agent.config import AgentConfig
from src.agent.models import SessionState
from src.api.rate_limit import RateLimitExceeded, get_http_limiter
from src.api.agent_prompt import SYSTEM_PROMPT
from src.evidence.schemas import (
    AgentChatMeta,
    AgentChatRequest,
    AgentChatResponse,
    AgentChatUsage,
    EvidenceRequest,
)
from src.evidence.service import evidence_service
from src.medlineplus_client import MedlinePlusClient, expand_consumer_query
from src.medical_retrieval.schemas import (
    MedicalArticleRecord,
    MedicalEvidenceRequest,
    MedicalEvidenceResponse,
    MedicalQuestionEntities,
    MedicalQuestionAnalysis,
    MedicalRetrievalMeta,
    MedicalStrategyExecution,
)

_agent_loop: AgentLoop | None = None
_agent_loop_lock = threading.Lock()
logger = logging.getLogger(__name__)

ComponentFactory = Callable[..., Any]
AgentLoopComposer = Callable[[], AgentLoop]


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


class SessionResponse(SessionState):
    pass


class HealthResponse(BaseModel):
    status: str
    agent_mode_enabled: bool
    version: str
    detail: str | None = None


class RateLimitStatusResponse(BaseModel):
    requests_last_minute: int
    max_requests_per_minute: int
    remaining_minute: int
    retry_after_seconds: float
    daily_count: int
    daily_limit: int | None
    remaining_daily: int | None


def _require_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this API deployment.")
    return api_key


def _load_agents_sdk() -> tuple[Any, Any, Any]:
    try:
        from agents import Agent, Runner, function_tool
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("openai-agents is required for /api/agent/chat.") from exc
    return Agent, Runner, function_tool


def _abstention_response(
    *,
    elapsed_ms: float,
    pubmed_calls: int,
    llm_calls: int,
    errors: list[dict[str, Any]],
    reason: str = "no_medlineplus_content",
    content: str | None = None,
) -> AgentChatResponse:
    response_errors = list(errors)
    if not response_errors:
        response_errors.append(
            {
                "layer": "orchestration",
                "reason": reason,
            }
        )
    response_content = content or (
        "MedlinePlus content could not be retrieved for this question, so I cannot provide a "
        "medication answer from this system.\n\n"
        "Try again in a moment or rephrase with the exact medication or supplement names.\n\n"
        "For personal decisions, contact a clinician or pharmacist. If symptoms are urgent or "
        "severe, seek timely medical care."
    )
    return AgentChatResponse(
        message={"role": "assistant", "content": response_content},
        meta=AgentChatMeta(
            stop_reason="retrieval_failed",
            usage=AgentChatUsage(
                pubmed_calls=pubmed_calls,
                llm_calls=llm_calls,
                output_tokens=len(response_content.split()),
                total_latency_ms=elapsed_ms,
            ),
            tool_calls=max(0, llm_calls - 1),
            total_latency_ms=elapsed_ms,
            errors=response_errors,
        ),
    )


def _is_medlineplus_article(article: object) -> bool:
    if not isinstance(article, dict):
        return False
    url = str(article.get("url") or "").lower()
    source_domain = str(article.get("source_domain") or "").lower()
    pmid = str(article.get("pmid") or "").lower()
    return "medlineplus.gov" in url or source_domain == "medlineplus.gov" or pmid.startswith("medlineplus:")


def _medlineplus_articles(retrieval_results: list[dict[str, Any]]) -> list[Any]:
    articles: list[Any] = []
    for result in retrieval_results:
        result_articles = result.get("articles") if isinstance(result, dict) else None
        if not isinstance(result_articles, list):
            continue
        articles.extend(article for article in result_articles if _is_medlineplus_article(article))
    return articles


async def _run_sdk_agent(messages: list[dict[str, str]], *, max_turns: int) -> AgentChatResponse:
    _require_openai_api_key()
    Agent, Runner, function_tool = _load_agents_sdk()
    budget_state: dict[str, Any] = {"used": 0, "calls": 0, "errors": [], "retrieval_results": []}
    start = time.perf_counter()
    medlineplus_client = get_medlineplus_client()

    @function_tool
    async def retrieve_medlineplus_content(
        question: str,
        max_results: int | None = None,
    ) -> dict:
        budget_state["calls"] += 1
        result = await asyncio.to_thread(
            _medlineplus_medical_response,
            question,
            client=medlineplus_client,
            max_results=max_results or _default_medlineplus_max_results(),
        )
        meta = result.meta
        budget_state["used"] += int(meta.pubmed_fetches or 0)
        budget_state["errors"].extend([err.model_dump() for err in meta.errors])
        payload = result.model_dump()
        budget_state["retrieval_results"].append(payload)
        return payload

    agent = Agent(
        name="MedAssistant",
        model=os.getenv("AGENT_MODEL", "gpt-5.4-mini"),
        instructions=SYSTEM_PROMPT,
        tools=[retrieve_medlineplus_content],
    )
    result = await Runner.run(agent, messages, max_turns=max_turns)
    final_output = str(getattr(result, "final_output", "") or "")
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    llm_calls = budget_state["calls"] + 1

    publishable_output = _research_informed_answer_from_retrieval_results(
        budget_state["retrieval_results"],
        fallback_output=final_output,
    )
    if publishable_output is not None:
        return _completed_agent_response(
            final_output=publishable_output,
            elapsed_ms=elapsed_ms,
            pubmed_calls=budget_state["used"],
            llm_calls=llm_calls,
            tool_calls=budget_state["calls"],
            errors=budget_state["errors"],
        )

    if _has_recoverable_agent_loop_exhaustion(budget_state["errors"]):
        direct_result = await asyncio.to_thread(
            _medlineplus_medical_response,
            _latest_user_question(messages),
            client=medlineplus_client,
            max_results=_direct_retrieval_fallback_max_passes(),
        )
        direct_payload = direct_result.model_dump()
        direct_errors = [err.model_dump() for err in direct_result.meta.errors]
        budget_state["used"] += int(direct_result.meta.pubmed_fetches or 0)
        budget_state["retrieval_results"].append(direct_payload)
        direct_output = _research_informed_answer_from_retrieval_results([direct_payload], fallback_output=None)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        if direct_output is not None:
            return _completed_agent_response(
                final_output=direct_output,
                elapsed_ms=elapsed_ms,
                pubmed_calls=budget_state["used"],
                llm_calls=llm_calls,
                tool_calls=budget_state["calls"],
                errors=[
                    {
                        "layer": "agent_discovery",
                        "reason": "agent_loop_exhaustion_recovered_by_internal_retrieval",
                    }
                ],
            )

    if budget_state["errors"] or not final_output.strip():
        return _abstention_response(
            elapsed_ms=elapsed_ms,
            pubmed_calls=budget_state["used"],
            llm_calls=llm_calls,
            errors=budget_state["errors"],
        )

    publishable_output = _research_informed_answer_from_retrieval_results(
        budget_state["retrieval_results"],
        fallback_output=final_output,
    )
    if publishable_output is None:
        publishable_output = _ensure_medical_disclaimer(final_output)
    return _completed_agent_response(
        final_output=publishable_output,
        elapsed_ms=elapsed_ms,
        pubmed_calls=budget_state["used"],
        llm_calls=llm_calls,
        tool_calls=budget_state["calls"],
        errors=budget_state["errors"],
    )


def _completed_agent_response(
    *,
    final_output: str,
    elapsed_ms: float,
    pubmed_calls: int,
    llm_calls: int,
    tool_calls: int,
    errors: list[dict[str, Any]],
) -> AgentChatResponse:
    return AgentChatResponse(
        message={"role": "assistant", "content": final_output},
        meta=AgentChatMeta(
            stop_reason="completed",
            usage=AgentChatUsage(
                pubmed_calls=pubmed_calls,
                llm_calls=llm_calls,
                output_tokens=len(final_output.split()),
                total_latency_ms=elapsed_ms,
            ),
            tool_calls=tool_calls,
            total_latency_ms=elapsed_ms,
            errors=errors,
        ),
    )


def _research_informed_answer_from_retrieval_results(
    retrieval_results: list[dict[str, Any]],
    *,
    fallback_output: str | None,
) -> str | None:
    final_output = (fallback_output or "").strip() or _verified_composed_answer(retrieval_results)
    if not final_output:
        return None
    final_output = _ensure_medical_disclaimer(final_output)
    return _append_supporting_references(final_output, _medlineplus_articles(retrieval_results))


def _ensure_medical_disclaimer(answer_text: str) -> str:
    disclaimer = (
        "This information is from MedlinePlus and general health information, and is not medical advice.\n"
        "Please consult a healthcare provider for personal medical decisions."
    )
    stripped = answer_text.strip()
    if "not medical advice" in stripped.lower() and "healthcare provider" in stripped.lower():
        return stripped
    return f"{stripped}\n\n{disclaimer}"


def _append_supporting_references(answer_text: str, articles: list[Any]) -> str:
    references: list[str] = []
    seen_urls: set[str] = set()
    lowered_answer = answer_text.lower()
    for article in articles:
        try:
            record = MedicalArticleRecord.model_validate(article)
        except Exception:
            continue
        url = str(record.url or "").strip()
        if not url or url in seen_urls or url.lower() in lowered_answer:
            continue
        seen_urls.add(url)
        title = str(record.title or "MedlinePlus source").strip()
        references.append(f"- [{title}]({url})")
        if len(references) >= 5:
            break
    if not references:
        return answer_text
    return f"{answer_text.strip()}\n\nSupporting MedlinePlus links:\n" + "\n".join(references)


def _has_recoverable_agent_loop_exhaustion(errors: list[dict[str, Any]]) -> bool:
    for error in errors:
        reason = str(error.get("reason") or "").lower()
        if "function call loop exceeded max_steps" in reason or "loop_exhaust" in reason:
            return True
    return False


def _latest_user_question(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and message.get("content"):
            return str(message["content"])
    return str(messages[-1].get("content", "")) if messages else ""


def _direct_retrieval_fallback_max_passes() -> int:
    try:
        return max(1, int(os.getenv("MEDLINEPLUS_DIRECT_FALLBACK_MAX_RESULTS", "3")))
    except ValueError:
        return 3


def _verified_composed_answer(retrieval_results: list[dict[str, Any]]) -> str | None:
    for result in reversed(retrieval_results):
        if not isinstance(result, dict):
            continue
        composed = result.get("composed_answer")
        if isinstance(composed, str) and composed.strip():
            return composed.strip()
    return None


def _parse_cors_allow_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "*")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or ["*"]


def _normalize_translator_model(model_name: str | None) -> str:
    if not model_name:
        return "70b"

    normalized = model_name.strip()
    supported = {
        "8b",
        "70b",
        "meta/llama-3.1-8b-instruct",
        "meta/llama-3.3-70b-instruct",
    }
    if normalized in supported:
        return normalized

    lowered = normalized.lower()
    if "70b" in lowered:
        return "70b"
    if "8b" in lowered:
        return "8b"
    return normalized


def _service_unavailable_exception() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="Agent service unavailable due to initialization or configuration failure.",
    )


def get_agent_config() -> AgentConfig:
    return AgentConfig.from_env()


def get_llm_client_factory() -> ComponentFactory:
    def _factory(config: AgentConfig) -> Any:
        if (config.llm_provider or "nvidia").lower() == "openai":
            from src.clients.openai_llm_client import OpenAILLMClient

            return OpenAILLMClient(api_key=config.openai_api_key)

        from src.nvidia_llm_client import NVIDIALLMClient

        return NVIDIALLMClient(api_key=config.nvidia_api_key)

    return _factory


def get_pubmed_client_factory() -> ComponentFactory:
    def _factory(config: AgentConfig) -> Any:
        from src.medlineplus_client import MedlinePlusClient

        return MedlinePlusClient(email=os.getenv("MEDLINEPLUS_EMAIL") or config.pubmed_email)

    return _factory


def get_medical_guardrails_factory() -> ComponentFactory:
    def _factory(config: AgentConfig) -> Any:
        from src.medical_guardrails import MedicalGuardrails, resolve_medical_guardrails_config_path

        return MedicalGuardrails(
            str(resolve_medical_guardrails_config_path()),
            enabled=config.enable_guardrails,
        )

    return _factory


def get_planner_factory() -> ComponentFactory:
    def _factory(llm_client: Any, config: AgentConfig) -> Any:
        from src.agent import QueryPlanner

        return QueryPlanner(llm_client=llm_client, config=config)

    return _factory


def get_evaluator_factory() -> ComponentFactory:
    def _factory(config: AgentConfig, retrieval_quality_scorer: Any) -> Any:
        from src.agent import RetrievalEvaluator

        return RetrievalEvaluator(config=config, retrieval_quality_scorer=retrieval_quality_scorer)

    return _factory


def get_pubmed_tool_factory() -> ComponentFactory:
    def _factory(pubmed_client: Any, config: AgentConfig) -> Any:
        from src.agent.tools.medlineplus_search import MedlinePlusSearchTool

        return MedlinePlusSearchTool(client=pubmed_client, max_results=config.pubmed_max_results_per_query)

    return _factory


def get_source_relevance_scorer_factory() -> ComponentFactory:
    def _factory(llm_client: Any, config: AgentConfig) -> Any | None:
        if not config.enable_source_relevance_scoring:
            return None

        from src.source_relevance_scorer import SourceRelevanceScorer

        return SourceRelevanceScorer(llm_client, threshold=config.source_relevance_threshold)

    return _factory


def get_retrieval_quality_scorer_factory() -> ComponentFactory:
    def _factory(config: AgentConfig, relevance_scorer: Any | None) -> Any:
        from src.agent.retrieval_quality import RetrievalQualityScorer
        from src.ranking_filter import StudyRankingFilter

        return RetrievalQualityScorer(
            config=config,
            ranking_filter=StudyRankingFilter(),
            source_relevance_scorer=relevance_scorer,
        )

    return _factory


def get_summarize_tool_factory() -> ComponentFactory:
    def _factory(config: AgentConfig, relevance_scorer: Any | None) -> Any:
        from src.agent.tools.summarize_evidence import SummarizeEvidenceTool
        from src.ddi_pk_processor import DDIPKProcessor
        from src.ranking_filter import StudyRankingFilter
        from src.synthesis_engine import SynthesisEngine

        return SummarizeEvidenceTool(
            ranking_filter=StudyRankingFilter(),
            synthesis_engine=SynthesisEngine(),
            ddi_pk_processor=DDIPKProcessor(),
            relevance_scorer=relevance_scorer,
            config=config,
        )

    return _factory


def get_translate_tool_factory() -> ComponentFactory:
    def _factory(llm_client: Any, config: AgentConfig) -> Any:
        from src.agent.tools.translate_for_consumer import TranslateForConsumerTool

        return TranslateForConsumerTool(
            llm_client=llm_client,
            config=config,
            model=_normalize_translator_model(config.translator_model),
        )

    return _factory


def get_agent_guardrails_factory() -> ComponentFactory:
    def _factory(medical_guardrails: Any, config: AgentConfig) -> Any:
        from src.agent import AgentGuardrails

        return AgentGuardrails(guardrails=medical_guardrails, config=config)

    return _factory


def get_session_manager_factory() -> ComponentFactory:
    def _factory(config: AgentConfig) -> Any:
        from src.agent.session import SessionManager

        return SessionManager(config)

    return _factory


def get_agent_loop_composer(
    config: AgentConfig = Depends(get_agent_config),
    llm_client_factory: ComponentFactory = Depends(get_llm_client_factory),
    pubmed_client_factory: ComponentFactory = Depends(get_pubmed_client_factory),
    medical_guardrails_factory: ComponentFactory = Depends(get_medical_guardrails_factory),
    planner_factory: ComponentFactory = Depends(get_planner_factory),
    evaluator_factory: ComponentFactory = Depends(get_evaluator_factory),
    pubmed_tool_factory: ComponentFactory = Depends(get_pubmed_tool_factory),
    source_relevance_scorer_factory: ComponentFactory = Depends(get_source_relevance_scorer_factory),
    retrieval_quality_scorer_factory: ComponentFactory = Depends(get_retrieval_quality_scorer_factory),
    summarize_tool_factory: ComponentFactory = Depends(get_summarize_tool_factory),
    translate_tool_factory: ComponentFactory = Depends(get_translate_tool_factory),
    agent_guardrails_factory: ComponentFactory = Depends(get_agent_guardrails_factory),
    session_manager_factory: ComponentFactory = Depends(get_session_manager_factory),
) -> AgentLoopComposer:
    def _compose() -> AgentLoop:
        llm_client = llm_client_factory(config)
        pubmed_client = pubmed_client_factory(config)
        medical_guardrails = medical_guardrails_factory(config)
        planner = planner_factory(llm_client, config)
        pubmed_tool = pubmed_tool_factory(pubmed_client, config)
        relevance_scorer = source_relevance_scorer_factory(llm_client, config)
        retrieval_quality_scorer = retrieval_quality_scorer_factory(config, relevance_scorer)
        evaluator = evaluator_factory(config, retrieval_quality_scorer)
        summarize_tool = summarize_tool_factory(config, relevance_scorer)
        translate_tool = translate_tool_factory(llm_client, config)
        guardrails = agent_guardrails_factory(medical_guardrails, config)
        session_manager = session_manager_factory(config)

        return AgentLoop(
            planner=planner,
            evaluator=evaluator,
            pubmed_tool=pubmed_tool,
            summarize_tool=summarize_tool,
            translate_tool=translate_tool,
            guardrails=guardrails,
            session_manager=session_manager,
            config=config,
        )

    return _compose


def _get_or_create_agent_loop(agent_loop_composer: AgentLoopComposer) -> AgentLoop:
    if not AGENT_MODE_ENABLED:
        raise HTTPException(status_code=503, detail="Agent mode is disabled")

    global _agent_loop
    if _agent_loop is None:
        with _agent_loop_lock:
            if _agent_loop is None:
                try:
                    _agent_loop = agent_loop_composer()
                except Exception:
                    logger.exception("Agent service initialization failed.")
                    raise _service_unavailable_exception() from None
    return _agent_loop


def get_agent_loop(agent_loop_composer: AgentLoopComposer = Depends(get_agent_loop_composer)) -> AgentLoop:
    return _get_or_create_agent_loop(agent_loop_composer)


_medical_retrieval_service: Any | None = None
_medical_retrieval_service_lock = threading.Lock()
_medlineplus_client: MedlinePlusClient | None = None
_medlineplus_client_lock = threading.Lock()


def get_medlineplus_client() -> MedlinePlusClient:
    """Lazily construct the MedlinePlus Web Service client."""
    global _medlineplus_client
    if _medlineplus_client is not None:
        return _medlineplus_client
    with _medlineplus_client_lock:
        if _medlineplus_client is None:
            _medlineplus_client = MedlinePlusClient(email=os.getenv("MEDLINEPLUS_EMAIL") or os.getenv("PUBMED_EMAIL"))
    return _medlineplus_client


def _default_medlineplus_max_results() -> int:
    try:
        return max(1, int(os.getenv("MEDLINEPLUS_MAX_RESULTS", "5")))
    except ValueError:
        return 5


def _medlineplus_medical_response(
    question: str,
    *,
    client: MedlinePlusClient,
    max_results: int,
) -> MedicalEvidenceResponse:
    try:
        results = client.search(expand_consumer_query(question), max_results=max_results)
        errors: list[Any] = []
    except Exception as exc:
        results = []
        errors = [{"layer": "medlineplus", "reason": str(exc)}]

    articles = [
        MedicalArticleRecord(
            pmid=result.identifier,
            title=result.title,
            abstract=_medlineplus_abstract(result.summary, result.sections),
            authors=["MedlinePlus"],
            journal="MedlinePlus",
            url=result.url,
            source_domain="medlineplus.gov",
            evidence_strength_hint="medium",
        )
        for result in results
    ]
    return MedicalEvidenceResponse(
        question_analysis=MedicalQuestionAnalysis(
            question_type="consumer_health_information",
            entities=MedicalQuestionEntities(),
            optional_drug_normalizations=[],
        ),
        strategies_executed=[
            MedicalStrategyExecution(
                strategy_label="medlineplus_web_service",
                query=expand_consumer_query(question),
                domains=["medlineplus.gov"],
                result_count=len(articles),
            )
        ],
        articles=articles,
        meta=MedicalRetrievalMeta(
            agent_passes=1,
            perplexity_searches=0,
            pubmed_fetches=1 if question.strip() else 0,
            candidate_articles=len(articles),
            ranked_articles=len(articles),
            evidence_strength_hint="medium" if articles else "unknown",
            cached=False,
            errors=errors,
        ),
    )


def _medlineplus_abstract(summary: str, sections: dict[str, list[str]]) -> str | None:
    parts = [summary] if summary else []
    for name, values in sections.items():
        for value in values:
            parts.append(f"{name}: {value}")
    text = "\n".join(part for part in parts if part)
    return text or None


def _build_medical_retrieval_service() -> Any | None:
    """Deprecated medication retrieval service hook.

    Consumer-facing chat now uses MedlinePlus Web Service directly. Returning
    None prevents the old service from being constructed by accident if this
    compatibility hook is still imported by older tests or integrations.
    """
    return None


def get_medical_retrieval_service() -> Any | None:
    """Lazily construct the medical-retrieval service from env; tests can override."""
    global _medical_retrieval_service
    if _medical_retrieval_service is not None:
        return _medical_retrieval_service
    with _medical_retrieval_service_lock:
        if _medical_retrieval_service is None:
            _medical_retrieval_service = _build_medical_retrieval_service()
    return _medical_retrieval_service


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_allow_origins(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def validate_openai_startup() -> None:
        _require_openai_api_key()

    @app.post("/api/chat")
    async def chat_stream(
        request_body: ChatRequest,
        request: Request,
        loop: AgentLoop = Depends(get_agent_loop),
    ) -> StreamingResponse:
        limiter = get_http_limiter()
        try:
            limiter.check_and_consume(limiter.get_client_ip(request))
        except RateLimitExceeded as exc:
            return JSONResponse(
                status_code=429,
                content={"detail": exc.detail, "retry_after_seconds": exc.retry_after_seconds},
                headers={"Retry-After": str(int(math.ceil(exc.retry_after_seconds)))},
            )

        async def generate():
            async for chunk in loop.answer_streaming(request_body.question, request_body.session_id):
                yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/tool/retrieve_evidence")
    async def retrieve_evidence_route(request_body: EvidenceRequest) -> dict[str, Any]:
        return await evidence_service.run(
            question=request_body.question,
            intent=request_body.intent,
            drug_names=request_body.drug_names,
        )

    @app.post(
        "/api/tool/retrieve_medical_evidence",
        response_model=MedicalEvidenceResponse,
        response_model_exclude={"claims", "validated_claims", "composed_answer"},
    )
    async def retrieve_medical_evidence_route(
        request_body: MedicalEvidenceRequest,
        medlineplus_client: MedlinePlusClient = Depends(get_medlineplus_client),
    ) -> MedicalEvidenceResponse:
        return await asyncio.to_thread(
            _medlineplus_medical_response,
            request_body.question,
            client=medlineplus_client,
            max_results=_default_medlineplus_max_results(),
        )

    @app.post("/api/agent/chat")
    async def agent_chat(request_body: AgentChatRequest, request: Request) -> dict[str, Any]:
        limiter = get_http_limiter()
        try:
            limiter.check_and_consume(limiter.get_client_ip(request))
        except RateLimitExceeded as exc:
            return JSONResponse(
                status_code=429,
                content={"detail": exc.detail, "retry_after_seconds": exc.retry_after_seconds},
                headers={"Retry-After": str(int(math.ceil(exc.retry_after_seconds)))},
            )
        try:
            response = await _run_sdk_agent(
                [message.model_dump() for message in request_body.messages],
                max_turns=int(os.getenv("AGENT_MAX_TURNS", "4")),
            )
        except Exception as exc:
            logger.exception("Agents SDK route failed")
            fallback = AgentChatResponse(
                message={
                    "role": "assistant",
                    "content": (
                        "I couldn't complete the evidence review right now.\n\n"
                        "This information is from MedlinePlus and general health information, and is not medical advice.\n"
                        "Please consult a healthcare provider for personal medical decisions."
                    ),
                },
                meta=AgentChatMeta(
                    stop_reason="error",
                    usage=AgentChatUsage(),
                    tool_calls=0,
                    total_latency_ms=0.0,
                    errors=[{"layer": "agent", "reason": str(exc)}],
                ),
            )
            return fallback.model_dump()
        return response.model_dump()

    @app.get("/api/rate-limit/status")
    async def get_rate_limit_status(request: Request) -> RateLimitStatusResponse:
        limiter = get_http_limiter()
        data = limiter.status(limiter.get_client_ip(request))
        return RateLimitStatusResponse(**data)

    @app.get("/api/session/{session_id}")
    async def get_session(
        session_id: str,
        loop: AgentLoop = Depends(get_agent_loop),
    ) -> SessionResponse:
        session = loop.session_manager.load_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionResponse.model_validate(session.model_dump())

    @app.get("/api/health")
    async def get_health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            agent_mode_enabled=AGENT_MODE_ENABLED,
            version=__version__,
        )

    return app


app = create_app()
