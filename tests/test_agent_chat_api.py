from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import src.api.app as app_module
from src.evidence.schemas import AgentChatMeta, AgentChatResponse, AgentChatUsage, ErrorRecord
from src.medlineplus_client import MedlinePlusResult
from src.medical_retrieval.schemas import (
    MedicalArticleRecord,
    MedicalEvidenceResponse,
    MedicalQuestionAnalysis,
    MedicalQuestionEntities,
    MedicalRetrievalError,
    MedicalRetrievalMeta,
)


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("AGENT_MODEL", "gpt-5.4-mini")

    monkeypatch.setattr(
        app_module,
        "_run_sdk_agent",
        AsyncMock(
            return_value=AgentChatResponse(
                message={"role": "assistant", "content": "Answer with disclaimer"},
                meta=AgentChatMeta(
                    stop_reason="completed",
                    usage=AgentChatUsage(),
                    tool_calls=1,
                    total_latency_ms=1.0,
                    errors=[],
                ),
            )
        ),
    )
    monkeypatch.setattr(
        app_module,
        "evidence_service",
        type("StubService", (), {"run": AsyncMock(return_value={"ok": True}), "run_request": AsyncMock()})(),
        raising=False,
    )

    return TestClient(app_module.app)


def test_agent_chat_returns_message_and_meta(api_client: TestClient) -> None:
    response = api_client.post(
        "/api/agent/chat",
        json={"messages": [{"role": "user", "content": "Is Tylenol safe to take?"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"message", "meta"}
    assert payload["message"] == {"role": "assistant", "content": "Answer with disclaimer"}
    assert payload["meta"]["stop_reason"] == "completed"
    assert set(payload["meta"].keys()) == {"stop_reason", "usage", "tool_calls", "total_latency_ms", "errors"}
    assert set(payload["meta"]["usage"].keys()) == {"pubmed_calls", "llm_calls", "output_tokens", "total_latency_ms"}


def test_retrieve_evidence_route_returns_service_payload(api_client: TestClient) -> None:
    app_module.evidence_service.run = AsyncMock(
        return_value={
            "normalized_drugs": [],
            "queries_executed": [],
            "articles": [],
            "meta": {
                "pubmed_calls": 0,
                "articles_considered": 0,
                "articles_heuristic_filtered": 0,
                "articles_llm_scored": 0,
                "normalization_used": True,
                "normalization_complete": False,
                "evidence_strength_hint": "low",
                "cached": False,
                "cost_estimate_usd": 0.0,
                "query_budget_used": 0,
                "query_budget_max": 12,
                "errors": [ErrorRecord(layer="llm_scorer", reason="fallback heuristic ordering used").model_dump()],
            },
        }
    )

    response = api_client.post(
        "/api/tool/retrieve_evidence",
        json={"question": "Is Tylenol safe to take?", "intent": "safety", "drug_names": ["Tylenol"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"normalized_drugs", "queries_executed", "articles", "meta"}
    assert payload["meta"]["cached"] is False
    assert set(payload["meta"].keys()) == {
        "pubmed_calls",
        "articles_considered",
        "articles_heuristic_filtered",
        "articles_llm_scored",
        "normalization_used",
        "normalization_complete",
        "evidence_strength_hint",
        "cached",
        "cost_estimate_usd",
        "query_budget_used",
        "query_budget_max",
        "errors",
    }


def test_run_sdk_agent_registers_retrieve_medlineplus_content_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    service_calls: list[dict] = []

    class StubClient:
        def search(self, question: str, *, max_results: int):
            service_calls.append({"question": question, "max_results": max_results})
            return [
                MedlinePlusResult(
                    identifier="medlineplus:aspirin",
                    title="Aspirin",
                    summary="Aspirin can increase bleeding risk.",
                    url="https://medlineplus.gov/druginfo/meds/a682878.html",
                )
            ]

    monkeypatch.setattr(app_module, "get_medlineplus_client", lambda: StubClient())

    captured: dict = {}

    class FakeAgent:
        def __init__(self, *, name, model, instructions, tools):
            captured["instructions"] = instructions
            captured["tools"] = tools
            captured["model"] = model

    class FakeResult:
        final_output = "answer"

    class FakeRunner:
        @staticmethod
        async def run(agent, messages, max_turns):
            tool = captured["tools"][0]
            captured["tool_name"] = getattr(tool, "tool_name", None) or getattr(tool, "__name__", None)
            return FakeResult()

    def _fake_function_tool(f):
        f.tool_name = f.__name__
        return f

    monkeypatch.setattr(
        app_module,
        "_load_agents_sdk",
        lambda: (FakeAgent, FakeRunner, _fake_function_tool),
    )

    response = asyncio.run(
        app_module._run_sdk_agent(
            [{"role": "user", "content": "Does aspirin cause bleeding?"}],
            max_turns=2,
        )
    )

    assert "retrieve_medlineplus_content" in captured["instructions"]
    assert response.message.role == "assistant"
    assert captured["tool_name"] == "retrieve_medlineplus_content"
    # Invoke the tool directly to confirm it delegates to MedlinePlus.
    tool = captured["tools"][0]
    payload = asyncio.run(tool(question="Is aspirin safe?", max_results=3))
    assert payload["question_analysis"]["question_type"] == "consumer_health_information"
    assert payload["articles"][0]["source_domain"] == "medlineplus.gov"
    assert service_calls == [{"question": "Is aspirin safe?", "max_results": 3}]


def test_run_sdk_agent_records_error_when_medlineplus_search_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    class FailingClient:
        def search(self, question: str, *, max_results: int):
            raise RuntimeError("network")

    monkeypatch.setattr(app_module, "get_medlineplus_client", lambda: FailingClient())

    captured: dict = {}

    class FakeAgent:
        def __init__(self, *, name, model, instructions, tools):
            captured["tools"] = tools

    class FakeResult:
        final_output = ""

    class FakeRunner:
        @staticmethod
        async def run(agent, messages, max_turns):
            return FakeResult()

    def _fake_function_tool(f):
        f.tool_name = f.__name__
        return f

    monkeypatch.setattr(
        app_module,
        "_load_agents_sdk",
        lambda: (FakeAgent, FakeRunner, _fake_function_tool),
    )

    response = asyncio.run(app_module._run_sdk_agent([{"role": "user", "content": "?"}], max_turns=1))

    tool = captured["tools"][0]
    payload = asyncio.run(tool(question="?"))
    assert payload["meta"]["errors"][0]["layer"] == "medlineplus"
    assert response.meta.stop_reason == "retrieval_failed"


def _medical_response(
    *,
    articles: list[MedicalArticleRecord] | None = None,
    pubmed_fetches: int = 0,
    ranked_articles: int = 0,
    errors: list[MedicalRetrievalError] | None = None,
) -> MedicalEvidenceResponse:
    return MedicalEvidenceResponse(
        question_analysis=MedicalQuestionAnalysis(
            question_type="medication_safety",
            entities=MedicalQuestionEntities(drugs=["Tylenol"]),
            optional_drug_normalizations=[],
        ),
        strategies_executed=[],
        articles=articles or [],
        meta=MedicalRetrievalMeta(
            agent_passes=1,
            perplexity_searches=1,
            pubmed_fetches=pubmed_fetches,
            candidate_articles=len(articles or []),
            ranked_articles=ranked_articles,
            evidence_strength_hint="medium" if articles else "unknown",
            cached=False,
            errors=errors or [],
        ),
    )


def _run_agent_with_tool_result(
    monkeypatch: pytest.MonkeyPatch,
    tool_result: MedicalEvidenceResponse,
    final_output: str,
) -> AgentChatResponse:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        app_module,
        "_medlineplus_medical_response",
        lambda question, *, client, max_results: tool_result,
    )
    monkeypatch.setattr(app_module, "get_medlineplus_client", lambda: object())
    captured: dict = {}

    class FakeAgent:
        def __init__(self, *, name, model, instructions, tools):
            captured["tools"] = tools

    class FakeResult:
        def __init__(self, output: str) -> None:
            self.final_output = output

    class FakeRunner:
        @staticmethod
        async def run(agent, messages, max_turns):
            await captured["tools"][0](question=messages[-1]["content"], max_results=1)
            return FakeResult(final_output)

    def _fake_function_tool(f):
        f.tool_name = f.__name__
        return f

    monkeypatch.setattr(
        app_module,
        "_load_agents_sdk",
        lambda: (FakeAgent, FakeRunner, _fake_function_tool),
    )

    return asyncio.run(
        app_module._run_sdk_agent(
            [{"role": "user", "content": "Can I take Tylenol and Advil together?"}],
            max_turns=2,
        )
    )


@pytest.mark.parametrize(
    "tool_result",
    [
        _medical_response(
            errors=[MedicalRetrievalError(layer="question_analysis", reason="Question analysis LLM response was invalid")]
        ),
        _medical_response(pubmed_fetches=0, ranked_articles=0, articles=[]),
        _medical_response(
            pubmed_fetches=0,
            ranked_articles=1,
            articles=[
                MedicalArticleRecord(
                    pmid="123",
                    title="Acetaminophen and ibuprofen",
                    url="https://example.test/not-medlineplus",
                    source_domain="example.test",
                )
            ],
        ),
    ],
)
def test_run_sdk_agent_publishes_research_informed_guidance_without_canonical_pubmed_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tool_result: MedicalEvidenceResponse,
) -> None:
    response = _run_agent_with_tool_result(
        monkeypatch,
        tool_result,
        "Tylenol and Advil can be taken together safely if you follow dosing limits.",
    )

    content = response.message.content.lower()
    assert response.meta.stop_reason == "completed"
    assert response.meta.usage.pubmed_calls == tool_result.meta.pubmed_fetches
    assert "can be taken together" in content
    assert "dosing limits" in content
    assert "consult" in content or "medical advice" in content


def test_run_sdk_agent_attaches_supporting_references_with_medlineplus_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _run_agent_with_tool_result(
        monkeypatch,
        _medical_response(
            pubmed_fetches=2,
            ranked_articles=1,
            articles=[
                MedicalArticleRecord(
                    pmid="medlineplus:painrelievers",
                    title="Acetaminophen and ibuprofen",
                    abstract="Acetaminophen and ibuprofen produced similar analgesia.",
                    url="https://medlineplus.gov/painrelievers.html",
                    source_domain="medlineplus.gov",
                )
            ],
        ),
        "Acetaminophen and ibuprofen produced similar analgesia.",
    )

    assert response.meta.stop_reason == "completed"
    assert "Acetaminophen and ibuprofen produced similar analgesia." in response.message.content
    assert "https://medlineplus.gov/painrelievers.html" in response.message.content
    assert "not medical advice" in response.message.content


def test_run_sdk_agent_publishes_research_informed_guidance_without_claim_traceability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _run_agent_with_tool_result(
        monkeypatch,
        _medical_response(
            pubmed_fetches=2,
            ranked_articles=1,
            articles=[
                MedicalArticleRecord(
                    pmid="123",
                    title="Acetaminophen and ibuprofen",
                    abstract="This review discusses acetaminophen and ibuprofen use.",
                    url="https://medlineplus.gov/painrelievers.html",
                    source_domain="medlineplus.gov",
                )
            ],
        ),
        "Tylenol and Advil can be taken together safely if dosing limits are followed.",
    )

    content = response.message.content.lower()
    assert response.meta.stop_reason == "completed"
    assert not any(error.reason == "claim_traceability_failed" for error in response.meta.errors)
    assert "can be taken together" in content
    assert "dosing limits" in content


def test_run_sdk_agent_uses_service_composed_answer_when_model_draft_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supported_answer = (
        "Safety\n"
        "Low doses of ASA may increase the risk of gastrointestinal bleeding.\n\n"
        "This information is from MedlinePlus and general health information, and is not medical advice.\n"
        "Please consult a healthcare provider for personal medical decisions."
    )
    tool_result = _medical_response(
        pubmed_fetches=2,
        ranked_articles=1,
        articles=[
            MedicalArticleRecord(
                pmid="medlineplus:aspirin",
                title="Aspirin",
                abstract="Aspirin may increase the risk of bleeding.",
                url="https://medlineplus.gov/druginfo/meds/a682878.html",
                source_domain="medlineplus.gov",
            )
        ],
    )
    tool_result.composed_answer = supported_answer

    response = _run_agent_with_tool_result(
        monkeypatch,
        tool_result,
        "",
    )

    assert response.meta.stop_reason == "completed"
    assert "Low doses of ASA may increase the risk of gastrointestinal bleeding." in response.message.content
    assert "https://medlineplus.gov/druginfo/meds/a682878.html" in response.message.content


def test_run_sdk_agent_publishes_traceable_patient_guide_for_tylenol_vs_advil_fever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span_text = (
        "Antipyretic effects of ibuprofen and acetaminophen are similar at physician-directed doses; "
        "ibuprofen may be modestly superior at over-the-counter doses."
    )
    guide = (
        "Quick comparison\n"
        "- Tylenol and Advil appear similarly effective for fever at physician-directed doses; "
        "at over-the-counter doses, Advil may be modestly more effective. "
        "(MedlinePlus: https://medlineplus.gov/fever.html)\n\n"
        "Bottom line\n"
        "- Tylenol and Advil appear similarly effective for fever at physician-directed doses; "
        "at over-the-counter doses, Advil may be modestly more effective. "
        "(MedlinePlus: https://medlineplus.gov/fever.html)\n\n"
        "This information is from MedlinePlus and general health information, and is not medical advice.\n"
        "Please consult a healthcare provider for personal medical decisions."
    )
    tool_result = _medical_response(
        pubmed_fetches=2,
        ranked_articles=1,
        articles=[
            MedicalArticleRecord(
                pmid="medlineplus:fever",
                title="Fever",
                abstract=span_text,
                url="https://medlineplus.gov/fever.html",
                source_domain="medlineplus.gov",
            )
        ],
    )
    tool_result.composed_answer = guide

    response = _run_agent_with_tool_result(
        monkeypatch,
        tool_result,
        guide,
    )

    assert response.meta.stop_reason == "completed"
    assert response.meta.errors == []
    assert "Quick comparison" in response.message.content
    assert "Bottom line" in response.message.content
    assert response.message.content.count("- ") >= 2
    assert "always best" not in response.message.content


def test_run_sdk_agent_recovers_from_perplexity_loop_failure_with_direct_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span_text = "Ibuprofen reduced temperature more than acetaminophen at 2, 4, and 6 hours after treatment."
    guide = (
        "Tylenol and Advil are not the same medicine.\n"
        "The points below use MedlinePlus as supporting context.\n\n"
        "Quick comparison\n"
        "- Advil reduced temperature more than Tylenol at 2, 4, and 6 hours after treatment. "
        "(MedlinePlus: https://medlineplus.gov/fever.html)\n\n"
        "Bottom line\n"
        "- Advil reduced temperature more than Tylenol at 2, 4, and 6 hours after treatment. "
        "(MedlinePlus: https://medlineplus.gov/fever.html)\n\n"
        "This information is from MedlinePlus and general health information, and is not medical advice.\n"
        "Please consult a healthcare provider for personal medical decisions."
    )
    loop_failed = _medical_response(
        pubmed_fetches=0,
        ranked_articles=0,
        articles=[],
        errors=[
            MedicalRetrievalError(
                layer="agent_discovery",
                reason="Perplexity Agent function call loop exceeded max_steps",
            )
        ],
    )
    recovered = _medical_response(
        pubmed_fetches=2,
        ranked_articles=1,
        articles=[
            MedicalArticleRecord(
                pmid="medlineplus:fever",
                title="Fever",
                abstract=span_text,
                url="https://medlineplus.gov/fever.html",
                source_domain="medlineplus.gov",
            )
        ],
    )
    recovered.composed_answer = guide
    service_calls: list[dict[str, int | None | str]] = []
    results = [loop_failed, recovered]

    def _fake_medlineplus_response(question, *, client, max_results):
        service_calls.append({"question": question, "max_results": max_results})
        return results.pop(0)

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(app_module, "_medlineplus_medical_response", _fake_medlineplus_response)
    monkeypatch.setattr(app_module, "get_medlineplus_client", lambda: object())
    captured: dict = {}

    class FakeAgent:
        def __init__(self, *, name, model, instructions, tools):
            captured["tools"] = tools

    class FakeResult:
        final_output = ""

    class FakeRunner:
        @staticmethod
        async def run(agent, messages, max_turns):
            await captured["tools"][0](question=messages[-1]["content"], max_results=1)
            return FakeResult()

    def _fake_function_tool(f):
        f.tool_name = f.__name__
        return f

    monkeypatch.setattr(
        app_module,
        "_load_agents_sdk",
        lambda: (FakeAgent, FakeRunner, _fake_function_tool),
    )

    response = asyncio.run(
        app_module._run_sdk_agent(
            [{"role": "user", "content": "Tylenol vs Advil for fever"}],
            max_turns=2,
        )
    )

    assert response.meta.stop_reason == "completed"
    assert response.meta.usage.pubmed_calls == 2
    assert "retrieval_failed" not in {error.reason for error in response.meta.errors}
    assert any(error.reason == "agent_loop_exhaustion_recovered_by_internal_retrieval" for error in response.meta.errors)
    assert "Quick comparison" in response.message.content
    assert "Bottom line" in response.message.content
    assert response.message.content.count("- ") >= 2
    assert "always best" not in response.message.content
    assert service_calls == [
        {"question": "Tylenol vs Advil for fever", "max_results": 1},
        {"question": "Tylenol vs Advil for fever", "max_results": 3},
    ]


def test_run_sdk_agent_keeps_draft_when_loop_failure_direct_retrieval_is_not_traceable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_failed = _medical_response(
        pubmed_fetches=0,
        ranked_articles=0,
        articles=[],
        errors=[
            MedicalRetrievalError(
                layer="agent_discovery",
                reason="Perplexity Agent function call loop exceeded max_steps",
            )
        ],
    )
    direct_failed = _medical_response(
        pubmed_fetches=2,
        ranked_articles=1,
        articles=[
            MedicalArticleRecord(
                pmid="medlineplus:painrelievers",
                title="Acetaminophen and ibuprofen",
                abstract="This review discusses acetaminophen and ibuprofen use.",
                url="https://medlineplus.gov/painrelievers.html",
                source_domain="medlineplus.gov",
            )
        ],
    )
    direct_failed.composed_answer = "Tylenol and Advil can always be taken together safely."

    results = [loop_failed, direct_failed]

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        app_module,
        "_medlineplus_medical_response",
        lambda question, *, client, max_results: results.pop(0),
    )
    monkeypatch.setattr(app_module, "get_medlineplus_client", lambda: object())
    captured: dict = {}

    class FakeAgent:
        def __init__(self, *, name, model, instructions, tools):
            captured["tools"] = tools

    class FakeResult:
        final_output = (
            "Tylenol and Advil can sometimes be used on the same day, but dosing depends on age, "
            "health conditions, and other medicines.\n\n"
            "This information is from MedlinePlus and general health information, and is not medical advice.\n"
            "Please consult a healthcare provider for personal medical decisions."
        )

    class FakeRunner:
        @staticmethod
        async def run(agent, messages, max_turns):
            await captured["tools"][0](question=messages[-1]["content"], max_results=1)
            return FakeResult()

    def _fake_function_tool(f):
        return f

    monkeypatch.setattr(
        app_module,
        "_load_agents_sdk",
        lambda: (FakeAgent, FakeRunner, _fake_function_tool),
    )

    response = asyncio.run(
        app_module._run_sdk_agent(
            [{"role": "user", "content": "Tylenol vs Advil for fever"}],
            max_turns=2,
        )
    )

    assert response.meta.stop_reason == "completed"
    assert "sometimes be used on the same day" in response.message.content
    assert "always be taken together" not in response.message.content
