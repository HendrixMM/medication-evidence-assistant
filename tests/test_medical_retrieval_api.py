from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import src.api.app as app_module
from src.medlineplus_client import MedlinePlusResult


def test_retrieve_medical_evidence_returns_medication_centric_route_stub(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    app_module._medical_retrieval_service = None
    client = TestClient(app_module.app)

    response = client.post(
        "/api/tool/retrieve_medical_evidence",
        json={
            "question": "Can NSAIDs worsen asthma?",
            "conversation_context": [],
            "hints": {
                "preferred_domains": ["pubmed", "pmc"],
                "max_passes": 3,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {
        "question_analysis",
        "strategies_executed",
        "articles",
        "meta",
    }
    assert set(payload["meta"].keys()) == {
        "agent_passes",
        "perplexity_searches",
        "pubmed_fetches",
        "candidate_articles",
        "ranked_articles",
        "evidence_strength_hint",
        "cached",
        "errors",
    }
    assert "search_passes" not in payload["meta"]
    assert payload["question_analysis"]["question_type"] == "consumer_health_information"
    assert payload["strategies_executed"][0]["strategy_label"] == "medlineplus_web_service"
    assert payload["meta"]["errors"] == []
    assert set(payload["question_analysis"]["entities"].keys()) == {
        "drugs",
        "supplements",
        "normalized_ingredients",
        "conditions",
        "populations",
        "outcomes",
    }
    for entities_value in payload["question_analysis"]["entities"].values():
        assert isinstance(entities_value, list)
    assert isinstance(payload["question_analysis"]["optional_drug_normalizations"], list)
    assert "normalized_drugs" not in payload
    assert "queries_executed" not in payload


def test_retrieve_medical_evidence_exposes_response_schema(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    client = TestClient(app_module.app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    route_schema = schema["paths"]["/api/tool/retrieve_medical_evidence"]["post"]
    success_schema = route_schema["responses"]["200"]["content"]["application/json"]["schema"]
    assert success_schema == {"$ref": "#/components/schemas/MedicalEvidenceResponse"}


def test_legacy_retrieve_evidence_route_keeps_legacy_shape_when_mocked(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        app_module.evidence_service,
        "run",
        AsyncMock(
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
                    "errors": [],
                },
            },
        ),
    )
    client = TestClient(app_module.app)

    response = client.post(
        "/api/tool/retrieve_evidence",
        json={
            "question": "Can NSAIDs worsen asthma?",
            "intent": "safety",
            "drug_names": ["ibuprofen"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {
        "normalized_drugs",
        "queries_executed",
        "articles",
        "meta",
    }
    assert "question_analysis" not in payload
    assert "strategies_executed" not in payload


def test_retrieve_medical_evidence_route_uses_medlineplus_client(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    class StubMedlinePlusClient:
        def search(self, query, *, max_results=5):
            assert query == "Does aspirin cause bleeding?"
            assert max_results >= 1
            return [
                MedlinePlusResult(
                    identifier="medlineplus:aspirin",
                    title="Aspirin",
                    summary="Aspirin can increase bleeding risk.",
                    url="https://medlineplus.gov/aspirin.html",
                    sections={"Warnings": ["Ask a clinician about bleeding risk."]},
                )
            ]

    app_module.app.dependency_overrides[app_module.get_medlineplus_client] = lambda: StubMedlinePlusClient()
    try:
        client = TestClient(app_module.app)
        response = client.post(
            "/api/tool/retrieve_medical_evidence",
            json={"question": "Does aspirin cause bleeding?", "conversation_context": [], "hints": {}},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["question_analysis"]["question_type"] == "consumer_health_information"
        assert payload["articles"][0]["pmid"] == "medlineplus:aspirin"
        assert payload["articles"][0]["source_domain"] == "medlineplus.gov"
        assert payload["strategies_executed"][0]["result_count"] == 1
        assert payload["meta"]["agent_passes"] == 1
        assert payload["meta"]["perplexity_searches"] == 0
        assert payload["meta"]["evidence_strength_hint"] == "medium"
        assert payload["meta"]["errors"] == []
    finally:
        app_module.app.dependency_overrides.clear()
