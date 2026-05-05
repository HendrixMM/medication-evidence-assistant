from __future__ import annotations

from typing import Any

import pytest

from src.medical_retrieval.cache import MedicalRetrievalCache
from src.medical_retrieval.claim_models import Claim
from src.medical_retrieval.errors import (
    PerplexityAgentError,
    QuestionAnalysisError,
    RerankerError,
)
from src.medical_retrieval.perplexity_agent_client import (
    DiscoveryCandidate,
    DiscoveryStrategy,
    PerplexityDiscoveryResult,
)
from src.medical_retrieval.pubmed_fetcher import PubmedFetcher
from src.medical_retrieval.question_analysis import (
    MedicationMention,
    MedicationQuestionAnalysis,
    MedicationQuestionEntities,
    NamedMedicationEntity,
    NormalizedIngredient,
    RxNormCandidate,
)
from src.medical_retrieval.ranker import RankedArticle, RankingResult
from src.medical_retrieval.schemas import MedicalEvidenceRequest, MedicalRetrievalHints
from src.medical_retrieval.service import MedicalRetrievalService


def _make_analysis() -> MedicationQuestionAnalysis:
    return MedicationQuestionAnalysis(
        question_type="medication_safety",
        entities=MedicationQuestionEntities(
            drugs=[MedicationMention(raw_name="aspirin")],
            supplements=[],
            normalized_ingredients=[NormalizedIngredient(raw_name="aspirin", canonical_name="acetylsalicylic acid")],
            drug_classes=[],
            conditions=[NamedMedicationEntity(name="bleeding")],
            populations=[],
            outcomes=[NamedMedicationEntity(name="GI bleeding")],
            mechanisms=[],
        ),
        rxnorm_candidates=[RxNormCandidate(raw_name="aspirin", canonical_name="acetylsalicylic acid")],
        retrieval_constraints=[],
    )


class FakeAnalyzer:
    def __init__(self, result: MedicationQuestionAnalysis | Exception) -> None:
        self.result = result
        self.calls: list[tuple[str, list[Any]]] = []

    def analyze(self, question: str, conversation_context: list[dict[str, str]] | None = None) -> MedicationQuestionAnalysis:
        self.calls.append((question, conversation_context or []))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeAgentClient:
    def __init__(
        self,
        result: PerplexityDiscoveryResult | Exception,
        *,
        invoke_handler_args: list[dict[str, Any]] | None = None,
    ) -> None:
        self.result = result
        self.invoke_handler_args = invoke_handler_args or []
        self.calls: list[dict[str, Any]] = []

    def discover(self, question, question_analysis, *, max_steps=None, tool_handlers=None):
        self.calls.append({
            "question": question,
            "question_analysis": question_analysis,
            "max_steps": max_steps,
            "tool_handlers_keys": sorted((tool_handlers or {}).keys()),
        })
        for args in self.invoke_handler_args:
            tool_handlers["pubmed_normalize_and_fetch"](args)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class SequenceAgentClient:
    def __init__(self, results: list[PerplexityDiscoveryResult]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def discover(self, question, question_analysis, *, max_steps=None, tool_handlers=None):
        self.calls.append({
            "question": question,
            "question_analysis": question_analysis,
            "max_steps": max_steps,
            "tool_handlers_keys": sorted((tool_handlers or {}).keys()),
        })
        return self.results.pop(0)


class FakePubmedClient:
    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self.records = records
        self._last_fetch: list[str] = []

    def efetch_xml(self, pmids: list[str]) -> str:
        self._last_fetch = list(pmids)
        return ""

    def parse_efetch(self, xml_text: str) -> list[dict[str, Any]]:
        return [self.records[p] for p in self._last_fetch if p in self.records]


def _record(pmid: str) -> dict[str, Any]:
    return {
        "pmid": pmid,
        "title": f"T{pmid}",
        "abstract": f"A{pmid}",
        "authors": "Doe, J",
        "journal": "JAMA",
        "publication_date": "2024-01-01",
        "publication_types": ["Journal Article"],
        "doi": f"10.0/{pmid}",
    }


class FakeReranker:
    def __init__(self, result: RankingResult | Exception) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def rank(self, question, question_analysis, candidates):
        self.calls.append({"question": question, "candidates": candidates})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _ranked(pmid: str, payload: dict[str, Any], rank: int = 1) -> RankedArticle:
    return RankedArticle(
        pmid=pmid,
        rank=rank,
        composite_score=0.9,
        include=True,
        reason="strong",
        arbitration_reason="top",
        payload=payload,
    )


def _discovery() -> PerplexityDiscoveryResult:
    return PerplexityDiscoveryResult(
        strategies_executed=[
            DiscoveryStrategy(strategy_label="literal_med", query="aspirin bleeding", domains=["pubmed.ncbi.nlm.nih.gov"], result_count=1),
        ],
        candidates=[
            DiscoveryCandidate(url="https://pubmed.ncbi.nlm.nih.gov/1/", snippet="s", query="aspirin bleeding", strategy_family="literal_med"),
        ],
        recall_rationale="ok",
        sufficient_recall=True,
        agent_passes=2,
        tool_counts={"web_search": 3, "pubmed_normalize_and_fetch": 1},
    )


def test_run_full_happy_path_assembles_response_and_meta() -> None:
    analyzer = FakeAnalyzer(_make_analysis())
    pubmed_client = FakePubmedClient({"1": _record("1")})
    fetcher = PubmedFetcher(pubmed_client=pubmed_client)
    candidates_arg = [{
        "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
        "snippet": "s",
        "query": "aspirin bleeding",
        "strategy_family": "literal_med",
    }]
    agent = FakeAgentClient(_discovery(), invoke_handler_args=[{"candidates": candidates_arg}])
    reranker = FakeReranker(RankingResult(
        ranked=[_ranked("1", {
            "pmid": "1", "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
            "title": "T1", "abstract": "A1", "authors": ["Doe, J"], "journal": "JAMA",
            "publication_date": "2024-01-01", "doi": "10.0/1",
            "publication_types": ["Journal Article"], "provenance": [],
        })],
        thin_evidence=False,
        rationale="ok",
    ))

    service = MedicalRetrievalService(
        analyzer=analyzer,
        agent_client=agent,
        fetcher=fetcher,
        reranker=reranker,
    )
    request = MedicalEvidenceRequest(question="Does aspirin cause bleeding?")
    response = service.run(request)

    assert response.question_analysis.question_type == "medication_safety"
    assert response.question_analysis.entities.drugs == ["aspirin"]
    assert response.question_analysis.optional_drug_normalizations == ["acetylsalicylic acid"]
    assert len(response.strategies_executed) == 1
    assert response.strategies_executed[0].strategy_label == "literal_med"
    assert len(response.articles) == 1
    assert response.articles[0].pmid == "1"
    assert response.articles[0].source_domain == "pubmed.ncbi.nlm.nih.gov"
    meta = response.meta
    assert meta.agent_passes == 2
    assert meta.perplexity_searches == 3
    assert meta.pubmed_fetches == 1
    assert meta.candidate_articles == 1
    assert meta.ranked_articles == 1
    assert meta.evidence_strength_hint == "low"  # 1 article -> low
    assert meta.cached is False
    assert meta.errors == []


def test_run_returns_cached_response_with_cached_flag_set() -> None:
    analyzer = FakeAnalyzer(_make_analysis())
    fetcher = PubmedFetcher(pubmed_client=FakePubmedClient({}))
    agent = FakeAgentClient(_discovery())
    reranker = FakeReranker(RankingResult(ranked=[], thin_evidence=True, rationale="x"))
    cache = MedicalRetrievalCache()

    service = MedicalRetrievalService(
        analyzer=analyzer, agent_client=agent, fetcher=fetcher, reranker=reranker, cache=cache,
    )
    request = MedicalEvidenceRequest(question="aspirin?")

    first = service.run(request)
    assert first.meta.cached is False
    assert len(analyzer.calls) == 1

    second = service.run(request)
    assert second.meta.cached is True
    assert len(analyzer.calls) == 1  # cache short-circuited the second call


def test_run_question_analysis_error_returns_empty_response_with_error() -> None:
    analyzer = FakeAnalyzer(QuestionAnalysisError("LLM down"))
    service = MedicalRetrievalService(
        analyzer=analyzer,
        agent_client=FakeAgentClient(_discovery()),
        fetcher=PubmedFetcher(pubmed_client=FakePubmedClient({})),
        reranker=FakeReranker(RankingResult(ranked=[], thin_evidence=True, rationale="x")),
    )
    response = service.run(MedicalEvidenceRequest(question="?"))

    assert response.articles == []
    assert response.meta.errors[0].layer == "question_analysis"
    assert "LLM down" in response.meta.errors[0].reason


def test_run_agent_discovery_error_returns_partial_response_with_error() -> None:
    analyzer = FakeAnalyzer(_make_analysis())
    agent = FakeAgentClient(PerplexityAgentError("agent boom"))
    service = MedicalRetrievalService(
        analyzer=analyzer,
        agent_client=agent,
        fetcher=PubmedFetcher(pubmed_client=FakePubmedClient({})),
        reranker=FakeReranker(RankingResult(ranked=[], thin_evidence=True, rationale="x")),
    )
    response = service.run(MedicalEvidenceRequest(question="aspirin?"))

    assert response.articles == []
    assert response.meta.errors[0].layer == "agent_discovery"
    assert response.meta.candidate_articles == 0


def test_run_ranker_error_returns_response_without_articles_but_with_candidates_counted() -> None:
    analyzer = FakeAnalyzer(_make_analysis())
    pubmed_client = FakePubmedClient({"1": _record("1")})
    fetcher = PubmedFetcher(pubmed_client=pubmed_client)
    candidates_arg = [{
        "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
        "snippet": "s", "query": "q", "strategy_family": "f",
    }]
    agent = FakeAgentClient(_discovery(), invoke_handler_args=[{"candidates": candidates_arg}])
    reranker = FakeReranker(RerankerError("ranker boom"))

    service = MedicalRetrievalService(
        analyzer=analyzer, agent_client=agent, fetcher=fetcher, reranker=reranker,
    )
    response = service.run(MedicalEvidenceRequest(question="aspirin?"))

    assert response.articles == []
    assert response.meta.candidate_articles == 1
    assert response.meta.ranked_articles == 0
    assert response.meta.errors[0].layer == "ranking"


def test_run_skips_caching_when_errors_present() -> None:
    analyzer = FakeAnalyzer(QuestionAnalysisError("nope"))
    cache = MedicalRetrievalCache()
    service = MedicalRetrievalService(
        analyzer=analyzer,
        agent_client=FakeAgentClient(_discovery()),
        fetcher=PubmedFetcher(pubmed_client=FakePubmedClient({})),
        reranker=FakeReranker(RankingResult(ranked=[], thin_evidence=True, rationale="x")),
        cache=cache,
    )
    request = MedicalEvidenceRequest(question="?")

    service.run(request)
    second = service.run(request)
    assert second.meta.cached is False
    assert len(analyzer.calls) == 2


def test_run_passes_max_steps_hint_to_agent_client() -> None:
    analyzer = FakeAnalyzer(_make_analysis())
    agent = FakeAgentClient(_discovery())
    service = MedicalRetrievalService(
        analyzer=analyzer,
        agent_client=agent,
        fetcher=PubmedFetcher(pubmed_client=FakePubmedClient({})),
        reranker=FakeReranker(RankingResult(ranked=[], thin_evidence=True, rationale="x")),
    )
    request = MedicalEvidenceRequest(question="aspirin?", hints=MedicalRetrievalHints(max_passes=4))

    service.run(request)
    assert agent.calls[0]["max_steps"] == 4
    assert agent.calls[0]["tool_handlers_keys"] == ["pubmed_normalize_and_fetch", "rxnorm_lookup"]


def test_run_evidence_strength_hint_high_with_many_ranked_articles() -> None:
    analyzer = FakeAnalyzer(_make_analysis())
    pubmed_client = FakePubmedClient({str(i): _record(str(i)) for i in range(1, 6)})
    fetcher = PubmedFetcher(pubmed_client=pubmed_client)
    candidates_arg = [
        {"url": f"https://pubmed.ncbi.nlm.nih.gov/{i}/", "snippet": "s", "query": "q", "strategy_family": "f"}
        for i in range(1, 6)
    ]
    agent = FakeAgentClient(_discovery(), invoke_handler_args=[{"candidates": candidates_arg}])
    ranked = [
        _ranked(str(i), {
            "pmid": str(i), "url": f"https://pubmed.ncbi.nlm.nih.gov/{i}/",
            "title": f"T{i}", "abstract": "a", "authors": [],
            "journal": "j", "publication_date": "2024", "doi": "d",
            "publication_types": [], "provenance": [],
        }, rank=i)
        for i in range(1, 6)
    ]
    reranker = FakeReranker(RankingResult(ranked=ranked, thin_evidence=False, rationale="ok"))

    service = MedicalRetrievalService(
        analyzer=analyzer, agent_client=agent, fetcher=fetcher, reranker=reranker,
    )
    response = service.run(MedicalEvidenceRequest(question="?"))

    assert response.meta.ranked_articles == 5
    assert response.meta.evidence_strength_hint == "high"


def test_run_does_not_call_reranker_when_pool_empty() -> None:
    analyzer = FakeAnalyzer(_make_analysis())
    agent = FakeAgentClient(_discovery())
    reranker = FakeReranker(RankingResult(ranked=[], thin_evidence=False, rationale="x"))

    service = MedicalRetrievalService(
        analyzer=analyzer,
        agent_client=agent,
        fetcher=PubmedFetcher(pubmed_client=FakePubmedClient({})),
        reranker=reranker,
    )
    service.run(MedicalEvidenceRequest(question="?"))

    assert reranker.calls == []


def test_run_fetches_structured_discovery_candidates_when_agent_skips_fetch_tool() -> None:
    analyzer = FakeAnalyzer(_make_analysis())
    pubmed_client = FakePubmedClient({"1": _record("1")})
    fetcher = PubmedFetcher(pubmed_client=pubmed_client)
    agent = FakeAgentClient(_discovery())
    ranked_payload = {
        "pmid": "1",
        "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
        "title": "T1",
        "abstract": "A1",
        "authors": ["Doe, J"],
        "journal": "JAMA",
        "publication_date": "2024-01-01",
        "doi": "10.0/1",
        "publication_types": ["Journal Article"],
        "provenance": [],
    }
    reranker = FakeReranker(RankingResult(
        ranked=[_ranked("1", ranked_payload)],
        thin_evidence=False,
        rationale="ok",
    ))

    service = MedicalRetrievalService(
        analyzer=analyzer,
        agent_client=agent,
        fetcher=fetcher,
        reranker=reranker,
    )
    response = service.run(MedicalEvidenceRequest(question="Does aspirin cause bleeding?"))

    assert response.meta.perplexity_searches == 3
    assert response.meta.pubmed_fetches == 1
    assert response.meta.candidate_articles == 1
    assert response.meta.ranked_articles == 1
    assert response.articles[0].pmid == "1"
    assert response.meta.errors == []


def test_run_fetches_all_structured_discovery_candidates_even_after_tool_fetch() -> None:
    analyzer = FakeAnalyzer(_make_analysis())
    pubmed_client = FakePubmedClient({"1": _record("1"), "2": _record("2")})
    fetcher = PubmedFetcher(pubmed_client=pubmed_client)
    initial_tool_candidates = [{
        "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
        "snippet": "first",
        "query": "aspirin bleeding",
        "strategy_family": "literal_med",
    }]
    discovery = PerplexityDiscoveryResult(
        strategies_executed=_discovery().strategies_executed,
        candidates=[
            DiscoveryCandidate(
                url="https://pubmed.ncbi.nlm.nih.gov/1/",
                snippet="first",
                query="aspirin bleeding",
                strategy_family="literal_med",
            ),
            DiscoveryCandidate(
                url="https://pubmed.ncbi.nlm.nih.gov/2/",
                snippet="second",
                query="aspirin GI bleeding",
                strategy_family="outcome",
            ),
        ],
        recall_rationale="two candidates",
        sufficient_recall=True,
        agent_passes=2,
        tool_counts={"web_search": 2, "pubmed_normalize_and_fetch": 1},
    )
    agent = FakeAgentClient(discovery, invoke_handler_args=[{"candidates": initial_tool_candidates}])
    ranked_payloads = [
        {
            "pmid": pmid,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "title": f"T{pmid}",
            "abstract": f"A{pmid}",
            "authors": ["Doe, J"],
            "journal": "JAMA",
            "publication_date": "2024-01-01",
            "doi": f"10.0/{pmid}",
            "publication_types": ["Journal Article"],
            "provenance": [],
        }
        for pmid in ("1", "2")
    ]
    reranker = FakeReranker(RankingResult(
        ranked=[_ranked("1", ranked_payloads[0], rank=1), _ranked("2", ranked_payloads[1], rank=2)],
        thin_evidence=False,
        rationale="ok",
    ))

    service = MedicalRetrievalService(
        analyzer=analyzer,
        agent_client=agent,
        fetcher=fetcher,
        reranker=reranker,
    )
    response = service.run(MedicalEvidenceRequest(question="Does aspirin cause bleeding?"))

    assert response.meta.pubmed_fetches == 2
    assert response.meta.candidate_articles == 2
    assert response.meta.ranked_articles == 2
    assert {article.pmid for article in response.articles} == {"1", "2"}
    assert response.meta.errors == []


def test_run_builds_composed_answer_from_validated_claims() -> None:
    analyzer = FakeAnalyzer(_make_analysis())
    pubmed_client = FakePubmedClient({"1": _record("1") | {
        "abstract": "Low doses of ASA increased the risk of gastrointestinal bleeding."
    }})
    fetcher = PubmedFetcher(pubmed_client=pubmed_client)
    candidates_arg = [{
        "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
        "snippet": "s",
        "query": "aspirin bleeding",
        "strategy_family": "literal_med",
    }]
    agent = FakeAgentClient(_discovery(), invoke_handler_args=[{"candidates": candidates_arg}])
    ranked_payload = {
        "pmid": "1",
        "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
        "title": "T1",
        "abstract": "Low doses of ASA increased the risk of gastrointestinal bleeding.",
        "authors": ["Doe, J"],
        "journal": "JAMA",
        "publication_date": "2024-01-01",
        "doi": "10.0/1",
        "publication_types": ["Journal Article"],
        "provenance": [],
    }
    reranker = FakeReranker(RankingResult(
        ranked=[_ranked("1", ranked_payload)],
        thin_evidence=False,
        rationale="ok",
    ))

    service = MedicalRetrievalService(
        analyzer=analyzer,
        agent_client=agent,
        fetcher=fetcher,
        reranker=reranker,
    )
    response = service.run(MedicalEvidenceRequest(question="Does aspirin increase bleeding risk?"))

    assert response.claims
    assert response.validated_claims
    assert any(claim.label == "supported" for claim in response.validated_claims)
    assert response.composed_answer
    assert "Low doses of ASA increased the risk of gastrointestinal bleeding." in response.composed_answer
    assert "[PMID 1](https://pubmed.ncbi.nlm.nih.gov/1/)" in response.composed_answer


def test_run_expands_unclear_claim_with_claim_level_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedClaimGenerator:
        def generate(self, **kwargs):
            return [
                Claim(
                    id="c1",
                    text="Low doses of ASA increased the risk of gastrointestinal bleeding.",
                    type="safety",
                )
            ]

    import src.medical_retrieval.service as service_module

    monkeypatch.setattr(service_module, "ClaimGenerator", FixedClaimGenerator)
    analyzer = FakeAnalyzer(_make_analysis())
    pubmed_client = FakePubmedClient({
        "1": _record("1") | {"abstract": "Aspirin was evaluated for cardiovascular prevention."},
        "2": _record("2") | {"abstract": "Low doses of ASA increased the risk of gastrointestinal bleeding."},
    })
    fetcher = PubmedFetcher(pubmed_client=pubmed_client)
    base_discovery = _discovery()
    claim_discovery = PerplexityDiscoveryResult(
        strategies_executed=[
            DiscoveryStrategy(
                strategy_label="claim_expansion",
                query="aspirin gastrointestinal bleeding",
                domains=["pubmed.ncbi.nlm.nih.gov"],
                result_count=1,
            )
        ],
        candidates=[
            DiscoveryCandidate(
                url="https://pubmed.ncbi.nlm.nih.gov/2/",
                snippet="claim support",
                query="aspirin gastrointestinal bleeding",
                strategy_family="claim_expansion",
            )
        ],
        recall_rationale="claim support",
        sufficient_recall=True,
        agent_passes=1,
        tool_counts={"web_search": 1},
    )
    agent = SequenceAgentClient([base_discovery, claim_discovery])
    ranked_payload = {
        "pmid": "1",
        "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
        "title": "T1",
        "abstract": "Aspirin was evaluated for cardiovascular prevention.",
        "authors": ["Doe, J"],
        "journal": "JAMA",
        "publication_date": "2024-01-01",
        "doi": "10.0/1",
        "publication_types": ["Journal Article"],
        "provenance": [],
    }
    reranker = FakeReranker(RankingResult(
        ranked=[_ranked("1", ranked_payload)],
        thin_evidence=False,
        rationale="ok",
    ))

    service = MedicalRetrievalService(
        analyzer=analyzer,
        agent_client=agent,
        fetcher=fetcher,
        reranker=reranker,
    )
    response = service.run(MedicalEvidenceRequest(question="Does aspirin increase bleeding risk?"))

    assert len(agent.calls) == 2
    assert response.meta.pubmed_fetches == 2
    assert {article.pmid for article in response.articles} == {"1", "2"}
    assert response.validated_claims[0].label == "supported"
    assert response.validated_claims[0].spans[0].pmid == "2"
    assert response.composed_answer
    assert "[PMID 2](https://pubmed.ncbi.nlm.nih.gov/2/)" in response.composed_answer
