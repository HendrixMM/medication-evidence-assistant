from __future__ import annotations

from typing import Any

from src.medical_retrieval.claim_models import Claim
from src.medical_retrieval.claim_composer import compose_patient_medication_guide
from src.medical_retrieval.claim_verification_service import (
    ClaimRetrievalContext,
    verify_claims_with_retrieval,
)
from src.medical_retrieval.perplexity_agent_client import (
    DiscoveryCandidate,
    PerplexityDiscoveryResult,
)
from src.medical_retrieval.pubmed_fetcher import PubmedFetcher
from src.medical_retrieval.schemas import MedicalArticleRecord


class FakeClaimAgent:
    def __init__(self, result: PerplexityDiscoveryResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def discover(self, question, question_analysis, *, max_steps=None, tool_handlers=None):
        self.calls.append({"question": question, "max_steps": max_steps})
        return self.result


class FakePubmedClient:
    def __init__(self, records: dict[str, dict[str, Any]], search_results: list[str] | None = None) -> None:
        self.records = records
        self.search_results = list(search_results or [])
        self._last_fetch: list[str] = []
        self.searches: list[dict[str, Any]] = []

    def esearch(self, query: str, retmax: int = 30) -> list[str]:
        self.searches.append({"query": query, "retmax": retmax})
        return self.search_results[:retmax]

    def efetch_xml(self, pmids: list[str]) -> str:
        self._last_fetch = list(pmids)
        return ""

    def parse_efetch(self, xml_text: str) -> list[dict[str, Any]]:
        return [self.records[pmid] for pmid in self._last_fetch if pmid in self.records]


def _article(pmid: str, abstract: str) -> MedicalArticleRecord:
    return MedicalArticleRecord(
        pmid=pmid,
        title=f"Article {pmid}",
        abstract=abstract,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )


def _record(pmid: str, abstract: str) -> dict[str, Any]:
    return {
        "pmid": pmid,
        "title": f"Article {pmid}",
        "abstract": abstract,
        "authors": [],
        "journal": "JAMA",
        "publication_date": "2025-01-01",
        "publication_types": ["Journal Article"],
        "doi": None,
    }


def _discovery_for(pmid: str) -> PerplexityDiscoveryResult:
    return PerplexityDiscoveryResult(
        strategies_executed=[],
        candidates=[
            DiscoveryCandidate(
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                snippet="claim-specific result",
                query="aspirin gastrointestinal bleeding",
                strategy_family="claim_expansion",
            )
        ],
        recall_rationale="claim expansion",
        sufficient_recall=True,
        agent_passes=1,
        tool_counts={"web_search": 1},
    )


def test_claim_retrieval_is_skipped_when_base_evidence_supports_claim() -> None:
    claim = Claim(
        id="c1",
        text="Low doses of ASA increased the risk of gastrointestinal bleeding.",
        type="safety",
    )
    agent = FakeClaimAgent(_discovery_for("2"))
    fetcher = PubmedFetcher(pubmed_client=FakePubmedClient({}))

    result = verify_claims_with_retrieval(
        [claim],
        [_article("1", "Low doses of ASA increased the risk of gastrointestinal bleeding.")],
        ClaimRetrievalContext(
            question="Does aspirin increase bleeding risk?",
            question_analysis=None,
            agent_client=agent,
            fetcher=fetcher,
            max_claim_retrieval_passes=1,
        ),
    )

    assert agent.calls == []
    assert result.validated_claims[0].label == "supported"
    assert result.extra_articles == []


def test_unclear_claim_triggers_bounded_claim_retrieval_and_becomes_supported() -> None:
    claim = Claim(
        id="c1",
        text="Low doses of ASA increased the risk of gastrointestinal bleeding.",
        type="safety",
    )
    agent = FakeClaimAgent(_discovery_for("2"))
    fetcher = PubmedFetcher(
        pubmed_client=FakePubmedClient(
            {"2": _record("2", "Low doses of ASA increased the risk of gastrointestinal bleeding.")}
        )
    )

    result = verify_claims_with_retrieval(
        [claim],
        [_article("1", "Aspirin was evaluated for cardiovascular prevention.")],
        ClaimRetrievalContext(
            question="Does aspirin increase bleeding risk?",
            question_analysis=None,
            agent_client=agent,
            fetcher=fetcher,
            max_claim_retrieval_passes=1,
            max_new_articles_per_claim=2,
        ),
    )

    assert len(agent.calls) == 1
    assert "Low doses of ASA increased the risk" in agent.calls[0]["question"]
    assert result.pubmed_fetches == 1
    assert [article.pmid for article in result.extra_articles] == ["2"]
    assert result.validated_claims[0].label == "supported"
    assert result.validated_claims[0].spans[0].pmid == "2"


def test_claim_retrieval_reuses_new_evidence_for_later_claims_without_extra_passes() -> None:
    claims = [
        Claim(
            id="c1",
            text="Antipyretic effects of ibuprofen and acetaminophen are similar at physician-directed doses; ibuprofen may be modestly superior at over-the-counter doses.",
            type="efficacy",
        ),
        Claim(
            id="c2",
            text="Tolerability profiles at physician dosing were similar.",
            type="safety",
        ),
    ]
    abstract = (
        "Antipyretic effects of ibuprofen and acetaminophen are similar at physician-directed doses; "
        "ibuprofen may be modestly superior at over-the-counter doses. "
        "Tolerability profiles at physician dosing were similar."
    )
    agent = FakeClaimAgent(_discovery_for("33966545"))
    fetcher = PubmedFetcher(pubmed_client=FakePubmedClient({"33966545": _record("33966545", abstract)}))

    result = verify_claims_with_retrieval(
        claims,
        [_article("1", "Aspirin was evaluated for cardiovascular prevention.")],
        ClaimRetrievalContext(
            question="Tylenol vs Advil for fever",
            question_analysis=None,
            agent_client=agent,
            fetcher=fetcher,
            max_claim_retrieval_passes=1,
            max_new_articles_per_claim=2,
        ),
    )

    assert len(agent.calls) == 1
    assert [claim.label for claim in result.validated_claims] == ["supported", "supported"]
    assert [article.pmid for article in result.extra_articles] == ["33966545"]


def test_claim_retrieval_uses_direct_pubmed_search_when_discovery_misses_bridge_article() -> None:
    claim = Claim(
        id="c1",
        text=(
            "Antipyretic effects of ibuprofen and acetaminophen are similar at physician-directed doses; "
            "ibuprofen may be modestly superior at over-the-counter doses."
        ),
        type="efficacy",
        evidence_terms=[
            "acetaminophen",
            "ibuprofen",
            "antipyretic",
            "fever",
            "similar",
            "superior",
        ],
    )
    abstract = claim.text
    agent = FakeClaimAgent(
        PerplexityDiscoveryResult(
            strategies_executed=[],
            candidates=[],
            recall_rationale="missed direct article",
            sufficient_recall=False,
            agent_passes=1,
            tool_counts={"web_search": 1},
        )
    )
    pubmed_client = FakePubmedClient(
        {"33966545": _record("33966545", abstract)},
        search_results=["33966545"],
    )
    fetcher = PubmedFetcher(pubmed_client=pubmed_client)

    result = verify_claims_with_retrieval(
        [claim],
        [_article("1", "Aspirin was evaluated for cardiovascular prevention.")],
        ClaimRetrievalContext(
            question="Tylenol vs Advil for fever",
            question_analysis=None,
            agent_client=agent,
            fetcher=fetcher,
            max_claim_retrieval_passes=1,
            max_new_articles_per_claim=2,
        ),
    )

    assert pubmed_client.searches
    assert result.validated_claims[0].label == "supported"
    assert [article.pmid for article in result.extra_articles] == ["33966545"]


def test_claim_retrieval_budget_is_honored_and_remaining_claims_stay_unclear() -> None:
    claims = [
        Claim(id="c1", text="Claim one increases bleeding risk.", type="safety"),
        Claim(id="c2", text="Claim two increases bleeding risk.", type="safety"),
    ]
    agent = FakeClaimAgent(_discovery_for("9"))
    fetcher = PubmedFetcher(pubmed_client=FakePubmedClient({}))

    result = verify_claims_with_retrieval(
        claims,
        [_article("1", "Aspirin was evaluated for cardiovascular prevention.")],
        ClaimRetrievalContext(
            question="Does aspirin increase bleeding risk?",
            question_analysis=None,
            agent_client=agent,
            fetcher=fetcher,
            max_claim_retrieval_passes=1,
        ),
    )

    assert len(agent.calls) == 1
    assert [claim.label for claim in result.validated_claims] == ["unclear", "unclear"]
    assert result.extra_articles == []


def test_prior_candidate_without_evidence_is_dropped_from_patient_guide() -> None:
    supported = "Ibuprofen reduced fever and pain in children."
    unsupported_prior = "Acetaminophen is always safer than ibuprofen."
    claims = [
        Claim(id="c1", text=supported, type="efficacy", importance=9),
        Claim(id="c2", text=unsupported_prior, type="safety", importance=8),
    ]
    agent = FakeClaimAgent(_discovery_for("9"))
    fetcher = PubmedFetcher(pubmed_client=FakePubmedClient({}))

    result = verify_claims_with_retrieval(
        claims,
        [_article("1", supported)],
        ClaimRetrievalContext(
            question="Tylenol vs Advil",
            question_analysis=None,
            agent_client=agent,
            fetcher=fetcher,
            max_claim_retrieval_passes=0,
        ),
    )
    answer = compose_patient_medication_guide(result.validated_claims, question="Tylenol vs Advil")

    assert [claim.label for claim in result.validated_claims] == ["supported", "unclear"]
    assert supported in answer
    assert unsupported_prior not in answer
