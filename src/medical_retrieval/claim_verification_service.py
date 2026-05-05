from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.medical_retrieval.claim_models import Claim, ValidatedClaim
from src.medical_retrieval.claim_traceability import ClaimTraceabilityGate
from src.medical_retrieval.errors import PerplexityAgentError, PubmedFetcherError
from src.medical_retrieval.perplexity_agent_client import PerplexityDiscoveryResult
from src.medical_retrieval.pubmed_fetcher import PUBMED_URL_TEMPLATE, PubmedFetcher
from src.medical_retrieval.schemas import MedicalArticleRecord
from src.medical_retrieval.source_normalizer import normalize_source_url


class _AgentClientProtocol(Protocol):
    def discover(
        self,
        question: str,
        question_analysis: Any,
        *,
        max_steps: int | None = ...,
        tool_handlers: dict[str, Any] | None = ...,
    ) -> PerplexityDiscoveryResult:
        ...


@dataclass
class ClaimRetrievalContext:
    question: str
    question_analysis: Any
    agent_client: _AgentClientProtocol
    fetcher: PubmedFetcher
    max_claim_retrieval_passes: int = 1
    max_new_articles_per_claim: int = 3
    max_agent_steps: int = 2


@dataclass
class ClaimVerificationResult:
    validated_claims: list[ValidatedClaim]
    extra_articles: list[MedicalArticleRecord] = field(default_factory=list)
    pubmed_fetches: int = 0
    perplexity_searches: int = 0
    agent_passes: int = 0
    errors: list[str] = field(default_factory=list)


def verify_claims_with_retrieval(
    claims: list[Claim],
    base_evidence: list[MedicalArticleRecord],
    context: ClaimRetrievalContext,
) -> ClaimVerificationResult:
    """Verify claims, optionally expanding unclear claims with bounded canonical PubMed retrieval."""

    gate = ClaimTraceabilityGate()
    validated_claims: list[ValidatedClaim] = []
    extra_articles: list[MedicalArticleRecord] = []
    errors: list[str] = []
    passes_used = 0
    starting_fetch_count = context.fetcher.fetch_count
    perplexity_searches = 0
    agent_passes = 0

    for claim in claims:
        available_evidence = base_evidence + extra_articles
        base_result = gate.validate_claims([claim], available_evidence)
        base_claim = base_result.validated_claims[0]
        if base_claim.label != "unclear":
            validated_claims.append(base_claim)
            continue

        if passes_used >= max(0, context.max_claim_retrieval_passes):
            validated_claims.append(base_claim)
            continue

        passes_used += 1
        try:
            discovery, fetched_articles = _retrieve_for_claim(
                claim,
                available_evidence,
                context,
                pass_index=passes_used,
            )
        except (PerplexityAgentError, PubmedFetcherError) as exc:
            errors.append(str(exc))
            validated_claims.append(base_claim)
            continue

        perplexity_searches += int((discovery.tool_counts or {}).get("web_search", 0) or 0)
        agent_passes += int(discovery.agent_passes or 0)
        new_articles = _dedupe_articles(
            fetched_articles,
            existing_pmids={article.pmid for article in available_evidence if article.pmid},
        )
        extra_articles.extend(new_articles)

        expanded_result = gate.validate_claims([claim], base_evidence + extra_articles)
        validated_claims.append(expanded_result.validated_claims[0])

    return ClaimVerificationResult(
        validated_claims=validated_claims,
        extra_articles=extra_articles,
        pubmed_fetches=context.fetcher.fetch_count - starting_fetch_count,
        perplexity_searches=perplexity_searches,
        agent_passes=agent_passes,
        errors=errors,
    )


def _retrieve_for_claim(
    claim: Claim,
    existing_evidence: list[MedicalArticleRecord],
    context: ClaimRetrievalContext,
    *,
    pass_index: int,
) -> tuple[PerplexityDiscoveryResult, list[MedicalArticleRecord]]:
    max_new_articles = max(0, context.max_new_articles_per_claim)
    fetched_articles = _direct_pubmed_search_for_claim(
        claim,
        context,
        existing_pmids={article.pmid for article in existing_evidence if article.pmid},
        pass_index=pass_index,
        remaining=max_new_articles,
    )
    if len(fetched_articles) >= max_new_articles:
        return _empty_discovery_result(), fetched_articles[:max_new_articles]

    existing_pmids = {article.pmid for article in existing_evidence if article.pmid}

    def normalize_handler(arguments: dict[str, Any]) -> dict[str, Any]:
        articles, result = _fetch_candidate_articles(
            arguments.get("candidates"),
            context,
            existing_pmids=existing_pmids | {article.pmid for article in fetched_articles if article.pmid},
            pass_index=pass_index,
        )
        fetched_articles.extend(articles)
        return result

    discovery = context.agent_client.discover(
        _focused_claim_query(context.question, claim),
        context.question_analysis,
        max_steps=context.max_agent_steps,
        tool_handlers={"pubmed_normalize_and_fetch": normalize_handler},
    )
    remaining_candidates = [
        candidate.model_dump()
        for candidate in discovery.candidates[: max_new_articles]
    ]
    articles, _result = _fetch_candidate_articles(
        remaining_candidates,
        context,
        existing_pmids=existing_pmids | {article.pmid for article in fetched_articles if article.pmid},
        pass_index=pass_index,
    )
    fetched_articles.extend(articles)
    if len(fetched_articles) < max_new_articles:
        direct_articles = _direct_pubmed_search_for_claim(
            claim,
            context,
            existing_pmids=existing_pmids | {article.pmid for article in fetched_articles if article.pmid},
            pass_index=pass_index,
            remaining=max_new_articles - len(fetched_articles),
        )
        fetched_articles.extend(direct_articles)
    return discovery, fetched_articles[:max_new_articles]


def _fetch_candidate_articles(
    candidates: Any,
    context: ClaimRetrievalContext,
    *,
    existing_pmids: set[str],
    pass_index: int,
) -> tuple[list[MedicalArticleRecord], dict[str, Any]]:
    if not isinstance(candidates, list) or not candidates:
        empty = {"normalized": [], "unmapped": [], "fetch_count": context.fetcher.fetch_count}
        return [], empty

    normalized_candidates = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        url = candidate.get("url")
        if not isinstance(url, str):
            continue
        pmid = normalize_source_url(url).pmid
        if pmid and pmid in existing_pmids:
            continue
        normalized_candidates.append(candidate)
        if len(normalized_candidates) >= max(0, context.max_new_articles_per_claim):
            break

    if not normalized_candidates:
        empty = {"normalized": [], "unmapped": [], "fetch_count": context.fetcher.fetch_count}
        return [], empty

    result = context.fetcher.normalize_and_fetch(
        {"candidates": normalized_candidates},
        pass_index=pass_index,
    )
    articles = [_payload_to_record(payload) for payload in result.get("normalized", [])]
    return articles, result


def _empty_discovery_result() -> PerplexityDiscoveryResult:
    return PerplexityDiscoveryResult(
        strategies_executed=[],
        candidates=[],
        recall_rationale="direct PubMed claim search used",
        sufficient_recall=True,
        agent_passes=0,
        tool_counts={},
    )


def _payload_to_record(payload: dict[str, Any]) -> MedicalArticleRecord:
    return MedicalArticleRecord(
        pmid=str(payload.get("pmid")) if payload.get("pmid") else None,
        title=payload.get("title"),
        abstract=payload.get("abstract"),
        authors=list(payload.get("authors") or []),
        journal=payload.get("journal"),
        publication_date=payload.get("publication_date"),
        doi=payload.get("doi"),
        url=payload.get("url"),
        source_domain="pubmed.ncbi.nlm.nih.gov",
        evidence_strength_hint="unknown",
    )


def _direct_pubmed_search_for_claim(
    claim: Claim,
    context: ClaimRetrievalContext,
    *,
    existing_pmids: set[str],
    pass_index: int,
    remaining: int,
) -> list[MedicalArticleRecord]:
    if remaining <= 0:
        return []
    esearch = getattr(context.fetcher.pubmed_client, "esearch", None)
    if not callable(esearch):
        return []

    fetched: list[MedicalArticleRecord] = []
    seen_pmids = set(existing_pmids)
    for query in _direct_pubmed_queries(context.question, claim):
        try:
            pmids = [str(pmid) for pmid in esearch(query, retmax=max(remaining * 4, remaining))]
        except Exception as exc:
            raise PubmedFetcherError("PubMed claim search failed") from exc
        candidates = []
        for pmid in pmids:
            if not pmid or pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)
            candidates.append(
                {
                    "url": PUBMED_URL_TEMPLATE.format(pmid=pmid),
                    "snippet": claim.text,
                    "query": query,
                    "strategy_family": "claim_direct_pubmed_search",
                }
            )
            if len(candidates) >= remaining:
                break
        if not candidates:
            continue
        articles, _result = _fetch_candidate_articles(
            candidates,
            context,
            existing_pmids={article.pmid for article in fetched if article.pmid} | existing_pmids,
            pass_index=pass_index,
        )
        fetched.extend(articles)
        if len(fetched) >= remaining:
            break
    return fetched[:remaining]


def _direct_pubmed_queries(question: str, claim: Claim) -> list[str]:
    lowered = f"{question} {claim.text}".lower()
    queries: list[str] = []
    if (
        ("tylenol" in lowered or "acetaminophen" in lowered or "paracetamol" in lowered)
        and ("advil" in lowered or "ibuprofen" in lowered or "motrin" in lowered)
        and ("fever" in lowered or "antipyretic" in lowered or "antipyresis" in lowered)
    ):
        queries.extend(
            [
                "physician-directed doses ibuprofen acetaminophen over-the-counter doses antipyretic",
                "ibuprofen acetaminophen antipyretic effects",
            ]
        )
    term_query = " ".join(_query_terms(claim))
    if term_query:
        queries.append(term_query)
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(query.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _query_terms(claim: Claim) -> list[str]:
    preferred = [
        "acetaminophen",
        "paracetamol",
        "ibuprofen",
        "antipyretic",
        "antipyresis",
        "fever",
        "analgesic",
        "pain",
        "anti-inflammatory",
        "inflammatory",
        "combination",
        "alternating",
        "tolerability",
        "safety",
        "superior",
        "similar",
    ]
    terms = [term for term in preferred if term in set(claim.evidence_terms)]
    if terms:
        return terms[:8]
    return [word for word in claim.text.lower().split() if len(word.strip(".,;:()")) > 4][:8]


def _dedupe_articles(
    articles: list[MedicalArticleRecord],
    *,
    existing_pmids: set[str | None],
) -> list[MedicalArticleRecord]:
    deduped: list[MedicalArticleRecord] = []
    seen = {pmid for pmid in existing_pmids if pmid}
    for article in articles:
        if not article.pmid or article.pmid in seen:
            continue
        seen.add(article.pmid)
        deduped.append(article)
    return deduped


def _focused_claim_query(question: str, claim: Claim) -> str:
    pieces = [question.strip(), claim.text.strip()]
    return " ".join(piece for piece in pieces if piece)


__all__ = [
    "ClaimRetrievalContext",
    "ClaimVerificationResult",
    "verify_claims_with_retrieval",
]
