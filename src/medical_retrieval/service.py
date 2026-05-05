from __future__ import annotations

import logging
from typing import Any, Protocol

from src.medical_retrieval.claim_composer import compose_patient_medication_guide
from src.medical_retrieval.claim_generation import ClaimGenerator
from src.medical_retrieval.claim_verification_service import (
    ClaimRetrievalContext,
    verify_claims_with_retrieval,
)
from src.medical_retrieval.candidate_pool import CandidatePool, DEFAULT_POOL_CANDIDATE_CAP
from src.medical_retrieval.errors import (
    LAYER_AGENT_DISCOVERY,
    LAYER_PUBMED_FETCH,
    LAYER_QUESTION_ANALYSIS,
    LAYER_RANKING,
    PerplexityAgentError,
    PubmedFetcherError,
    QuestionAnalysisError,
    RerankerError,
)
from src.medical_retrieval.metrics import RetrievalMetrics
from src.medical_retrieval.cache import MedicalRetrievalCache
from src.medical_retrieval.perplexity_agent_client import PerplexityDiscoveryResult
from src.medical_retrieval.pubmed_fetcher import (
    NormalizedArticle,
    PubmedFetcher,
    CandidateProvenance,
)
from src.medical_retrieval.question_analysis import (
    MedicationQuestionAnalysis,
    QuestionAnalyzer,
)
from src.medical_retrieval.ranker import LLMReranker, RankingResult
from src.medical_retrieval.schemas import (
    EvidenceStrengthHint,
    MedicalArticleRecord,
    MedicalEvidenceRequest,
    MedicalEvidenceResponse,
    MedicalQuestionAnalysis,
    MedicalQuestionEntities,
    MedicalStrategyExecution,
)
from src.medical_retrieval.source_normalizer import normalize_source_url

logger = logging.getLogger(__name__)


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


class MedicalRetrievalService:
    """Orchestrates question analysis, Agent discovery, fetch, and ranking."""

    def __init__(
        self,
        *,
        analyzer: QuestionAnalyzer,
        agent_client: _AgentClientProtocol,
        fetcher: PubmedFetcher,
        reranker: LLMReranker,
        cache: MedicalRetrievalCache | None = None,
        max_pool_candidates: int = DEFAULT_POOL_CANDIDATE_CAP,
        rxnorm_handler: Any = None,
        max_claim_retrieval_passes: int = 1,
        max_claim_retrieval_articles: int = 3,
    ) -> None:
        self.analyzer = analyzer
        self.agent_client = agent_client
        self.fetcher = fetcher
        self.reranker = reranker
        self.cache = cache
        self.max_pool_candidates = max_pool_candidates
        self.rxnorm_handler = rxnorm_handler
        self.max_claim_retrieval_passes = max_claim_retrieval_passes
        self.max_claim_retrieval_articles = max_claim_retrieval_articles

    def run(self, request: MedicalEvidenceRequest) -> MedicalEvidenceResponse:
        cached = self._lookup_cache(request)
        if cached is not None:
            return cached

        metrics = RetrievalMetrics()
        analysis = self._analyze(request, metrics)
        if analysis is None:
            return self._empty_response(_fallback_question_analysis(), metrics, [])

        pool = CandidatePool(max_candidates=self.max_pool_candidates)
        discovery = self._discover(request, analysis, pool, metrics)
        self._fetch_discovery_candidates_if_needed(discovery, pool, metrics)
        ranking = self._rank(request, analysis, pool, metrics)

        response = self._build_response(request, analysis, discovery, ranking, pool, metrics)
        self._maybe_cache(request, response, metrics)
        return response

    def _lookup_cache(self, request: MedicalEvidenceRequest) -> MedicalEvidenceResponse | None:
        if self.cache is None:
            return None
        cached = self.cache.get(self.cache.key_for(request))
        if cached is None:
            return None
        cached.meta.cached = True
        return cached

    def _maybe_cache(
        self,
        request: MedicalEvidenceRequest,
        response: MedicalEvidenceResponse,
        metrics: RetrievalMetrics,
    ) -> None:
        if self.cache is None or metrics.errors:
            return
        self.cache.set(self.cache.key_for(request), response)

    def _analyze(
        self,
        request: MedicalEvidenceRequest,
        metrics: RetrievalMetrics,
    ) -> MedicationQuestionAnalysis | None:
        try:
            return self.analyzer.analyze(
                request.question,
                conversation_context=[m.model_dump() for m in request.conversation_context],
            )
        except QuestionAnalysisError as exc:
            metrics.add_error(LAYER_QUESTION_ANALYSIS, str(exc))
            return None

    def _discover(
        self,
        request: MedicalEvidenceRequest,
        analysis: MedicationQuestionAnalysis,
        pool: CandidatePool,
        metrics: RetrievalMetrics,
    ) -> PerplexityDiscoveryResult | None:
        handler = _make_pubmed_handler(self.fetcher, pool, metrics)
        rxnorm_handler = self.rxnorm_handler or _default_rxnorm_handler(metrics)
        max_steps = request.hints.max_passes
        try:
            discovery = self.agent_client.discover(
                request.question,
                analysis,
                max_steps=max_steps,
                tool_handlers={
                    "pubmed_normalize_and_fetch": handler,
                    "rxnorm_lookup": rxnorm_handler,
                },
            )
        except PerplexityAgentError as exc:
            metrics.add_error(LAYER_AGENT_DISCOVERY, str(exc))
            return None
        except PubmedFetcherError as exc:
            metrics.add_error(LAYER_PUBMED_FETCH, str(exc))
            return None

        metrics.agent_passes = discovery.agent_passes or 0
        tool_counts = discovery.tool_counts or {}
        metrics.perplexity_searches = int(tool_counts.get("web_search", 0) or 0)
        metrics.pubmed_fetches = self.fetcher.fetch_count
        return discovery

    def _rank(
        self,
        request: MedicalEvidenceRequest,
        analysis: MedicationQuestionAnalysis,
        pool: CandidatePool,
        metrics: RetrievalMetrics,
    ) -> RankingResult | None:
        metrics.candidate_articles = pool.size
        if pool.size == 0:
            return None
        try:
            ranking = self.reranker.rank(request.question, analysis, pool.to_payloads())
        except RerankerError as exc:
            metrics.add_error(LAYER_RANKING, str(exc))
            return None
        metrics.ranked_articles = len(ranking.ranked)
        return ranking

    def _fetch_discovery_candidates_if_needed(
        self,
        discovery: PerplexityDiscoveryResult | None,
        pool: CandidatePool,
        metrics: RetrievalMetrics,
    ) -> None:
        if discovery is None or not discovery.candidates:
            return

        existing_pmids = {candidate.pmid for candidate in pool.candidates()}
        candidates_to_fetch = [
            candidate
            for candidate in discovery.candidates
            if normalize_source_url(candidate.url).pmid not in existing_pmids
        ]
        if not candidates_to_fetch:
            return

        handler = _make_pubmed_handler(self.fetcher, pool, metrics)
        try:
            handler({"candidates": [candidate.model_dump() for candidate in candidates_to_fetch]})
        except PubmedFetcherError as exc:
            metrics.add_error(LAYER_PUBMED_FETCH, str(exc))
            return
        metrics.pubmed_fetches = self.fetcher.fetch_count

    def _build_response(
        self,
        request: MedicalEvidenceRequest,
        analysis: MedicationQuestionAnalysis,
        discovery: PerplexityDiscoveryResult | None,
        ranking: RankingResult | None,
        pool: CandidatePool,
        metrics: RetrievalMetrics,
    ) -> MedicalEvidenceResponse:
        response_analysis = _adapt_analysis(analysis)
        strategies = _adapt_strategies(discovery)
        articles = _adapt_articles(ranking, pool)
        evidence_hint = _evidence_strength_hint(ranking, pool)
        claim_generator_kwargs: dict[str, Any] = {}
        llm_client = getattr(self.analyzer, "llm_client", None)
        if llm_client is not None:
            claim_generator_kwargs = {
                "llm_client": llm_client,
                "model": getattr(self.analyzer, "model", None),
            }
        claims = ClaimGenerator(**claim_generator_kwargs).generate(
            question=request.question,
            articles=articles,
            question_analysis=response_analysis,
        )
        claim_result = verify_claims_with_retrieval(
            claims,
            articles,
            ClaimRetrievalContext(
                question=request.question,
                question_analysis=analysis,
                agent_client=self.agent_client,
                fetcher=self.fetcher,
                max_claim_retrieval_passes=self.max_claim_retrieval_passes,
                max_new_articles_per_claim=self.max_claim_retrieval_articles,
            ),
        )
        metrics.pubmed_fetches = self.fetcher.fetch_count
        metrics.perplexity_searches += claim_result.perplexity_searches
        metrics.agent_passes += claim_result.agent_passes
        articles = _merge_article_records(articles, claim_result.extra_articles)
        composed_answer = compose_patient_medication_guide(claim_result.validated_claims, request.question) or None
        return MedicalEvidenceResponse(
            question_analysis=response_analysis,
            strategies_executed=strategies,
            articles=articles,
            meta=metrics.to_meta(evidence_hint),
            claims=claims,
            validated_claims=claim_result.validated_claims,
            composed_answer=composed_answer,
        )

    def _empty_response(
        self,
        analysis: MedicalQuestionAnalysis,
        metrics: RetrievalMetrics,
        articles: list[MedicalArticleRecord],
    ) -> MedicalEvidenceResponse:
        return MedicalEvidenceResponse(
            question_analysis=analysis,
            strategies_executed=[],
            articles=articles,
            meta=metrics.to_meta("unknown"),
        )


def _make_pubmed_handler(fetcher: PubmedFetcher, pool: CandidatePool, metrics: RetrievalMetrics):
    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        result = fetcher.normalize_and_fetch(arguments)
        articles = [_payload_to_article(p) for p in result.get("normalized", [])]
        pool.add_articles(articles)
        for entry in result.get("unmapped", []):
            reason = entry.get("reason")
            if reason == "pmcid_unmapped":
                metrics.pmcid_mapping_unmapped += 1
        metrics.pmcid_mapping_success += sum(
            1 for a in articles if a.pmcid is not None
        )
        return result

    return handler


def _default_rxnorm_handler(metrics: RetrievalMetrics):
    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        metrics.rxnorm_calls += 1
        return {"normalized": [], "reason": "rxnorm_helper_not_configured"}

    return handler


def _payload_to_article(payload: dict[str, Any]) -> NormalizedArticle:
    provenance = [
        CandidateProvenance(
            query=str(p.get("query") or ""),
            strategy_family=str(p.get("strategy_family") or ""),
            snippet=p.get("snippet") if isinstance(p.get("snippet"), str) else None,
            source_url=p.get("source_url"),
            pass_index=p.get("pass_index"),
        )
        for p in (payload.get("provenance") or [])
    ]
    return NormalizedArticle(
        pmid=str(payload["pmid"]),
        title=payload.get("title"),
        abstract=payload.get("abstract"),
        authors=list(payload.get("authors") or []),
        journal=payload.get("journal"),
        publication_date=payload.get("publication_date"),
        publication_types=list(payload.get("publication_types") or []),
        doi=payload.get("doi"),
        pubmed_url=str(payload.get("url") or ""),
        pmcid=payload.get("pmcid"),
        provenance=provenance,
    )


def _fallback_question_analysis() -> MedicalQuestionAnalysis:
    return MedicalQuestionAnalysis(
        question_type="unknown",
        entities=MedicalQuestionEntities(),
        optional_drug_normalizations=[],
    )


def _adapt_analysis(analysis: MedicationQuestionAnalysis) -> MedicalQuestionAnalysis:
    entities = MedicalQuestionEntities(
        drugs=[m.raw_name for m in analysis.entities.drugs],
        supplements=[m.raw_name for m in analysis.entities.supplements],
        normalized_ingredients=[n.canonical_name for n in analysis.entities.normalized_ingredients],
        conditions=[c.name for c in analysis.entities.conditions],
        populations=[p.name for p in analysis.entities.populations],
        outcomes=[o.name for o in analysis.entities.outcomes],
    )
    optional = [
        c.canonical_name or c.raw_name
        for c in analysis.rxnorm_candidates
        if (c.canonical_name or c.raw_name)
    ]
    return MedicalQuestionAnalysis(
        question_type=analysis.question_type,
        entities=entities,
        optional_drug_normalizations=optional,
    )


def _adapt_strategies(
    discovery: PerplexityDiscoveryResult | None,
) -> list[MedicalStrategyExecution]:
    if discovery is None:
        return []
    return [
        MedicalStrategyExecution(
            strategy_label=s.strategy_label,
            query=s.query,
            domains=list(s.domains),
            result_count=s.result_count,
        )
        for s in discovery.strategies_executed
    ]


def _adapt_articles(
    ranking: RankingResult | None,
    pool: CandidatePool,
) -> list[MedicalArticleRecord]:
    if ranking is None or not ranking.ranked:
        return []
    return [_payload_to_record(article.payload) for article in ranking.ranked]


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


def _merge_article_records(
    base_articles: list[MedicalArticleRecord],
    extra_articles: list[MedicalArticleRecord],
) -> list[MedicalArticleRecord]:
    merged = list(base_articles)
    seen_pmids = {article.pmid for article in merged if article.pmid}
    for article in extra_articles:
        if not article.pmid or article.pmid in seen_pmids:
            continue
        seen_pmids.add(article.pmid)
        merged.append(article)
    return merged


def _evidence_strength_hint(
    ranking: RankingResult | None,
    pool: CandidatePool,
) -> EvidenceStrengthHint:
    if ranking is None:
        return "unknown" if pool.size == 0 else "low"
    if ranking.thin_evidence:
        return "low"
    if len(ranking.ranked) >= 5:
        return "high"
    if len(ranking.ranked) >= 2:
        return "medium"
    return "low"


__all__ = ["MedicalRetrievalService"]
