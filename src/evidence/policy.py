from __future__ import annotations

import os
from typing import Any

from src.clients.openai_llm_client import OpenAILLMClient

from .errors import capture_expected_failure
from .metrics import log_event
from .query_builder import QueryBuilder
from .ranking_filter import RankingFilter
from .rxnorm_resolver import RxNormResolver
from .schemas import (
    ArticleRecord,
    ErrorRecord,
    EvidenceRequest,
    EvidenceResponse,
    NormalizedDrug,
    QueryExecution,
    ResponseMeta,
)
from .llm_scorer import LLMScorer
from .pubmed_rate_limited import RateLimitedPubMedClient

SEMANTIC_RELEVANCE_MINIMUM = 0.45


def enforce_query_budget(*, current_used: int, additional_queries: int, maximum: int) -> int:
    return min(maximum, current_used + max(0, additional_queries))


def compute_evidence_strength_hint(response: EvidenceResponse) -> str:
    scores = [article.llm_relevance_score for article in response.articles if article.llm_relevance_score is not None]
    if not scores:
        return "low"
    mean_score = sum(scores) / len(scores)
    if mean_score >= 0.75:
        return "high"
    if mean_score >= 0.45:
        return "medium"
    return "low"


class EvidencePolicy:
    def __init__(
        self,
        *,
        resolver: RxNormResolver | None = None,
        query_builder: QueryBuilder | None = None,
        pubmed_client: RateLimitedPubMedClient | None = None,
        ranking_filter: RankingFilter | None = None,
        llm_scorer: LLMScorer | None = None,
        query_budget_max: int | None = None,
        llm_scoring_candidates: int | None = None,
    ) -> None:
        self.resolver = resolver or RxNormResolver()
        llm_client = OpenAILLMClient(enable_logging=False) if (query_builder is None or llm_scorer is None) else None
        self.query_builder = query_builder or QueryBuilder(llm_client=llm_client)
        self.pubmed_client = pubmed_client or RateLimitedPubMedClient()
        self.ranking_filter = ranking_filter or RankingFilter()
        self.llm_scorer = llm_scorer or LLMScorer(llm_client=llm_client)
        self.query_budget_max = query_budget_max or int(os.getenv("EVIDENCE_QUERY_BUDGET_MAX", "12"))
        self.llm_scoring_candidates = llm_scoring_candidates or int(
            os.getenv("EVIDENCE_MAX_LLM_SCORING_CANDIDATES", "15")
        )

    def _context_entry_to_mapping(self, item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return dict(item)
        if isinstance(item, NormalizedDrug):
            return {
                "raw_name": item.raw_name,
                "canonical_name": item.generic_name,
                "match_type": item.match_type,
                "rxnorm_cui": item.rxnorm_cui,
            }
        if hasattr(item, "model_dump"):
            dumped = item.model_dump()
            if isinstance(dumped, dict):
                return dict(dumped)
        return {
            "raw_name": getattr(item, "raw_name", None),
            "canonical_name": getattr(item, "canonical_name", getattr(item, "generic_name", None)),
            "match_type": getattr(item, "match_type", None),
            "rxnorm_cui": getattr(item, "rxnorm_cui", None),
            "confidence": getattr(item, "confidence", None),
            "aliases": getattr(item, "aliases", None),
        }

    def _public_drug_from_context_entry(self, item: Any) -> NormalizedDrug:
        if isinstance(item, NormalizedDrug):
            return item
        payload = self._context_entry_to_mapping(item)
        raw_name = str(payload.get("raw_name") or "")
        canonical_name = str(payload.get("canonical_name") or payload.get("generic_name") or raw_name)
        match_type = str(payload.get("match_type") or "none")
        if match_type not in {"exact", "approximate", "none"}:
            match_type = "none"
        return NormalizedDrug(
            raw_name=raw_name,
            generic_name=canonical_name,
            match_type=match_type,
            rxnorm_cui=str(payload["rxnorm_cui"]) if payload.get("rxnorm_cui") else None,
        )

    def _context_dict_from_public_drug(self, drug: NormalizedDrug) -> dict[str, Any]:
        return {
            "raw_name": drug.raw_name,
            "canonical_name": drug.generic_name,
            "match_type": drug.match_type,
            "rxnorm_cui": drug.rxnorm_cui,
            "confidence": 1.0 if drug.match_type == "exact" else 0.6 if drug.match_type == "approximate" else 0.0,
            "aliases": [name for name in (drug.raw_name, drug.generic_name) if name],
        }

    def _context_dict_from_entry(self, item: Any) -> dict[str, Any]:
        if isinstance(item, NormalizedDrug):
            return self._context_dict_from_public_drug(item)
        payload = self._context_entry_to_mapping(item)
        drug = self._public_drug_from_context_entry(payload)
        confidence = payload.get("confidence")
        aliases = payload.get("aliases")
        return {
            "raw_name": drug.raw_name,
            "canonical_name": drug.generic_name,
            "match_type": drug.match_type,
            "rxnorm_cui": drug.rxnorm_cui,
            "confidence": (
                float(confidence)
                if confidence is not None
                else 1.0 if drug.match_type == "exact" else 0.6 if drug.match_type == "approximate" else 0.0
            ),
            "aliases": [str(alias) for alias in aliases if alias]
            if isinstance(aliases, (list, tuple))
            else [name for name in (drug.raw_name, drug.generic_name) if name],
        }

    def _build_scoring_pool(
        self,
        *,
        filtered_articles: list[ArticleRecord],
        pmid_sources: dict[str, set[str]],
        strategy_order: list[str],
    ) -> list[ArticleRecord]:
        seeded: list[ArticleRecord] = []
        seen_pmids: set[str] = set()
        by_strategy: dict[str, list[ArticleRecord]] = {}
        ordered_strategies = list(dict.fromkeys(strategy_order))

        for article in filtered_articles:
            for strategy_label in pmid_sources.get(article.pmid, set()):
                by_strategy.setdefault(strategy_label, []).append(article)

        for strategy_label in ordered_strategies:
            if strategy_label not in by_strategy:
                continue
            strategy_articles = sorted(
                by_strategy[strategy_label],
                key=lambda item: item.heuristic_score,
                reverse=True,
            )
            top_article = strategy_articles[0]
            if top_article.pmid not in seen_pmids:
                seeded.append(top_article)
                seen_pmids.add(top_article.pmid)
            if len(seeded) >= self.llm_scoring_candidates:
                return seeded[: self.llm_scoring_candidates]

        remainder = [article for article in filtered_articles if article.pmid not in seen_pmids]
        return (seeded + remainder)[: self.llm_scoring_candidates]

    async def run(self, request: EvidenceRequest, *, query_budget_remaining: int | None = None) -> EvidenceResponse:
        errors: list[ErrorRecord] = []
        budget_max = self.query_budget_max if query_budget_remaining is None else min(self.query_budget_max, query_budget_remaining)

        fallback_normalized_drugs = [
            NormalizedDrug(raw_name=name, generic_name=name, match_type="none", rxnorm_cui=None)
            for name in request.drug_names
        ]
        fallback_drug_contexts = [
            {
                "raw_name": name,
                "canonical_name": name,
                "match_type": "none",
                "rxnorm_cui": None,
                "confidence": 0.0,
                "aliases": [name],
            }
            for name in request.drug_names
        ]

        def _resolve_with_context() -> tuple[list[NormalizedDrug], list[dict[str, Any]]]:
            if hasattr(self.resolver, "resolve_names_with_context"):
                resolved_contexts = self.resolver.resolve_names_with_context(request.drug_names)
                return (
                    [self._public_drug_from_context_entry(item) for item in resolved_contexts],
                    [self._context_dict_from_entry(item) for item in resolved_contexts],
                )
            normalized = self.resolver.resolve_names(request.drug_names)
            return (
                normalized,
                [self._context_dict_from_public_drug(drug) for drug in normalized],
            )

        (normalized_drugs, drug_contexts), resolver_error = capture_expected_failure(
            "rxnorm",
            _resolve_with_context,
            (fallback_normalized_drugs, fallback_drug_contexts),
        )
        if resolver_error:
            errors.append(resolver_error)

        normalization_complete = all(drug.match_type != "none" for drug in normalized_drugs) if normalized_drugs else False

        query_candidates, query_builder_error = capture_expected_failure(
            "query_builder",
            lambda: self.query_builder.build_queries(
                question=request.question,
                intent=request.intent,
                normalized_drugs=normalized_drugs,
                drug_contexts=drug_contexts,
            ),
            None,
        )
        if not query_candidates:
            query_candidates = [self._build_fallback_query(request=request, normalized_drugs=normalized_drugs)]
        if query_builder_error:
            errors.append(query_builder_error)

        query_candidates = query_candidates[: min(3, budget_max)]
        queries_executed: list[QueryExecution] = []
        all_articles: dict[str, dict[str, Any]] = {}
        pmid_sources: dict[str, set[str]] = {}
        pubmed_calls = 0

        for candidate in query_candidates:
            articles, pubmed_error = capture_expected_failure(
                "pubmed",
                lambda candidate=candidate: self.pubmed_client.search_and_fetch(candidate.query, max_items=15),
                [],
            )
            pubmed_calls += 1
            if pubmed_error:
                errors.append(pubmed_error)
            queries_executed.append(
                QueryExecution(
                    query=candidate.query,
                    strategy_label=candidate.strategy_label,
                    result_count=len(articles),
                )
            )
            for article in articles:
                pmid = str(article.get("pmid") or "")
                if not pmid:
                    continue
                pmid_sources.setdefault(pmid, set()).add(candidate.strategy_label)
                if pmid not in all_articles:
                    all_articles[pmid] = article

        drug_term_groups = [
            list(
                dict.fromkeys(
                    str(term)
                    for term in (
                        [context.get("raw_name"), context.get("canonical_name")]
                        + list(context.get("aliases") or [])
                    )
                    if term
                )
            )
            for context in drug_contexts
            if context.get("raw_name") or context.get("canonical_name") or context.get("aliases")
        ]
        low_confidence_normalization = any(float(item.get("confidence", 0.0)) < 0.75 for item in drug_contexts)
        is_multi_group_request = request.intent in {"interactions", "comparison"} and len(drug_term_groups) > 1
        filtered_articles = self.ranking_filter.filter_and_rank(
            list(all_articles.values()),
            drug_term_groups=drug_term_groups,
            intent=request.intent,
            require_all_groups=request.intent in {"interactions", "comparison"},
        )
        if not filtered_articles and low_confidence_normalization and not is_multi_group_request:
            filtered_articles = self.ranking_filter.filter_and_rank(
                list(all_articles.values()),
                drug_term_groups=None,
                intent=request.intent,
                require_all_groups=False,
            )
        scoring_pool = self._build_scoring_pool(
            filtered_articles=filtered_articles,
            pmid_sources=pmid_sources,
            strategy_order=[candidate.strategy_label for candidate in query_candidates],
        )
        scored_articles, scorer_fallback_used = self.llm_scorer.score_articles(
            question=request.question,
            intent=request.intent,
            normalized_drug_names=[drug.generic_name for drug in normalized_drugs if drug.generic_name],
            articles=scoring_pool,
        )
        if scorer_fallback_used:
            errors.append(ErrorRecord(layer="llm_scorer", reason="fallback heuristic ordering used"))

        if scorer_fallback_used:
            top_articles = list(scored_articles)[:5]
        else:
            top_articles = [
                article
                for article in scored_articles
                if article.llm_relevance_score is not None and article.llm_relevance_score >= SEMANTIC_RELEVANCE_MINIMUM
            ][:5]
        meta = ResponseMeta(
            pubmed_calls=pubmed_calls,
            articles_considered=len(all_articles),
            articles_heuristic_filtered=max(0, len(all_articles) - len(filtered_articles)),
            articles_llm_scored=len(scoring_pool),
            normalization_used=bool(request.drug_names),
            normalization_complete=normalization_complete,
            evidence_strength_hint="low",
            cached=False,
            cost_estimate_usd=round(0.005 + 0.001 * len(query_candidates) + 0.0005 * len(top_articles), 4),
            query_budget_used=enforce_query_budget(
                current_used=0,
                additional_queries=len(query_candidates),
                maximum=budget_max,
            ),
            query_budget_max=budget_max,
            errors=errors,
        )
        response = EvidenceResponse(
            normalized_drugs=normalized_drugs,
            queries_executed=queries_executed,
            articles=[self._coerce_article(article) for article in top_articles],
            meta=meta,
        )
        response.meta.evidence_strength_hint = (
            "low" if query_builder_error or scorer_fallback_used else compute_evidence_strength_hint(response)
        )
        log_event("policy_complete", query_budget_used=response.meta.query_budget_used, articles=len(response.articles))
        return response

    def _coerce_article(self, article: ArticleRecord | dict[str, Any]) -> ArticleRecord:
        if isinstance(article, ArticleRecord):
            return article
        return ArticleRecord.model_validate(article)

    def _build_fallback_query(
        self,
        *,
        request: EvidenceRequest,
        normalized_drugs: list[NormalizedDrug],
    ):
        terms = [item.generic_name for item in normalized_drugs] or request.drug_names or [request.question]
        keywords = {
            "side_effects": "adverse effects",
            "benefits": "benefit efficacy",
            "interactions": "drug interaction",
            "comparison": "comparison",
            "dosage": "dosage",
            "safety": "safety",
        }
        from .schemas import QueryCandidate

        return QueryCandidate(
            query=f"{' '.join(terms)} {keywords.get(request.intent, request.intent)}".strip(),
            strategy_label="fallback",
        )
