from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


DEFAULT_RANKER_MODEL = "gpt-5.4-mini"
DEFAULT_BATCH_SIZE = 10
DEFAULT_TOP_N = 15
MAX_BATCH_SIZE = 25


class RerankerError(RuntimeError):
    """Expected operational failure from the LLM reranker."""


class CandidateScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pmid: str
    relevance: float
    directness: float
    design_strength: float
    population_match: float
    medication_match: float
    outcome_match: float
    include: bool
    reason: str

    @field_validator("pmid")
    @classmethod
    def _pmid_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("pmid must not be empty")
        return value

    @property
    def composite(self) -> float:
        return (
            self.relevance
            + self.directness
            + self.design_strength
            + self.population_match
            + self.medication_match
            + self.outcome_match
        ) / 6.0


class BatchRankingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: list[CandidateScore]


class ArbitrationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pmid: str
    rank: int
    reason: str


class ArbitrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected: list[ArbitrationDecision]
    thin_evidence: bool
    rationale: str


class RankedArticle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pmid: str
    rank: int
    composite_score: float
    include: bool
    reason: str
    arbitration_reason: str
    payload: dict[str, Any]


class RankingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranked: list[RankedArticle]
    thin_evidence: bool
    rationale: str


COMPACT_FIELDS = (
    "pmid",
    "url",
    "title",
    "abstract",
    "journal",
    "publication_date",
    "publication_types",
    "authors",
    "provenance",
)


def _compact_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {field_name: candidate.get(field_name) for field_name in COMPACT_FIELDS}


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def _is_expected_llm_error(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    if getattr(exc, "is_expected_llm_error", False) is True:
        return True
    expected_names = {
        "APIConnectionError",
        "APIError",
        "APITimeoutError",
        "ProviderError",
        "ProviderTransportError",
        "RateLimitError",
        "ServiceUnavailableError",
    }
    return type(exc).__name__ in expected_names


class LLMReranker:
    """Compact GPT-5.4-mini reranker with batched scoring and cross-batch arbitration."""

    def __init__(
        self,
        *,
        llm_client: Any,
        model: str = DEFAULT_RANKER_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        top_n: int = DEFAULT_TOP_N,
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> None:
        if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        self.llm_client = llm_client
        self.model = model
        self.batch_size = batch_size
        self.top_n = top_n
        self.temperature = temperature
        self.max_tokens = max_tokens

    def rank(
        self,
        question: str,
        question_analysis: dict[str, Any] | BaseModel,
        candidates: list[dict[str, Any]],
    ) -> RankingResult:
        if not candidates:
            return RankingResult(ranked=[], thin_evidence=True, rationale="no candidates available")

        analysis_payload = self._coerce_analysis(question_analysis)
        payloads = {c["pmid"]: c for c in candidates if isinstance(c, dict) and c.get("pmid")}
        if not payloads:
            raise RerankerError("candidates must include pmid")

        scored: dict[str, CandidateScore] = {}
        for batch in self._batch(list(payloads.values())):
            batch_result = self._score_batch(question, analysis_payload, batch)
            for score in batch_result.scores:
                if score.pmid in payloads and score.pmid not in scored:
                    scored[score.pmid] = score

        if not scored:
            raise RerankerError("LLM returned no scored candidates")

        arbitration = self._arbitrate(question, analysis_payload, scored)
        return self._materialize(scored, arbitration, payloads)

    def _coerce_analysis(self, question_analysis: dict[str, Any] | BaseModel) -> dict[str, Any]:
        if isinstance(question_analysis, BaseModel):
            return question_analysis.model_dump()
        if isinstance(question_analysis, dict):
            return question_analysis
        raise RerankerError("question_analysis must be a dict or BaseModel")

    def _batch(self, items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        return [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]

    def _score_batch(
        self,
        question: str,
        analysis_payload: dict[str, Any],
        batch: list[dict[str, Any]],
    ) -> BatchRankingResult:
        compact = [_compact_payload(c) for c in batch]
        response = self._call_llm(
            self._scoring_system_prompt(),
            self._scoring_user_prompt(question, analysis_payload, compact),
        )
        try:
            return BatchRankingResult.model_validate(response)
        except ValidationError as exc:
            raise RerankerError("LLM batch scoring response failed validation") from exc

    def _arbitrate(
        self,
        question: str,
        analysis_payload: dict[str, Any],
        scored: dict[str, CandidateScore],
    ) -> ArbitrationResult:
        summary = [
            {
                "pmid": s.pmid,
                "include": s.include,
                "composite_score": round(s.composite, 4),
                "relevance": s.relevance,
                "directness": s.directness,
                "design_strength": s.design_strength,
                "population_match": s.population_match,
                "medication_match": s.medication_match,
                "outcome_match": s.outcome_match,
                "reason": s.reason,
            }
            for s in scored.values()
        ]
        response = self._call_llm(
            self._arbitration_system_prompt(),
            self._arbitration_user_prompt(question, analysis_payload, summary),
        )
        try:
            return ArbitrationResult.model_validate(response)
        except ValidationError as exc:
            raise RerankerError("LLM arbitration response failed validation") from exc

    def _materialize(
        self,
        scored: dict[str, CandidateScore],
        arbitration: ArbitrationResult,
        payloads: dict[str, dict[str, Any]],
    ) -> RankingResult:
        ranked: list[RankedArticle] = []
        seen: set[str] = set()
        for decision in sorted(arbitration.selected, key=lambda d: d.rank):
            if decision.pmid in seen or decision.pmid not in scored:
                continue
            seen.add(decision.pmid)
            score = scored[decision.pmid]
            ranked.append(RankedArticle(
                pmid=decision.pmid,
                rank=len(ranked) + 1,
                composite_score=round(score.composite, 4),
                include=score.include,
                reason=score.reason,
                arbitration_reason=decision.reason,
                payload=payloads[decision.pmid],
            ))
            if len(ranked) >= self.top_n:
                break
        return RankingResult(
            ranked=ranked,
            thin_evidence=arbitration.thin_evidence,
            rationale=arbitration.rationale,
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            response = self.llm_client.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            if _is_expected_llm_error(exc):
                raise RerankerError("Reranker LLM request failed") from exc
            raise

        try:
            return json.loads(_strip_markdown_fences(str(response)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise RerankerError("Reranker LLM response was not valid JSON") from exc

    def _scoring_system_prompt(self) -> str:
        return (
            "You are a biomedical evidence ranker. Score each PubMed candidate against the user's "
            "medication-centric question on relevance, directness, study design strength, population "
            "match, medication or supplement match, and outcome or mechanism match. Each score is a "
            "float between 0.0 and 1.0. Set include=true only when the candidate is genuinely useful "
            "for the question. Use the title, abstract, journal, publication types, year, and "
            "strategy provenance only. Do not assume content beyond the provided fields. Return only "
            "JSON of the form {\"scores\": [...]} with one object per input candidate, where each "
            "object has fields pmid, relevance, directness, design_strength, population_match, "
            "medication_match, outcome_match, include, reason. Do not include markdown."
        )

    def _scoring_user_prompt(
        self,
        question: str,
        analysis_payload: dict[str, Any],
        compact_candidates: list[dict[str, Any]],
    ) -> str:
        return json.dumps({
            "question": question,
            "question_analysis": analysis_payload,
            "candidates": compact_candidates,
        })

    def _arbitration_system_prompt(self) -> str:
        return (
            "You are the final arbiter for a compact PubMed reranker. Given per-candidate scores "
            "across multiple batches, select the top articles for downstream synthesis. Prefer "
            "directly relevant, well-designed evidence with good population and medication match. "
            "Preserve useful strategy provenance diversity when scores are close. Mark thin_evidence "
            "true when the pool genuinely lacks direct, strong evidence for the question. Return "
            "only JSON of the form {\"selected\": [{\"pmid\":..., \"rank\":..., \"reason\":...}], "
            "\"thin_evidence\": bool, \"rationale\": str}. Rank starts at 1. Do not include markdown."
        )

    def _arbitration_user_prompt(
        self,
        question: str,
        analysis_payload: dict[str, Any],
        summary: list[dict[str, Any]],
    ) -> str:
        return json.dumps({
            "question": question,
            "question_analysis": analysis_payload,
            "scored_candidates": summary,
            "max_to_select": self.top_n,
        })
