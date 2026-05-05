"""Consumer-answer translation adapter for agent workflows."""
from __future__ import annotations

import json
import logging
import re

from guardrails.constants import MEDICAL_DISCLAIMER
from src.nvidia_llm_client import NVIDIALLMAPIError, NVIDIALLMClient

from ..config import AgentConfig
from ..models import ConsumerAnswer, EvidenceSummary, QueryIntent, RetrievedArticle, Source

logger = logging.getLogger(__name__)


class TranslateForConsumerTool:
    """Generate a patient-facing answer from structured evidence."""

    def __init__(
        self,
        llm_client: NVIDIALLMClient,
        config: AgentConfig | None = None,
        model: str = "70b",
        temperature: float = 0.3,
        max_tokens: int = 1200,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or AgentConfig()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def execute(
        self,
        user_question: str,
        intent: QueryIntent,
        evidence: EvidenceSummary,
    ) -> ConsumerAnswer:
        ranked_for_context = sorted(
            evidence.ranked_articles,
            key=lambda article: article.relevance_score,
            reverse=True,
        )[:10]
        context_string = "\n\n---\n\n".join(
            f"[{article.source_system}: {article.pmid}] {article.title or ''}\n"
            f"{article.url or ''}\n{article.abstract or ''}".rstrip()
            for article in ranked_for_context
        )
        if not context_string:
            context_string = "No supporting MedlinePlus pages were retrieved for this question."
        formatted_prompt = self._research_informed_prompt(user_question, intent, context_string)
        system_prompt = (
            "You are a consumer medication information assistant. Return a JSON object with keys "
            "answer, uncertainties, safety_warnings, and verdict. Generate helpful high-level "
            "guidance first, use MedlinePlus material as supporting references rather than hard "
            "constraints, and clearly avoid personalized diagnosis or treatment orders. "
            f"{MEDICAL_DISCLAIMER}"
        )
        sources = self._safe_build_sources(evidence.ranked_articles)

        try:
            raw_text = self.llm_client.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_prompt},
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            parsed = self._parse_response(raw_text)
        except (NVIDIALLMAPIError, TypeError, ValueError, AttributeError):
            logger.exception("TranslateForConsumerTool failed due to LLM or parsing error")
            return ConsumerAnswer(
                answer=self._generation_fallback_answer(has_sources=bool(sources)),
                sources=sources,
                evidence_level=evidence.evidence_level,
                uncertainties=[] if sources else ["No supporting MedlinePlus sources were available."],
                safety_warnings=[] if sources else ["Ask a clinician or pharmacist before making medication decisions."],
                disclaimer=MEDICAL_DISCLAIMER,
                verdict=None,
            )

        answer_text = str(parsed["answer"]).strip() or "I found relevant studies but could not generate a summary."
        return ConsumerAnswer(
            answer=answer_text,
            sources=sources,
            evidence_level=evidence.evidence_level,
            uncertainties=parsed["uncertainties"],
            safety_warnings=parsed["safety_warnings"],
            disclaimer=MEDICAL_DISCLAIMER,
            verdict=parsed["verdict"],
        )

    @staticmethod
    def _research_informed_prompt(user_question: str, intent: QueryIntent, context_string: str) -> str:
        return f"""Patient question: {user_question}

Intent: {intent.value}

Write a concise consumer guide using general medication knowledge as the primary basis. Use the
MedlinePlus material below as backing references only. Do not block useful high-level advice just
because a source does not support every sentence. Prefer broad, practical sections such as
"Quick comparison", "Which is better?", "Can you take them together?", and "Bottom line" when
they fit the question.

Avoid niche or overly specific study details unless they materially change safety guidance.
Include a strong medical disclaimer and make clear this is not personalized medical advice.

Supporting MedlinePlus material:
{context_string}
"""

    @staticmethod
    def _generation_fallback_answer(*, has_sources: bool) -> str:
        if has_sources:
            return (
                "I found relevant MedlinePlus pages but had trouble generating "
                "a consumer summary. Here are the sources I found:"
            )
        return (
            "I could not generate a full research-informed summary right now. "
            "Please ask a clinician or pharmacist for personal medication decisions."
        )

    def _parse_response(self, raw_text: str) -> dict[str, object]:
        cleaned = raw_text.strip()
        if "```" in cleaned:
            if "```json" in cleaned:
                cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
            else:
                cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "answer": raw_text,
                "uncertainties": [],
                "safety_warnings": [],
                "verdict": None,
            }

        if isinstance(payload, list) and payload:
            first_item = payload[0]
            if isinstance(first_item, dict):
                payload = first_item
            else:
                return {
                    "answer": str(first_item),
                    "uncertainties": [],
                    "safety_warnings": [],
                    "verdict": None,
                }
        if not isinstance(payload, dict):
            return {
                "answer": str(payload),
                "uncertainties": [],
                "safety_warnings": [],
                "verdict": None,
            }

        answer = self._extract_answer_text(payload, raw_text)
        return {
            "answer": answer,
            "uncertainties": self._normalize_text_list(payload.get("uncertainties")),
            "safety_warnings": self._normalize_text_list(payload.get("safety_warnings")),
            "verdict": payload.get("verdict"),
        }

    @staticmethod
    def _extract_answer_text(payload: dict[str, object], raw_text: str) -> str:
        for key in ("answer", "summary", "text", "content", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if value not in (None, ""):
                text = str(value).strip()
                if text:
                    return text

        cleaned_raw_text = raw_text.strip()
        if cleaned_raw_text:
            return cleaned_raw_text
        return "I found relevant studies but could not generate a summary."

    def _build_sources(self, articles: list[RetrievedArticle]) -> list[Source]:
        ranked_articles = sorted(
            articles,
            key=lambda article: article.relevance_score,
            reverse=True,
        )
        threshold_matches = [
            article for article in ranked_articles if article.relevance_score >= self.config.source_relevance_threshold
        ]
        selected_articles = threshold_matches or ranked_articles[:10]
        return [
            Source(
                pmid=article.pmid,
                title=article.title or "",
                authors=list(article.authors),
                journal=article.journal,
                year=self._extract_year(article.publication_date),
                url=article.url,
                relevance_score=article.relevance_score,
                study_type=article.publication_types[0] if article.publication_types else "unknown",
                snippet=(article.abstract[:200] if article.abstract else None),
            )
            for article in selected_articles
        ]

    def build_sources(self, articles: list[RetrievedArticle]) -> list[Source]:
        """Public source-building method for orchestration use."""
        return self._safe_build_sources(articles)

    def _safe_build_sources(self, articles: list[RetrievedArticle]) -> list[Source]:
        try:
            return self._build_sources(articles)
        except Exception:
            logger.exception("TranslateForConsumerTool failed to build sources")
            return []

    @staticmethod
    def _normalize_text_list(value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return []

    @staticmethod
    def _extract_year(publication_date: str | None) -> int | None:
        if not publication_date:
            return None
        match = re.search(r"\b(19|20)\d{2}\b", publication_date)
        if match is None:
            return None
        return int(match.group(0))
