from __future__ import annotations

import json
import math
import os
from json import JSONDecodeError
from typing import Any

from src.clients.openai_llm_client import OpenAILLMAPIError

from .schemas import ArticleRecord


class LLMScorer:
    def __init__(self, *, llm_client: Any, model: str | None = None) -> None:
        self.llm_client = llm_client
        self.model = model or os.getenv("LLM_MODEL", "gpt-5.4-mini")

    def score_articles(
        self,
        *,
        question: str,
        intent: str,
        normalized_drug_names: list[str],
        articles: list[ArticleRecord],
    ) -> tuple[list[ArticleRecord], bool]:
        if not articles:
            return [], False
        prompt = json.dumps(
            {
                "question": question,
                "intent": intent,
                "normalized_drug_names": normalized_drug_names,
                "articles": [
                    {
                        "pmid": article.pmid,
                        "title": article.title,
                        "abstract": article.abstract,
                        "study_type": article.study_type,
                    }
                    for article in articles
                ],
                "instructions": [
                    "Return JSON with a 'scores' array. Each item must include pmid, score (0-1), and reason.",
                    "Judge whether each article directly answers the user's question, not whether it is loosely related.",
                    "Anchor the score to the named drug context in normalized_drug_names and the specific question being asked.",
                    "High scores require evidence that would directly help answer the question for that drug.",
                    "Tangential, adjacent, background-only, or class-level evidence should score below 0.45.",
                ],
            }
        )
        try:
            raw = self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=800,
            )
        except OpenAILLMAPIError:
            ranked = [article.model_copy(update={"llm_relevance_score": None, "llm_reason": None}) for article in articles]
            ranked.sort(key=lambda item: item.heuristic_score, reverse=True)
            return ranked, True
        try:
            expected_pmids = [article.pmid for article in articles]
            scores = self._parse_scores(self._extract_json(raw), expected_pmids=expected_pmids)
        except (JSONDecodeError, ValueError):
            ranked = [article.model_copy(update={"llm_relevance_score": None, "llm_reason": None}) for article in articles]
            ranked.sort(key=lambda item: item.heuristic_score, reverse=True)
            return ranked, True

        ranked: list[ArticleRecord] = []
        for article in articles:
            score = scores.get(article.pmid)
            ranked.append(
                article.model_copy(
                    update={
                        "llm_relevance_score": score["score"] if score and score.get("score") is not None else None,
                        "llm_reason": str(score.get("reason")) if score and score.get("reason") else None,
                    }
                )
            )
        ranked.sort(
            key=lambda item: (
                item.llm_relevance_score if item.llm_relevance_score is not None else -1.0,
                item.heuristic_score,
            ),
            reverse=True,
        )
        return ranked, False

    def _extract_json(self, raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        return json.loads(cleaned)

    def _parse_scores(self, payload: dict[str, Any], *, expected_pmids: list[str]) -> dict[str, dict[str, Any]]:
        raw_scores = payload.get("scores")
        if not isinstance(raw_scores, list):
            raise ValueError("Malformed scorer payload: scores must be a list")

        scores: dict[str, dict[str, Any]] = {}
        expected_pmid_set = set(expected_pmids)
        for item in raw_scores:
            if not isinstance(item, dict):
                raise ValueError("Malformed scorer payload: score items must be objects")
            pmid = item.get("pmid")
            if pmid is None or str(pmid).strip() == "":
                raise ValueError("Malformed scorer payload: pmid is required")
            normalized_pmid = str(pmid)
            if normalized_pmid in scores:
                raise ValueError("Malformed scorer payload: duplicate pmid")
            if "score" not in item:
                raise ValueError("Malformed scorer payload: score is required")
            reason = item.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("Malformed scorer payload: reason is required")
            score = self._validate_score(item["score"])
            scores[normalized_pmid] = {
                "pmid": normalized_pmid,
                "score": score,
                "reason": reason,
            }
        if set(scores) != expected_pmid_set:
            raise ValueError("Malformed scorer payload: pmid coverage mismatch")
        return scores

    def _validate_score(self, raw_score: Any) -> float:
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("Malformed scorer payload: score must be numeric") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Malformed scorer payload: score must be finite and between 0 and 1")
        return score
