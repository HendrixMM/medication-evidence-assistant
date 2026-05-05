"""
LLM-based source relevance scoring.

After retrieving sources from PubMed, use an LLM to evaluate
whether each source actually answers the user's question.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SourceRelevanceScore:
    """Relevance score for a retrieved source."""

    pmid: str
    title: str
    relevance_score: float  # 1-5 scale
    reasoning: str
    is_relevant: bool  # True if score >= threshold


class SourceRelevanceScorer:
    """Score source relevance using an LLM."""

    def __init__(
        self,
        llm_client: Any,
        threshold: float = 3.0,
        max_sources: int = 10,
        max_abstract_chars: int = 500,
        model: Optional[str] = "8b",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        self.llm = llm_client
        self.threshold = float(threshold)
        self.max_sources = max(1, int(max_sources))
        self.max_abstract_chars = max(0, int(max_abstract_chars))
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = timeout_seconds

    def score_sources(self, user_query: str, sources: List[Dict[str, Any]]) -> List[SourceRelevanceScore]:
        """
        Score each source for relevance to user query.

        Args:
            user_query: Original patient question
            sources: List of PubMed articles with title/abstract

        Returns:
            List of scored sources, sorted by relevance
        """
        if not sources:
            return []

        prepared_sources = self._prepare_sources(sources[: self.max_sources])
        prompt = self._build_batch_prompt(user_query, prepared_sources)
        response_text = self._call_llm(prompt)
        scores = self._parse_scores(response_text, prepared_sources)
        scores.sort(key=lambda x: x.relevance_score, reverse=True)
        return scores

    def _prepare_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Normalize sources for prompt construction."""
        prepared: List[Dict[str, str]] = []
        for idx, source in enumerate(sources, 1):
            pmid = str(source.get("pmid") or "").strip()
            if not pmid:
                pmid = f"unknown_{idx}"
            title = str(source.get("title") or "No title").strip() or "No title"
            abstract_value = source.get("abstract") or source.get("content") or ""
            if isinstance(abstract_value, list):
                abstract_value = " ".join(str(chunk) for chunk in abstract_value if chunk)
            abstract = str(abstract_value).strip() or "No abstract"
            if self.max_abstract_chars and len(abstract) > self.max_abstract_chars:
                abstract = abstract[: self.max_abstract_chars].rstrip()
            prepared.append({"pmid": pmid, "title": title, "abstract": abstract})
        return prepared

    def _build_batch_prompt(self, user_query: str, sources: List[Dict[str, str]]) -> str:
        """Build prompt to score multiple sources at once."""
        sources_text = ""
        for idx, source in enumerate(sources, 1):
            sources_text += (
                f"\n\nSource {idx} (PMID: {source['pmid']}):\n"
                f"Title: {source['title']}\n"
                f"Abstract: {source['abstract']}\n"
            )

        prompt = f"""You are a medical literature relevance expert. Rate how well each source answers the user's question.

USER QUESTION: "{user_query}"

SOURCES TO EVALUATE:
{sources_text}

For each source, provide:
1. Relevance score (1-5):
   - 5 = Directly answers the question with specific information
   - 4 = Highly relevant, addresses core aspects
   - 3 = Moderately relevant, provides related information
   - 2 = Tangentially related, minimal useful info
   - 1 = Not relevant, completely off-topic

2. Brief reasoning (1 sentence)

Return ONLY valid JSON array:
[
  {{"pmid": "123", "score": 4.5, "reasoning": "..."}},
  {{"pmid": "456", "score": 2.0, "reasoning": "..."}}
]
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.model:
            kwargs["model"] = self.model
        if self.timeout_seconds is not None:
            kwargs["timeout"] = self.timeout_seconds

        try:
            response = self.llm.generate(messages=messages, **kwargs)
        except TypeError:
            response = self.llm.generate(messages=messages)
        return response if isinstance(response, str) else str(response)

    def _parse_scores(self, response: str, sources: List[Dict[str, str]]) -> List[SourceRelevanceScore]:
        """Parse LLM response into scores."""
        try:
            json_str = self._extract_json_array(response)
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                parsed = [parsed]
            scores_by_pmid: Dict[str, Dict[str, Any]] = {}
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    pmid = str(item.get("pmid") or "").strip()
                    if not pmid:
                        continue
                    score = self._coerce_score(item.get("score"))
                    if score is None:
                        continue
                    reasoning = str(item.get("reasoning") or "").strip() or "No reasoning provided."
                    scores_by_pmid[pmid] = {"score": score, "reasoning": reasoning}

            results: List[SourceRelevanceScore] = []
            for source in sources:
                pmid = source["pmid"]
                title = source["title"]
                entry = scores_by_pmid.get(pmid)
                if entry:
                    score_value = entry["score"]
                    reasoning = entry["reasoning"]
                else:
                    score_value = float(self.threshold)
                    reasoning = "No score returned; defaulting to neutral relevance."
                results.append(
                    SourceRelevanceScore(
                        pmid=pmid,
                        title=title,
                        relevance_score=score_value,
                        reasoning=reasoning,
                        is_relevant=(score_value >= self.threshold),
                    )
                )
            return results
        except Exception as exc:
            logger.error("Failed to parse relevance scores: %s", exc)
            return self._neutral_scores(sources, reason="Scoring failed")

    def _extract_json_array(self, response: str) -> str:
        text = response.strip()
        if "```" in text:
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            else:
                text = text.split("```", 1)[1].split("```", 1)[0].strip()
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    def _coerce_score(self, raw_score: Any) -> Optional[float]:
        try:
            score_value = float(raw_score)
        except (TypeError, ValueError):
            return None
        return max(1.0, min(5.0, score_value))

    def _neutral_scores(self, sources: List[Dict[str, str]], reason: str) -> List[SourceRelevanceScore]:
        neutral_score = max(1.0, min(5.0, float(self.threshold)))
        results: List[SourceRelevanceScore] = []
        for source in sources:
            results.append(
                SourceRelevanceScore(
                    pmid=source["pmid"],
                    title=source["title"],
                    relevance_score=neutral_score,
                    reasoning=reason,
                    is_relevant=True,
                )
            )
        return results
