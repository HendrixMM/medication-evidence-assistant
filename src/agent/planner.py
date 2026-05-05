"""Planning utilities for the agent workflow."""
from __future__ import annotations

import json
import logging
import re

from src.llm_query_generator import validate_pubmed_query
from src.nvidia_llm_client import NVIDIALLMClient

from .config import AgentConfig
from .models import EvaluationVerdict, PlannedQuery, QueryIntent, QueryPlan, SubQuestion

logger = logging.getLogger(__name__)

_INTENT_FALLBACK_TERMS: dict[str, str] = {
    "drug_interaction": "interaction[tiab]",
    "side_effects": "adverse effects[tiab]",
    "dosage": "dosage[tiab]",
    "timing": "pharmacokinetics[tiab]",
    "mechanism": "pharmacology[tiab]",
    "comparison": "comparative study[tiab]",
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "with",
}

# Tokens that describe the *type* of question (side effects, dosage, taking...)
# rather than a searchable biomedical concept. Excluded when picking the lead
# PubMed keyword so "side effects of metformin" doesn't search for "side".
_GENERIC_MEDICAL_TOKENS = {
    "side",
    "effect",
    "effects",
    "dose",
    "doses",
    "dosage",
    "dosages",
    "taking",
    "take",
    "takes",
    "taken",
    "use",
    "uses",
    "using",
    "used",
    "interaction",
    "interactions",
    "adverse",
    "reaction",
    "reactions",
    "symptom",
    "symptoms",
}


def _normalize_str_list(value: object) -> list[str]:
    """Normalize an LLM response field to a list of non-empty strings.

    Treats a bare string as a single-item list, accepts list/tuple by filtering
    to non-empty strings, and rejects other types to an empty list.
    """
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _normalize_planner_model(model_name: str | None, provider: str = "nvidia") -> str:
    """Resolve to a client-supported model identifier.

    For NVIDIA, falls back to "70b" when unset or unsupported. For OpenAI, passes
    the configured model name through unchanged (empty string lets the OpenAI
    client pick its default).
    """
    if (provider or "nvidia").lower() == "openai":
        return (model_name or "").strip()

    if not model_name or not model_name.strip():
        return "70b"
    normalized = model_name.strip()
    supported = set(NVIDIALLMClient.MODELS) | set(NVIDIALLMClient.MODELS.values())
    if normalized in supported:
        return normalized
    return "70b"


class QueryPlanner:
    """LLM-backed planner for multi-step retrieval."""

    def __init__(self, llm_client: NVIDIALLMClient, config: AgentConfig) -> None:
        self.llm_client = llm_client
        self.config = config
        self.model = _normalize_planner_model(config.planner_model, config.llm_provider)

    def plan(self, user_question: str) -> QueryPlan:
        system_prompt = (
            "Return a JSON object with keys normalized_question, intent, drugs_identified, sub_questions, "
            "planned_queries, and reasoning. sub_questions must be a list of objects with text, rationale, and "
            "priority. planned_queries must be a list of objects with pubmed_query, sub_question_index, and "
            "strategy_label."
        )
        try:
            response = self.llm_client.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question},
                ],
                model=self.model,
                temperature=0.2,
                max_tokens=800,
            )
            payload = json.loads(self._strip_markdown_fences(response))
            intent_str = str(payload.get("intent") or "unknown")
            drugs_identified = _normalize_str_list(payload.get("drugs_identified"))
            planned_queries = []
            for entry in payload.get("planned_queries", []):
                query = str((entry or {}).get("pubmed_query") or "").strip()
                is_valid, error = validate_pubmed_query(query, intent_str, drugs_identified, user_question)
                if is_valid:
                    planned_queries.append(entry)
                else:
                    logger.warning("Discarding invalid planned query: %s", error)

            if not planned_queries:
                planned_queries = [item.model_dump() for item in self._fallback_keyword_query(
                    user_question, drugs_identified, intent=intent_str, substances=drugs_identified
                )]

            normalized_intent = self._coerce_intent(intent_str)
            return QueryPlan.model_validate(
                {
                    "normalized_question": payload.get("normalized_question") or user_question,
                    "intent": normalized_intent,
                    "drugs_identified": drugs_identified,
                    "sub_questions": payload.get("sub_questions") or [],
                    "planned_queries": planned_queries,
                    "reasoning": payload.get("reasoning") or "LLM-generated plan",
                }
            )
        except Exception:
            fallback_queries = self._fallback_keyword_query(user_question, [])
            return QueryPlan(
                normalized_question=user_question,
                intent=QueryIntent.unknown,
                sub_questions=[
                    SubQuestion(
                        text=user_question,
                        rationale="Fallback plan created after planner failure.",
                        priority=1,
                    )
                ],
                planned_queries=fallback_queries,
                drugs_identified=[],
                reasoning="Fallback plan generated due to planner failure.",
            )

    def refine_plan(self, original_plan: QueryPlan, verdict: EvaluationVerdict) -> list[PlannedQuery]:
        prompt = json.dumps(
            {
                "normalized_question": original_plan.normalized_question,
                "coverage_gaps": verdict.coverage_gaps,
                "recommendation": verdict.recommendation,
                "failure_modes": verdict.failure_modes,
                "subquestion_coverages": [item.model_dump() for item in verdict.subquestion_coverages],
            }
        )
        try:
            response = self.llm_client.generate(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Use both failure_modes and subquestion_coverages to decide whether to broaden entities, "
                            "narrow outcomes, or pivot to mechanism evidence. Prioritize uncovered "
                            "subquestion_coverages by using each uncovered sub_question_text to target refined "
                            "queries. Return a JSON array of 1-2 objects with pubmed_query, sub_question_index, "
                            "and strategy_label."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                temperature=0.2,
                max_tokens=400,
            )
            payload = json.loads(self._strip_markdown_fences(response))
            if isinstance(payload, dict):
                payload = [payload]
            refined_queries: list[PlannedQuery] = []
            for entry in payload:
                query = str((entry or {}).get("pubmed_query") or "").strip()
                try:
                    sub_question_index = int((entry or {}).get("sub_question_index", 0))
                except (TypeError, ValueError):
                    sub_question_index = 0
                is_valid, _ = validate_pubmed_query(
                    query,
                    original_plan.intent.value,
                    original_plan.drugs_identified,
                    original_plan.normalized_question,
                )
                if is_valid:
                    refined_queries.append(
                        PlannedQuery(
                            pubmed_query=query,
                            sub_question_index=sub_question_index,
                            strategy_label=str((entry or {}).get("strategy_label") or "targeted_refine"),
                        )
                    )
            return refined_queries
        except Exception:
            return []

    def _fallback_keyword_query(
        self,
        question: str,
        drugs: list[str],
        intent: str = "",
        substances: list[str] | None = None,
    ) -> list[PlannedQuery]:
        drug_terms = [drug.strip() for drug in drugs if drug.strip()]
        specific_keywords = self._extract_keywords(question, drop_generic=True)

        if drug_terms:
            drug_clause = " OR ".join(f"{drug}[tiab]" for drug in drug_terms)
            drugs_part = f"({drug_clause})"
        else:
            lead_term = specific_keywords[0] if specific_keywords else "medication"
            drugs_part = f"({lead_term}[tiab])"

        intent_term = _INTENT_FALLBACK_TERMS.get(intent, "")
        if drug_terms and intent_term:
            # Drug is already in drugs_part; prefer the intent's canonical term
            # over a raw question token (which can be "side" from "side effects").
            keyword_clause = intent_term
        elif intent_term:
            keyword_term = specific_keywords[0] if specific_keywords else "safety"
            keyword_clause = f"{keyword_term}[tiab] OR {intent_term}"
        else:
            keyword_term = specific_keywords[0] if specific_keywords else "safety"
            keyword_clause = f"{keyword_term}[tiab]"

        all_substances = list(substances or [])
        if "alcohol" in question.lower() or any("alcohol" in s.lower() for s in all_substances):
            keyword_clause += " OR alcohol[tiab]"

        query = f"{drugs_part} AND ({keyword_clause}) AND English[Language] AND Humans[Mesh]"

        is_valid, _ = validate_pubmed_query(query, intent, drug_terms, question)
        if not is_valid:
            extra: list[str] = []
            if intent == "drug_interaction" and "interaction" not in query.lower():
                extra.append("interaction[tiab]")
            if "alcohol" in question.lower() and not any(
                t in query.lower() for t in ["alcohol", "ethanol"]
            ):
                extra.append("alcohol[tiab]")
            if extra:
                keyword_clause = keyword_clause + " OR " + " OR ".join(extra)
                query = f"{drugs_part} AND ({keyword_clause}) AND English[Language] AND Humans[Mesh]"

        return [PlannedQuery(pubmed_query=query, sub_question_index=0, strategy_label="fallback")]

    @staticmethod
    def _strip_markdown_fences(payload_text: str) -> str:
        text = payload_text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        object_start = text.find("{")
        array_start = text.find("[")
        starts = [start for start in (object_start, array_start) if start != -1]
        if starts:
            start = min(starts)
            if start == object_start:
                end = text.rfind("}")
                if end != -1 and end > start:
                    return text[start : end + 1]
            if start == array_start:
                end = text.rfind("]")
                if end != -1 and end > start:
                    return text[start : end + 1]
        return text

    @staticmethod
    def _extract_keywords(question: str, drop_generic: bool = False) -> list[str]:
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]*\b", question.lower())
        filtered = [w for w in words if w not in _STOPWORDS]
        if drop_generic:
            filtered = [w for w in filtered if w not in _GENERIC_MEDICAL_TOKENS]
        return filtered

    @staticmethod
    def _coerce_intent(intent: str) -> QueryIntent:
        try:
            return QueryIntent(intent)
        except ValueError:
            return QueryIntent.unknown
