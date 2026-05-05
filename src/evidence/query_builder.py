from __future__ import annotations

import json
import os
import re
from typing import Any

from .schemas import NormalizedDrug, QueryCandidate


class QueryBuilder:
    _BANNED_TERMS = {
        "pubmed",
        "pmid",
        "study",
        "studies",
        "article",
        "articles",
        "paper",
        "papers",
        "journal",
        "journals",
        "review",
        "reviews",
    }

    _INTENT_KEYWORDS = {
        "safety": "adverse effects toxicity contraindications overdose",
        "side_effects": "adverse effects side effects toxicity",
        "interactions": "drug interaction interaction safety",
        "benefits": "efficacy benefit clinical outcome",
        "comparison": "comparison versus efficacy safety",
        "dosage": "dose dosing safety",
    }
    _MAX_NON_DRUG_TOKENS = {
        "safety": 4,
        "side_effects": 4,
        "interactions": 4,
        "benefits": 5,
        "comparison": 5,
        "dosage": 4,
    }
    _ANCHOR_CONFIDENCE_THRESHOLD = 0.75
    _SPECIAL_POPULATION_PATTERNS = (
        r"\bpregnan(?:cy|t)\b",
        r"\bpediatric(?:s)?\b",
        r"\bchildren\b",
        r"\binfants?\b",
        r"\bneonat(?:e|al)\b",
        r"\blactation\b",
        r"\bbreast(?:feeding|fed)\b",
        r"\bgeriatric(?:s)?\b",
        r"\bspecial populations?\b",
    )

    def __init__(self, *, llm_client: Any, model: str | None = None) -> None:
        self.llm_client = llm_client
        self.model = model or os.getenv("LLM_MODEL", "gpt-5.4-mini")

    def build_queries(
        self,
        *,
        question: str,
        intent: str,
        normalized_drugs: list[NormalizedDrug],
        drug_contexts: list[dict[str, object]] | None = None,
    ) -> list[QueryCandidate]:
        normalized_contexts = self._normalize_drug_contexts(normalized_drugs=normalized_drugs, drug_contexts=drug_contexts)
        prompt = json.dumps(
            {
                "question": question,
                "intent": intent,
                "normalized_drugs": [item.model_dump() for item in normalized_drugs],
                "drug_contexts": normalized_contexts,
                "instructions": (
                    "Return JSON with a top-level 'queries' array. Each item needs 'query' and 'strategy_label'. "
                    "Emit 1 to 3 PubMed-ready queries. Use ingredient-level names when available. "
                    "Use canonical anchors when confidence is high. When confidence is low, include the raw mention "
                    "rather than forcing only the canonical ingredient. "
                    "For broad safety or benefit questions, emit one short broad query plus up to two focused queries. "
                    "Do not include words like PubMed, PMID, study, article, paper, journal, or review unless they are "
                    "part of a genuine biomedical concept. Safety questions should focus on direct safety outcomes "
                    "such as adverse effects, toxicity, overdose, contraindications, organ injury, and long-term safety. "
                    "Avoid broad class drift unless the question is explicitly about a drug class. "
                    "Prefer short PubMed queries with about 2 to 4 content terms after the drug anchor. "
                    "Do not introduce pregnancy, pediatrics, lactation, geriatrics, or other special populations "
                    "unless the question mentions them."
                ),
            }
        )
        try:
            raw = self.llm_client.generate_simple(prompt, model=self.model, temperature=0.2)
            payload = self._extract_json(raw)
            queries = []
            for item in payload.get("queries", []):
                if not isinstance(item, dict) or not item.get("query") or not item.get("strategy_label"):
                    continue
                query = self._sanitize_query(
                    str(item["query"]),
                    question=question,
                    intent=intent,
                    normalized_drugs=normalized_drugs,
                    drug_contexts=normalized_contexts,
                )
                if not query:
                    continue
                queries.append(QueryCandidate(query=query, strategy_label=str(item["strategy_label"]).strip()))
            if queries:
                return queries[:3]
        except Exception:
            pass
        return [
            self._fallback_query(
                question=question,
                intent=intent,
                normalized_drugs=normalized_drugs,
                drug_contexts=normalized_contexts,
            )
        ]

    def _fallback_query(
        self,
        *,
        question: str,
        intent: str,
        normalized_drugs: list[NormalizedDrug],
        drug_contexts: list[dict[str, object]],
    ) -> QueryCandidate:
        terms = self._preferred_anchor_terms(normalized_drugs=normalized_drugs, drug_contexts=drug_contexts) or [question]
        query = f"{' '.join(terms)} {self._INTENT_KEYWORDS.get(intent, intent)}".strip()
        query = self._sanitize_query(
            query,
            question=question,
            intent=intent,
            normalized_drugs=normalized_drugs,
            drug_contexts=drug_contexts,
        )
        return QueryCandidate(query=query, strategy_label="fallback")

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

    def _sanitize_query(
        self,
        query: str,
        *,
        question: str,
        intent: str,
        normalized_drugs: list[NormalizedDrug],
        drug_contexts: list[dict[str, object]],
    ) -> str:
        cleaned = " ".join(query.split())
        cleaned = re.sub(
            r"\b(" + "|".join(re.escape(term) for term in sorted(self._BANNED_TERMS, key=len, reverse=True)) + r")\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = self._apply_anchor_preferences(cleaned, drug_contexts=drug_contexts)
        if not self._allows_special_populations(question):
            for pattern in self._SPECIAL_POPULATION_PATTERNS:
                cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

        tokens: list[str] = []
        seen: set[str] = set()
        for token in cleaned.split():
            lower = token.lower()
            if not token or lower in seen:
                continue
            tokens.append(token)
            seen.add(lower)
        return self._compact_query(
            tokens,
            intent=intent,
            normalized_drugs=normalized_drugs,
            drug_contexts=drug_contexts,
        )

    def _compact_query(
        self,
        tokens: list[str],
        *,
        intent: str,
        normalized_drugs: list[NormalizedDrug],
        drug_contexts: list[dict[str, object]],
    ) -> str:
        max_non_drug_tokens = self._MAX_NON_DRUG_TOKENS.get(intent, 5)
        drug_token_set = {
            part.lower()
            for drug in normalized_drugs
            for name in (drug.raw_name, drug.generic_name)
            if name
            for part in name.split()
        }
        for context in drug_contexts:
            for name in [context.get("raw_name"), context.get("canonical_name"), *(context.get("aliases") or [])]:
                if not name:
                    continue
                for part in str(name).split():
                    drug_token_set.add(part.lower())
        compacted: list[str] = []
        non_drug_count = 0
        for token in tokens:
            lower = token.lower()
            if lower in drug_token_set:
                compacted.append(token)
                continue
            if non_drug_count >= max_non_drug_tokens:
                continue
            compacted.append(token)
            non_drug_count += 1
        return " ".join(compacted).strip()

    def _normalize_drug_contexts(
        self,
        *,
        normalized_drugs: list[NormalizedDrug],
        drug_contexts: list[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        if drug_contexts:
            return [dict(item) for item in drug_contexts]
        return [
            {
                "raw_name": drug.raw_name,
                "canonical_name": drug.generic_name,
                "confidence": 1.0 if drug.match_type == "exact" else 0.6 if drug.match_type == "approximate" else 0.0,
                "aliases": [name for name in (drug.raw_name, drug.generic_name) if name],
            }
            for drug in normalized_drugs
        ]

    def _preferred_anchor_terms(
        self,
        *,
        normalized_drugs: list[NormalizedDrug],
        drug_contexts: list[dict[str, object]],
    ) -> list[str]:
        anchors: list[str] = []
        for context in drug_contexts:
            confidence = float(context.get("confidence", 0.0) or 0.0)
            canonical = str(context.get("canonical_name") or "").strip()
            raw = str(context.get("raw_name") or "").strip()
            if confidence >= self._ANCHOR_CONFIDENCE_THRESHOLD and canonical:
                anchors.append(canonical)
                continue
            if raw:
                anchors.append(raw)
            if canonical and canonical.lower() != raw.lower():
                anchors.append(canonical)
        if anchors:
            return list(dict.fromkeys(item for item in anchors if item))
        return [item.generic_name or item.raw_name for item in normalized_drugs if item.generic_name or item.raw_name]

    def _apply_anchor_preferences(self, query: str, *, drug_contexts: list[dict[str, object]]) -> str:
        cleaned = query
        for context in drug_contexts:
            confidence = float(context.get("confidence", 0.0) or 0.0)
            canonical = str(context.get("canonical_name") or "").strip()
            raw = str(context.get("raw_name") or "").strip()
            aliases = [str(alias).strip() for alias in (context.get("aliases") or []) if str(alias).strip()]
            preferred = canonical if confidence >= self._ANCHOR_CONFIDENCE_THRESHOLD and canonical else raw or canonical
            if not preferred:
                continue
            for anchor in sorted({raw, canonical, *aliases} - {""}, key=len, reverse=True):
                if anchor.lower() == preferred.lower():
                    continue
                cleaned = re.sub(rf"\b{re.escape(anchor)}\b", preferred, cleaned, flags=re.IGNORECASE)
        return cleaned

    def _allows_special_populations(self, question: str) -> bool:
        lowered = question.lower()
        return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in self._SPECIAL_POPULATION_PATTERNS)
