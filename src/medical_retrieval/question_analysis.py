from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


class QuestionAnalysisError(RuntimeError):
    """Raised when medication question analysis cannot be produced from LLM output."""


class MedicationMention(BaseModel):
    raw_name: str
    role: str | None = None

    @field_validator("raw_name")
    @classmethod
    def raw_name_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("raw_name must not be empty")
        return value


class NamedMedicationEntity(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be empty")
        return value


class NormalizedIngredient(BaseModel):
    raw_name: str
    canonical_name: str

    @field_validator("raw_name", "canonical_name")
    @classmethod
    def required_names_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ingredient names must not be empty")
        return value


class RxNormCandidate(BaseModel):
    raw_name: str
    canonical_name: str | None = None
    generic_name: str | None = None
    match_type: str | None = None
    rxnorm_cui: str | None = None
    confidence: float | None = None

    @field_validator("raw_name")
    @classmethod
    def raw_name_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("raw_name must not be empty")
        return value


class MedicationQuestionEntities(BaseModel):
    drugs: list[MedicationMention]
    supplements: list[MedicationMention]
    normalized_ingredients: list[NormalizedIngredient]
    drug_classes: list[NamedMedicationEntity]
    conditions: list[NamedMedicationEntity]
    populations: list[NamedMedicationEntity]
    outcomes: list[NamedMedicationEntity]
    mechanisms: list[NamedMedicationEntity]


class MedicationQuestionAnalysis(BaseModel):
    question_type: str
    entities: MedicationQuestionEntities
    rxnorm_candidates: list[RxNormCandidate] = Field(default_factory=list)
    retrieval_constraints: list[str]
    rationale: str | None = None
    reasoning: str | None = None

    @field_validator("question_type")
    @classmethod
    def question_type_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question_type must not be empty")
        return value

    @field_validator("retrieval_constraints", mode="before")
    @classmethod
    def normalize_retrieval_constraints(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("retrieval_constraints must be a list of strings")
        constraints: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("retrieval_constraints must be a list of strings")
            stripped = item.strip()
            if stripped:
                constraints.append(stripped)
        return constraints


QuestionAnalysis = MedicationQuestionAnalysis


class QuestionAnalyzer:
    """LLM-backed analyzer for medication and supplement retrieval questions."""

    def __init__(
        self,
        *,
        llm_client: Any,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def analyze(
        self,
        question: str,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> MedicationQuestionAnalysis:
        try:
            response = self.llm_client.generate(
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": self._user_prompt(question, conversation_context or [])},
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=_question_analysis_response_format(),
            )
        except Exception as exc:
            if _is_expected_llm_error(exc):
                raise QuestionAnalysisError("Question analysis LLM request failed") from exc
            raise

        try:
            payload = _normalize_known_output_variants(
                json.loads(_strip_markdown_fences(str(response)))
            )
            return MedicationQuestionAnalysis.model_validate(payload)
        except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as exc:
            raise QuestionAnalysisError("Question analysis LLM response was invalid") from exc

    def _system_prompt(self) -> str:
        return (
            "Analyze medication and supplement questions for evidence retrieval. Return only JSON with "
            "question_type, entities, optional rxnorm_candidates, retrieval_constraints, and optional "
            "rationale or reasoning. The entities object must contain list fields named drugs, supplements, "
            "normalized_ingredients, drug_classes, conditions, populations, outcomes, and mechanisms. "
            "Use object entries so Python only validates schema and preserves the LLM's domain analysis. "
            "Do not add deterministic Python policy assumptions; downstream Perplexity Agent handles "
            "strategy discovery and requery. Include RxNorm candidates only for medication or supplement "
            "mentions when available. Do not include markdown outside the JSON."
        )

    def _user_prompt(self, question: str, conversation_context: list[dict[str, str]]) -> str:
        return json.dumps(
            {
                "question": question,
                "conversation_context": conversation_context,
                "instructions": (
                    "Extract medication-centric entities exactly from the question and context: drugs, "
                    "supplements, normalized ingredients, drug classes, conditions, populations, outcomes, "
                    "mechanisms, optional RxNorm candidates, and retrieval constraints. Leave downstream "
                    "search strategy planning and requery to the Perplexity Agent."
                ),
            }
        )


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def _normalize_known_output_variants(payload: object) -> object:
    """Normalize observed structured-output aliases, then let Pydantic enforce schema."""
    if not isinstance(payload, dict):
        return payload

    normalized = deepcopy(payload)
    entities = normalized.get("entities")
    if isinstance(entities, dict):
        for bucket in ("drugs", "supplements"):
            _normalize_raw_name_entries(entities.get(bucket))
        for bucket in ("drug_classes", "conditions", "populations", "outcomes", "mechanisms"):
            _normalize_name_entries(entities.get(bucket))
        _normalize_normalized_ingredient_entries(entities.get("normalized_ingredients"))

    _normalize_rxnorm_candidates(normalized.get("rxnorm_candidates"))

    constraints = normalized.get("retrieval_constraints")
    if isinstance(constraints, dict):
        normalized["retrieval_constraints"] = _flatten_retrieval_constraints(constraints)

    return normalized


def _normalize_raw_name_entries(entries: object) -> None:
    if not isinstance(entries, list):
        return
    for idx, entry in enumerate(entries):
        if isinstance(entry, str):
            entries[idx] = {"raw_name": entry}
            continue
        if not isinstance(entry, dict):
            continue
        name = _non_empty_str(entry.get("name"))
        if name and not _non_empty_str(entry.get("raw_name")):
            entry["raw_name"] = name


def _normalize_name_entries(entries: object) -> None:
    if not isinstance(entries, list):
        return
    for idx, entry in enumerate(entries):
        if isinstance(entry, str):
            entries[idx] = {"name": entry}


def _normalize_normalized_ingredient_entries(entries: object) -> None:
    if not isinstance(entries, list):
        return
    for idx, entry in enumerate(entries):
        if isinstance(entry, str):
            entries[idx] = {"raw_name": entry, "canonical_name": entry}
            continue
        if not isinstance(entry, dict):
            continue
        name = _non_empty_str(entry.get("name"))
        if name and not _non_empty_str(entry.get("raw_name")):
            entry["raw_name"] = name
        if name and not _non_empty_str(entry.get("canonical_name")):
            entry["canonical_name"] = name
        generic_name = _non_empty_str(entry.get("generic_name"))
        if generic_name and not _non_empty_str(entry.get("canonical_name")):
            entry["canonical_name"] = generic_name


def _normalize_rxnorm_candidates(entries: object) -> None:
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = _non_empty_str(entry.get("name")) or _non_empty_str(entry.get("mention"))
        if name and not _non_empty_str(entry.get("raw_name")):
            entry["raw_name"] = name

        nested_candidate = _first_mapping(entry.get("candidates"))
        if nested_candidate is not None:
            candidate_name = _non_empty_str(nested_candidate.get("name"))
            if candidate_name and not _non_empty_str(entry.get("canonical_name")):
                entry["canonical_name"] = candidate_name
            if candidate_name and not _non_empty_str(entry.get("generic_name")):
                entry["generic_name"] = candidate_name

        rxnorm = (
            _non_empty_str(entry.get("rxnorm"))
            or _non_empty_str(entry.get("rxnorm_id"))
            or (
                _non_empty_str(nested_candidate.get("rxnorm_id"))
                if nested_candidate is not None
                else None
            )
            or (
                _non_empty_str(nested_candidate.get("rxnorm_cui"))
                if nested_candidate is not None
                else None
            )
        )
        if rxnorm and not _non_empty_str(entry.get("rxnorm_cui")):
            entry["rxnorm_cui"] = rxnorm


def _first_mapping(value: object) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    for item in value:
        if isinstance(item, dict):
            return item
    return None


def _flatten_retrieval_constraints(constraints: dict[str, Any]) -> list[str]:
    ordered_keys = [
        "must_include",
        "exclude",
        "time_sensitivity",
        "evidence_preferences",
    ]
    ordered_keys.extend(sorted(key for key in constraints if key not in ordered_keys))

    flattened: list[str] = []
    for key in ordered_keys:
        if key not in constraints:
            continue
        value = constraints[key]
        if isinstance(value, list):
            for item in value:
                rendered = _render_constraint_value(item)
                if rendered:
                    flattened.append(f"{key}: {rendered}")
            continue
        rendered = _render_constraint_value(value)
        if rendered:
            flattened.append(f"{key}: {rendered}")
    return flattened


def _render_constraint_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        if not value:
            return None
        return json.dumps(value, sort_keys=True)
    return str(value)


def _non_empty_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _question_analysis_response_format() -> dict[str, Any]:
    named_entity = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    medication_mention = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "raw_name": {"type": "string"},
            "role": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["raw_name", "role"],
    }
    normalized_ingredient = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "raw_name": {"type": "string"},
            "canonical_name": {"type": "string"},
        },
        "required": ["raw_name", "canonical_name"],
    }
    rxnorm_candidate = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "raw_name": {"type": "string"},
            "canonical_name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "generic_name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "match_type": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "rxnorm_cui": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "confidence": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        },
        "required": [
            "raw_name",
            "canonical_name",
            "generic_name",
            "match_type",
            "rxnorm_cui",
            "confidence",
        ],
    }
    entities = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "drugs": {"type": "array", "items": medication_mention},
            "supplements": {"type": "array", "items": medication_mention},
            "normalized_ingredients": {"type": "array", "items": normalized_ingredient},
            "drug_classes": {"type": "array", "items": named_entity},
            "conditions": {"type": "array", "items": named_entity},
            "populations": {"type": "array", "items": named_entity},
            "outcomes": {"type": "array", "items": named_entity},
            "mechanisms": {"type": "array", "items": named_entity},
        },
        "required": [
            "drugs",
            "supplements",
            "normalized_ingredients",
            "drug_classes",
            "conditions",
            "populations",
            "outcomes",
            "mechanisms",
        ],
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_type": {"type": "string"},
            "entities": entities,
            "rxnorm_candidates": {"type": "array", "items": rxnorm_candidate},
            "retrieval_constraints": {"type": "array", "items": {"type": "string"}},
            "rationale": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "reasoning": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": [
            "question_type",
            "entities",
            "rxnorm_candidates",
            "retrieval_constraints",
            "rationale",
            "reasoning",
        ],
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "medication_question_analysis",
            "strict": True,
            "schema": schema,
        },
    }


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
