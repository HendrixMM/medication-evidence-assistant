"""
LLM-guided PubMed query generation with deterministic context.

This module uses GPT-4o-mini to generate optimized PubMed queries by combining:
1. Semantic understanding from the LLM
2. Medical domain knowledge from deterministic mappings (MeSH, drug synonyms)
3. Structured prompt engineering with examples

The LLM makes the final query generation decision while being guided by
deterministic context that provides medical domain expertise.
"""
# DEPRECATED: subsumed by src/agent/planner.py
# This module is superseded by the agent layer introduced in Phase 4.
# It is retained for backward compatibility with Streamlit apps and existing tests.
# Do not add new functionality here. See src/agent/ for the active implementation.
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from src.nvidia_llm_client import NVIDIALLMClient
from src.patient_entity_extractor import _BRAND_TO_GENERIC
from src.patient_entity_extractor import PatientQueryEntities
from src.patient_query_constants import ADDITIONAL_DRUG_SYNONYMS
from src.patient_query_constants import DEFAULT_SUBHEADINGS
from src.patient_query_constants import MESH_SUBHEADING_MAP

logger = logging.getLogger(__name__)

DRUG_SYNONYMS = {**_BRAND_TO_GENERIC, **ADDITIONAL_DRUG_SYNONYMS}
ALCOHOL_TERMS = [
    "Alcohol Drinking[Mesh]",
    "alcohol[tiab]",
    "ethanol[tiab]",
]


@dataclass
class QueryGenerationResult:
    """Result from LLM query generation with metadata."""

    query: str
    reasoning: str
    generation_time_ms: float
    tokens_used: int | None
    fallback_used: bool = False
    validation_passed: bool = True
    error_message: str | None = None

    def __repr__(self) -> str:
        """Readable representation for debugging."""
        status = "✅ VALID" if self.validation_passed else "❌ INVALID"
        source = "🔄 FALLBACK" if self.fallback_used else "🤖 LLM"
        tokens = f"{self.tokens_used} tokens" if self.tokens_used else "no token data"
        return f"QueryGenerationResult({source} {status}, " f"{self.generation_time_ms:.1f}ms, {tokens})"


@dataclass
class QueryContext:
    """Deterministic context to guide LLM query generation."""

    mesh_subheadings: list[str] = field(default_factory=list)
    drug_synonyms: dict[str, list[str]] = field(default_factory=dict)
    intent_keywords: list[str] = field(default_factory=list)
    generic_class_expansions: dict[str, list[str]] = field(default_factory=dict)
    substance_terms: list[str] = field(default_factory=list)
    alcohol_terms: list[str] = field(default_factory=list)


def validate_pubmed_query(
    query: str,
    intent: str,
    substances: list[str],
    user_query: str,
) -> tuple[bool, str | None]:
    """Validate generated query for common issues."""
    if not query or not query.strip():
        return False, "Query is empty"

    if query.count("(") != query.count(")"):
        return False, "Unbalanced parentheses"

    if "English[Language]" not in query:
        return False, "Missing English[Language] filter"

    if "[" not in query or "]" not in query:
        return False, "Missing PubMed field tags ([mesh], [tiab], etc.)"

    if not any(tag in query.lower() for tag in ["[mesh]", "[tiab]", "[majr]"]):
        return False, "No valid search terms found"

    query_lower = query.lower()
    normalized_intent = (intent or "").lower()
    if normalized_intent == "drug_interaction":
        if not any(term in query_lower for term in ["drug interactions", "contraindication", "interaction"]):
            return False, "Missing interaction-focused terms"

    alcohol_in_query = "alcohol" in user_query.lower()
    alcohol_in_entities = any("alcohol" in s.lower() for s in substances)
    if alcohol_in_query or alcohol_in_entities:
        if not any(term in query_lower for term in ["alcohol", "ethanol", "alcohol drinking"]):
            return False, "Missing alcohol-specific terms"

    return True, None


class LLMQueryGenerator:
    """Generate PubMed queries using LLM with deterministic guidance."""

    # Intent-specific keywords for context
    INTENT_KEYWORDS: dict[str, list[str]] = {
        "side_effects": ["adverse effects", "side effects", "toxicity", "safety"],
        "dosage": ["dosage", "dose", "administration", "therapeutic use"],
        "drug_interaction": [
            "drug interactions",
            "contraindications",
            "drug incompatibility",
            "coadministration",
            "concomitant",
            "safety",
        ],
        "mechanism": ["mechanism of action", "pharmacology", "pharmacokinetics", "pharmacodynamics"],
        "timing": ["pharmacokinetics", "time factors", "onset of action", "duration"],
        "comparison": ["comparative effectiveness", "treatment outcome", "therapeutic equivalency"],
        "usage": ["therapeutic use", "administration", "indications"],
        "pregnancy_safety": ["pregnancy", "teratogenicity", "lactation", "breast feeding"],
        "lifestyle": ["drug interactions", "diet", "alcohol drinking", "smoking"],
    }

    # Generic drug class expansions for improved specificity
    GENERIC_CLASS_EXPANSIONS: dict[str, list[str]] = {
        "blood thinners": ["warfarin", "apixaban", "rivaroxaban", "dabigatran", "enoxaparin"],
        "anticoagulants": ["warfarin", "apixaban", "rivaroxaban", "dabigatran", "enoxaparin"],
        "antibiotics": [
            "amoxicillin",
            "azithromycin",
            "ciprofloxacin",
            "doxycycline",
            "penicillin",
            "rifampin",
            "rifabutin",
        ],
        "birth control": [
            "ethinyl estradiol",
            "ethinylestradiol",
            "levonorgestrel",
            "norethindrone",
            "oral contraceptives",
        ],
        "antidepressants": ["sertraline", "fluoxetine", "escitalopram", "venlafaxine", "bupropion"],
        "statins": ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin"],
        "nsaids": ["ibuprofen", "naproxen", "diclofenac", "celecoxib"],
        "beta blockers": ["metoprolol", "atenolol", "carvedilol", "propranolol"],
        "ace inhibitors": ["lisinopril", "enalapril", "ramipril", "benazepril"],
        "blood pressure medication": [
            "lisinopril",
            "amlodipine",
            "losartan",
            "metoprolol",
            "hydrochlorothiazide",
        ],
    }

    def __init__(
        self,
        llm_client: NVIDIALLMClient,
        temperature: float = 0.2,
        max_tokens: int = 350,
        timeout_seconds: int = 10,
        enable_validation: bool = True,
    ) -> None:
        """Initialize LLM query generator.

        Args:
            llm_client: NVIDIA LLM client for API calls
            temperature: LLM temperature (0.0-1.0, lower = more deterministic)
            max_tokens: Maximum tokens in LLM response
            timeout_seconds: API call timeout
            enable_validation: Whether to validate generated queries
        """
        self.llm = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.enable_validation = enable_validation

    def _build_context(self, user_query: str, entities: PatientQueryEntities) -> QueryContext:
        """Build deterministic context from existing mappings.

        This provides the medical domain knowledge that guides the LLM's
        query generation decisions.
        """
        context = QueryContext()

        # Get MeSH subheadings for intent
        intent = entities.intent or "general"
        context.mesh_subheadings = MESH_SUBHEADING_MAP.get(intent, DEFAULT_SUBHEADINGS)

        # Get drug synonyms
        for drug in entities.drugs:
            drug_lower = drug.lower()
            if drug_lower in DRUG_SYNONYMS:
                context.drug_synonyms[drug] = DRUG_SYNONYMS[drug_lower]
            else:
                context.drug_synonyms[drug] = [drug]  # Just the drug itself

        # Get intent keywords
        context.intent_keywords = self.INTENT_KEYWORDS.get(intent, [])

        # Detect and expand generic drug classes
        query_lower = user_query.lower()
        for generic_class, specific_drugs in self.GENERIC_CLASS_EXPANSIONS.items():
            if generic_class in query_lower:
                context.generic_class_expansions[generic_class] = specific_drugs

        context.substance_terms = list(entities.substances or [])
        if "alcohol" in query_lower or "alcohol" in [s.lower() for s in context.substance_terms]:
            context.alcohol_terms = list(ALCOHOL_TERMS)

        return context

    def _build_prompt(
        self,
        user_query: str,
        entities: PatientQueryEntities,
        context: QueryContext,
    ) -> str:
        """Build the prompt for LLM query generation with examples."""

        # Format entities
        entities_dict = {
            "drugs": entities.drugs,
            "intent": entities.intent,
            "substances": entities.substances,
            "dosage": entities.dosage,
        }

        # Format context
        context_dict = {
            "mesh_subheadings": context.mesh_subheadings,
            "drug_synonyms": context.drug_synonyms,
            "intent_keywords": context.intent_keywords,
            "generic_class_expansions": context.generic_class_expansions,
            "substance_terms": context.substance_terms,
            "alcohol_terms": context.alcohol_terms,
        }

        prompt = f"""You are a medical literature search expert. Your job is to generate PubMed queries that retrieve 15-40 highly relevant articles (avg relevance ≥4.5/5.0).

USER QUERY: {user_query}

CRITICAL SUCCESS RULES (Source Quality = Everything):
1. ALWAYS use MULTIPLE strategies: (MeSH strategy) OR (TIAB strategy) OR (Broad fallback)
   - Single strategy = failure. Must combine at least 3 approaches.

2. ALWAYS include ALL drug name variations:
   - Generic name (e.g., metformin)
   - Common brand names (e.g., glucophage, fortamet, glumetza)
   - Chemical names if well-known (e.g., acetylsalicylic acid for aspirin)
   - Search PubMed for brand names if unsure - include top 2-3

3. ALWAYS expand generic drug classes to specific medications:
   - "blood thinners" → warfarin, apixaban, rivaroxaban, dabigatran, enoxaparin
   - "birth control" → ethinyl estradiol, levonorgestrel, norethindrone, oral contraceptives
   - "antibiotics" → amoxicillin, azithromycin, ciprofloxacin, doxycycline
   - "blood pressure meds" → lisinopril, amlodipine, losartan, metoprolol
   - "antidepressants" → sertraline, fluoxetine, escitalopram, venlafaxine
   - Search medical databases if you encounter an unfamiliar generic term

4. ALWAYS use multiple MeSH subheadings (not just one):
   - Side effects → /adverse effects, /toxicity, /contraindications
   - Dosage → /administration & dosage, /therapeutic use
   - Drug interactions → /adverse effects, /contraindications, /pharmacology
   - Timing → /pharmacokinetics, /administration & dosage

5. ALWAYS include intent-specific keywords based on question type:
   - Side effects: (adverse effects[tiab] OR side effects[tiab] OR toxicity[tiab] OR safety[tiab] OR tolerability[tiab])
   - Drug interactions: ("drug interactions"[mesh] OR contraindications[tiab] OR coadministration[tiab] OR concomitant[tiab])
   - Dosage: (dosage[tiab] OR dose[tiab] OR "recommended dose"[tiab] OR administration[tiab])
   - Timing: ("onset of action"[tiab] OR "time to effect"[tiab] OR duration[tiab] OR "starts working"[tiab])

6. IF user mentions specific symptoms, include symptom-specific terms:
   - "diarrhea" → (diarrhea[tiab] OR gastrointestinal[tiab] OR "GI effects"[tiab] OR "digestive problems"[tiab])
   - "headache" → (headache[tiab] OR "head pain"[tiab] OR cephalgia[tiab] OR migraine[tiab])
   - "tired" → (fatigue[tiab] OR tiredness[tiab] OR "low energy"[tiab] OR lethargy[tiab])

7. ALWAYS end with: AND English[Language] AND Humans[Mesh]

QUALITY TARGETS:
- Query should retrieve 15-40 articles (not 2-5, not 100+)
- Average source relevance ≥4.5/5.0
- Queries should be 250-450 characters (comprehensive, not minimal)

QUERY STRUCTURE TEMPLATE:
(
  # Strategy 1: MeSH with multiple subheadings
  (drug[mesh:noexp]/subheading1 OR drug[mesh:noexp]/subheading2 OR drug[mesh:noexp]/subheading3)
  OR
  # Strategy 2: TIAB with all drug variations + intent keywords
  ((drug[tiab] OR brand1[tiab] OR brand2[tiab] OR brand3[tiab]) AND
   (intent_keyword1[tiab] OR intent_keyword2[tiab] OR intent_keyword3[tiab]))
  OR
  # Strategy 3: Symptom-specific if mentioned (optional but important)
  ((drug[tiab] OR brand1[tiab]) AND
   (symptom[tiab] OR related_symptom1[tiab] OR related_symptom2[tiab]))
)
AND English[Language] AND Humans[Mesh]

EXAMPLES:

Example 1 - Side Effects with Specific Symptom:
Query: "Does metformin cause diarrhea?"
Output: {{"query": "((metformin[mesh:noexp]/adverse effects OR metformin[mesh:noexp]/toxicity OR metformin[mesh:noexp]/contraindications) OR ((metformin[tiab] OR glucophage[tiab] OR fortamet[tiab] OR glumetza[tiab]) AND (adverse effects[tiab] OR side effects[tiab] OR toxicity[tiab] OR safety[tiab])) OR ((metformin[tiab] OR glucophage[tiab]) AND (diarrhea[tiab] OR gastrointestinal[tiab] OR \"GI effects\"[tiab] OR \"digestive problems\"[tiab]))) AND English[Language] AND Humans[Mesh]", "reasoning": "Multi-strategy query with MeSH (3 subheadings), TIAB with 4 brand names + safety keywords, and symptom-specific diarrhea/GI terms to maximize relevant article retrieval."}}

Example 2 - Drug Interaction with Generic Class:
Query: "Can I take aspirin with blood thinners?"
Output: {{"query": "(((aspirin[mesh] OR acetylsalicylic acid[mesh]) AND (warfarin[mesh] OR apixaban[mesh] OR rivaroxaban[mesh] OR dabigatran[mesh] OR enoxaparin[mesh])) OR ((aspirin[tiab] OR bayer[tiab] OR \"acetylsalicylic acid\"[tiab] OR ecotrin[tiab]) AND (warfarin[tiab] OR apixaban[tiab] OR rivaroxaban[tiab] OR anticoagulant[tiab] OR \"blood thinner\"[tiab]))) AND (\"drug interactions\"[mesh] OR contraindications[tiab] OR coadministration[tiab]) AND English[Language] AND Humans[Mesh]", "reasoning": "Expanded 'blood thinners' to 5 specific anticoagulants with both MeSH and TIAB. Included aspirin brand names (Bayer, Ecotrin) and interaction-focused terms for comprehensive coverage."}}

Example 3 - Dosage Query:
Query: "How much ibuprofen can I take?"
Output: {{"query": "((ibuprofen[mesh:noexp]/administration & dosage OR ibuprofen[mesh:noexp]/therapeutic use) OR ((ibuprofen[tiab] OR advil[tiab] OR motrin[tiab] OR nurofen[tiab]) AND (dosage[tiab] OR dose[tiab] OR \"recommended dose\"[tiab] OR administration[tiab] OR \"usual dose\"[tiab] OR \"maximum dose\"[tiab]))) AND English[Language] AND Humans[Mesh]", "reasoning": "MeSH subheadings for dosing info + TIAB with 4 brand names and 6 dosage-related keywords to capture clinical dosing guidance from multiple article types."}}

Example 4 - Timing Query:
Query: "How long does amoxicillin take to work?"
Output: {{"query": "((amoxicillin[mesh:noexp]/pharmacokinetics OR amoxicillin[mesh:noexp]/administration & dosage) OR ((amoxicillin[tiab] OR amoxil[tiab] OR trimox[tiab]) AND (\"onset of action\"[tiab] OR \"time to effect\"[tiab] OR \"starts working\"[tiab] OR duration[tiab] OR \"how long\"[tiab] OR \"time to response\"[tiab]))) AND English[Language] AND Humans[Mesh]", "reasoning": "Pharmacokinetic MeSH focus + TIAB with brand names and 6 timing-related phrases to find articles discussing amoxicillin onset and duration."}}

Now generate a comprehensive, multi-strategy query for: "{user_query}"

Think step-by-step:
1. What drugs are mentioned? List ALL brand names you know.
2. Any generic drug classes to expand? (blood thinners, antibiotics, etc.)
3. What's the intent? (side effects, dosage, interaction, timing)
4. Any specific symptoms mentioned?
5. Build query with 3 strategies: MeSH + TIAB + symptom-specific

Return ONLY valid JSON with "query" and "reasoning" fields:
"""
        return prompt.strip()

    def generate_query(
        self,
        user_query: str,
        entities: PatientQueryEntities,
        fallback_query: str | None = None,
    ) -> QueryGenerationResult:
        """Generate PubMed query using LLM with deterministic guidance.

        Args:
            user_query: Original patient question
            entities: Extracted entities (drugs, intent, symptoms)
            fallback_query: Deterministic query to use if LLM fails

        Returns:
            QueryGenerationResult with query string and metadata
        """
        start_time = time.time()

        try:
            # Build deterministic context
            context = self._build_context(user_query, entities)

            # Build prompt with examples
            prompt = self._build_prompt(user_query, entities, context)

            # Call LLM
            logger.info(f"Calling LLM for query generation: model={self.llm.model}")
            max_tokens = min(self.max_tokens, 350)
            response_text = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens,
                timeout=self.timeout_seconds,
            )

            # Parse response (handle potential markdown code blocks)
            payload_text = response_text.strip()
            if "```json" in payload_text:
                # Extract JSON from markdown code block
                payload_text = payload_text.split("```json")[1].split("```")[0].strip()
            elif "```" in payload_text:
                # Extract from generic code block
                payload_text = payload_text.split("```")[1].split("```")[0].strip()
            elif "{" in payload_text and "}" in payload_text:
                # Extract JSON object
                payload_text = payload_text[payload_text.find("{") : payload_text.rfind("}") + 1]

            payload = json.loads(payload_text)
            query = str(payload.get("query", "")).strip()
            reasoning = str(payload.get("reasoning", "No reasoning provided")).strip()

            if not query:
                raise ValueError("LLM response missing query field.")

            # Validate query
            validation_passed = True
            error_message = None
            if self.enable_validation:
                validation_passed, error_message = self._validate_query(query, entities, user_query)
                if not validation_passed:
                    logger.warning(f"LLM query validation failed: {error_message}")
                    if fallback_query:
                        logger.info("Using fallback deterministic query")
                        query = fallback_query
                        reasoning = f"Fallback used due to validation failure: {error_message}"
                        fallback_used = True
                        validation_passed = True  # Fallback is pre-validated
                    else:
                        fallback_used = False
                else:
                    fallback_used = False
            else:
                fallback_used = False

            generation_time_ms = (time.time() - start_time) * 1000
            tokens_used = getattr(self.llm, "last_usage_tokens", None)

            return QueryGenerationResult(
                query=query,
                reasoning=reasoning,
                generation_time_ms=generation_time_ms,
                tokens_used=tokens_used,
                fallback_used=fallback_used,
                validation_passed=validation_passed,
                error_message=error_message,
            )

        except Exception as e:
            logger.error(f"LLM query generation failed: {e}", exc_info=True)
            generation_time_ms = (time.time() - start_time) * 1000

            if fallback_query:
                logger.info("Using fallback deterministic query due to LLM error")
                tokens_used = getattr(self.llm, "last_usage_tokens", None)
                return QueryGenerationResult(
                    query=fallback_query,
                    reasoning=f"Fallback used due to LLM error: {str(e)}",
                    generation_time_ms=generation_time_ms,
                    tokens_used=tokens_used,
                    fallback_used=True,
                    validation_passed=True,
                    error_message=str(e),
                )
            else:
                raise RuntimeError(f"LLM query generation failed and no fallback provided: {e}")

    def _validate_query(
        self,
        query: str,
        entities: PatientQueryEntities,
        user_query: str,
    ) -> tuple[bool, str | None]:
        return validate_pubmed_query(query, entities.intent or "", list(entities.substances or []), user_query)

    def explain_generation(
        self,
        user_query: str,
        entities: PatientQueryEntities,
    ) -> dict[str, Any]:
        """Provide detailed explanation of query generation for debugging.

        Returns:
            Dictionary with context, reasoning, and metadata
        """
        context = self._build_context(user_query, entities)

        return {
            "user_query": user_query,
            "entities": {
                "drugs": entities.drugs,
                "intent": entities.intent,
                "substances": entities.substances,
                "dosage": entities.dosage,
            },
            "context": {
                "mesh_subheadings": context.mesh_subheadings,
                "drug_synonyms": context.drug_synonyms,
                "intent_keywords": context.intent_keywords,
                "generic_class_expansions": context.generic_class_expansions,
            },
            "deterministic_guidance": {
                "mesh_subheading_count": len(context.mesh_subheadings),
                "drug_synonym_count": sum(len(syns) for syns in context.drug_synonyms.values()),
                "intent_keyword_count": len(context.intent_keywords),
                "generic_class_expansion_count": sum(len(drugs) for drugs in context.generic_class_expansions.values()),
            },
        }


__all__ = ["LLMQueryGenerator", "QueryGenerationResult", "validate_pubmed_query"]
