"""
Patient-Focused Entity Extraction - MVP

Simplified extraction optimized for patient/non-professional queries.
Covers 80% of patient use cases with minimal complexity.

Design Principles:
- Fast: Regex-based for <5ms latency
- Focused: Top 50 drugs covering 80% of queries
- Simple: 5 intent categories instead of complex NLP
- Testable: Clear patterns for validation
"""
# DEPRECATED: subsumed by src/agent/planner.py
# This module is superseded by the agent layer introduced in Phase 4.
# It is retained for backward compatibility with Streamlit apps and existing tests.
# Do not add new functionality here. See src/agent/ for the active implementation.
import re
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


@dataclass
class PatientQueryEntities:
    """Extracted entities from patient query."""

    drugs: List[str]
    dosage: Optional[Dict[str, str]]  # {"amount": "20", "unit": "mg"}
    substances: List[str]  # ["alcohol", "grapefruit", etc.]
    intent: str  # "side_effects", "drug_interaction", etc.
    confidence: float  # 0.0-1.0
    matched_patterns: List[str]  # For debugging


# Priority 1: Common OTC & Prescription Drugs (Top 50)
# Covers 80% of patient queries based on common medication usage
_PATIENT_COMMON_DRUGS = {
    # Pain/Fever (OTC)
    "tylenol",
    "acetaminophen",
    "advil",
    "ibuprofen",
    "aspirin",
    "aleve",
    "naproxen",
    "motrin",
    "excedrin",
    # Antibiotics (Common prescriptions)
    "amoxicillin",
    "azithromycin",
    "doxycycline",
    "ciprofloxacin",
    "penicillin",
    "cephalexin",
    "levofloxacin",
    "antibiotic",
    "antibiotics",  # Both singular and plural
    # Chronic Conditions - Cardiovascular
    "atorvastatin",
    "lipitor",
    "simvastatin",
    "rosuvastatin",  # Statins
    "lisinopril",
    "amlodipine",
    "losartan",
    "metoprolol",  # BP meds
    "warfarin",
    "coumadin",
    "eliquis",
    "xarelto",  # Blood thinners
    # Chronic Conditions - Metabolic
    "metformin",
    "glucophage",  # Diabetes
    "levothyroxine",
    "synthroid",  # Thyroid
    # Gastrointestinal
    "omeprazole",
    "prilosec",
    "nexium",
    "esomeprazole",
    "pantoprazole",
    # Mental Health
    "sertraline",
    "zoloft",
    "escitalopram",
    "lexapro",
    "fluoxetine",
    "prozac",
    "antidepressant",
    "antidepressants",  # Generic terms
    # Other Common
    "gabapentin",
    "prednisone",
    "tramadol",
    "meloxicam",
    # Generic terms patients use
    "blood thinner",
    "blood thinners",
    "statin",
    "statins",
    "birth control",
    "inhaler",
    "insulin",
    "pain meds",
    "pain medication",
    "allergy medicine",
    "nsaid",
    "nsaids",
    "blood pressure medication",
    "blood pressure medications",
    "blood pressure medicine",
    "blood pressure meds",
    "bp medication",
    "bp meds",
}

# Brand-to-generic mapping (patients often use brand names)
_BRAND_TO_GENERIC = {
    "tylenol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "aleve": "naproxen",
    "lipitor": "atorvastatin",
    "coumadin": "warfarin",
    "prilosec": "omeprazole",
    "nexium": "esomeprazole",
    "synthroid": "levothyroxine",
    "glucophage": "metformin",
    "zoloft": "sertraline",
    "lexapro": "escitalopram",
    "prozac": "fluoxetine",
}


# Priority 2: Patient Query Intent Patterns
_PATIENT_INTENT_PATTERNS = {
    "side_effects": [
        re.compile(r"(?:side effects?|adverse effects?) (?:of |from )?(\w+)", re.IGNORECASE),
        re.compile(r"does (\w+) cause (\w+)", re.IGNORECASE),
        re.compile(r"(\w+) (?:make|makes) me (\w+)", re.IGNORECASE),
        re.compile(r"why (?:does|do) (\w+) (?:give|cause)", re.IGNORECASE),
        re.compile(r"common (?:side effects?|reactions?) (?:of )?(\w+)", re.IGNORECASE),
        re.compile(r"(\w+) (?:side effects?|reactions?)", re.IGNORECASE),
        # Simple patterns: "Advil stomach pain", "ibuprofen headache"
        re.compile(r"(\w+)\s+(?:pain|headache|nausea|diarrhea|stomach|dizziness)", re.IGNORECASE),
    ],
    "drug_interaction": [
        re.compile(r"can i (?:take|use|drink|have) (\w+) (?:with|and|on|while taking) (\w+)", re.IGNORECASE),
        re.compile(r"(?:taking|on) (\w+) (?:with|and) (\w+)", re.IGNORECASE),
        re.compile(r"(\w+) (?:and|with|\+) (\w+) together", re.IGNORECASE),
        re.compile(r"is it safe.*(?:to )?(?:take|drink|use).*(\w+)", re.IGNORECASE),
        re.compile(r"(?:is|are) ([\w\s]+) safe with ([\w\s]+)", re.IGNORECASE),
        re.compile(r"(\w+) interaction (?:with )?(\w+)?", re.IGNORECASE),
        re.compile(r"(\w+)\s+(\w+)\s+interaction", re.IGNORECASE),
        re.compile(r"mixing (\w+) (?:and|with) (\w+)", re.IGNORECASE),
        re.compile(r"does (?:[\w\s]+) work with (?:[\w\s]+)", re.IGNORECASE),  # Multi-word support
        # Simple patterns: "grapefruit and atorvastatin", "birth control and antibiotics"
        re.compile(r"([\w\s]+)\s+and\s+([\w\s]+)(?:\s|$)", re.IGNORECASE),
    ],
    "dosage": [
        re.compile(r"how (?:much|many) (\w+) (?:can|should)", re.IGNORECASE),
        re.compile(r"(\d+)\s*(?:mg|mcg|pills?|tablets?) (?:of )?(\w+)", re.IGNORECASE),
        re.compile(r"can i take (\d+) (\w+)", re.IGNORECASE),
        re.compile(r"(?:should i take|take) (\w+) with food", re.IGNORECASE),
        re.compile(r"is (\d+)\s*(?:mg|mcg) (?:of )?(\w+) a lot", re.IGNORECASE),
        re.compile(r"(?:maximum|max) (?:dose|dosage) (?:of )?(\w+)", re.IGNORECASE),
    ],
    "timing": [
        re.compile(r"how long (?:does|until|before) (\w+)", re.IGNORECASE),
        re.compile(
            r"when (?:does|will|should) ([\w\s]+) (?:work|start|kick in|become effective)", re.IGNORECASE
        ),  # Multi-word
        re.compile(r"how (?:long|many days) (?:should i|to|for) (?:take )?(\w+)", re.IGNORECASE),
        re.compile(r"can i stop (?:taking )?(\w+)", re.IGNORECASE),
        re.compile(r"when (?:to take|should i take) (\w+)", re.IGNORECASE),
        re.compile(r"(\w+) (?:morning|night|before bed)", re.IGNORECASE),
        re.compile(r"when will my (\w+) (?:work|start)", re.IGNORECASE),
    ],
    "comparison": [
        re.compile(r"(?:what'?s better|better).*(\w+) (?:or|vs) (\w+)", re.IGNORECASE),
        re.compile(r"(\w+) (?:instead of|versus) (\w+)", re.IGNORECASE),
        re.compile(r"(\w+)\s+vs\s+(\w+)", re.IGNORECASE),  # Simple "X vs Y"
        re.compile(r"(?:generic|brand) (?:version of )?(\w+)", re.IGNORECASE),
        re.compile(r"(?:alternative|substitute) (?:to|for) (\w+)", re.IGNORECASE),
        re.compile(r"(\w+) (?:compared to|vs\.?) (\w+)", re.IGNORECASE),
    ],
    "mechanism": [
        re.compile(r"how does ([\w\s]+) work", re.IGNORECASE),
        re.compile(r"how do ([\w\s]+) work", re.IGNORECASE),
        re.compile(r"mechanism of action", re.IGNORECASE),
        re.compile(r"works by", re.IGNORECASE),
        re.compile(r"(\w+)\s+mechanism", re.IGNORECASE),
        re.compile(r"mode of action", re.IGNORECASE),
    ],
}


# Priority 3: Simple Dosage Extraction
_DOSAGE_PATTERN = re.compile(
    r"(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|μg|ug|g|pills?|tablets?|capsules?)", re.IGNORECASE
)


# Priority 4: Common Substances (for interaction queries)
_SUBSTANCES = {
    "alcohol": ["alcohol", "beer", "wine", "liquor", "drinking", "drink"],
    "food": ["food", "meal", "eat", "eating"],
    "grapefruit": ["grapefruit", "grapefruit juice"],
    "caffeine": ["caffeine", "coffee", "energy drink", "tea"],
    "dairy": ["milk", "dairy", "cheese", "yogurt"],
    "vitamins": ["vitamin", "supplement", "multivitamin"],
}


class PatientEntityExtractor:
    """
    Simplified entity extractor for patient queries.

    Optimized for:
    - Speed: <5ms per query
    - Coverage: 80% of patient queries
    - Simplicity: Regex only, no ML
    """

    def __init__(self):
        """Initialize the patient entity extractor."""
        self.drug_names = _PATIENT_COMMON_DRUGS
        self.brand_to_generic = _BRAND_TO_GENERIC
        self.intent_patterns = _PATIENT_INTENT_PATTERNS
        self.dosage_pattern = _DOSAGE_PATTERN
        self.substances = _SUBSTANCES

    def extract(self, query: str) -> PatientQueryEntities:
        """
        Extract all entities from a patient query.

        Args:
            query: Patient's question (e.g., "Can I take Tylenol and Advil together?")

        Returns:
            PatientQueryEntities with all extracted information
        """
        query_lower = query.lower()
        matched_patterns = []

        # Extract drugs
        drugs = self._extract_drugs(query_lower)

        # Extract dosage
        dosage = self._extract_dosage(query)

        # Extract substances
        substances = self._extract_substances(query_lower)

        # Classify intent
        intent, intent_patterns = self._classify_intent(query_lower)
        matched_patterns.extend(intent_patterns)

        # Calculate confidence
        confidence = self._calculate_confidence(intent, drugs, substances, dosage)

        return PatientQueryEntities(
            drugs=drugs,
            dosage=dosage,
            substances=substances,
            intent=intent,
            confidence=confidence,
            matched_patterns=matched_patterns,
        )

    def _extract_drugs(self, query_lower: str) -> List[str]:
        """Extract drug names from query."""
        found_drugs = []

        # Simple word matching (fast)
        words = re.findall(r"\b\w+\b", query_lower)

        for word in words:
            # Check if word is a known drug
            if word in self.drug_names:
                # Normalize to generic if it's a brand name
                generic = self.brand_to_generic.get(word, word)
                if generic not in found_drugs:
                    found_drugs.append(generic)

        # Also check multi-word drugs (e.g., "birth control", "blood thinner")
        for drug in self.drug_names:
            if " " in drug and drug in query_lower:
                generic = self.brand_to_generic.get(drug, drug)
                if generic not in found_drugs:
                    found_drugs.append(generic)

        return found_drugs

    def _extract_dosage(self, query: str) -> Optional[Dict[str, str]]:
        """Extract dosage information."""
        match = self.dosage_pattern.search(query)
        if match:
            return {"amount": match.group("amount"), "unit": match.group("unit").lower()}
        return None

    def _extract_substances(self, query_lower: str) -> List[str]:
        """Extract common substances (alcohol, food, etc.)."""
        found_substances = []

        for substance_category, keywords in self.substances.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if substance_category not in found_substances:
                        found_substances.append(substance_category)
                    break

        return found_substances

    def _classify_intent(self, query_lower: str) -> Tuple[str, List[str]]:
        """
        Classify query intent.

        Returns:
            (intent_name, list_of_matched_patterns)

        Note: Check in priority order to avoid ambiguous matches.
        Comparison should be checked before side_effects to avoid
        "ibuprofen vs naproxen for pain" matching the pain pattern.
        """
        # Priority order (check specific intents first)
        priority_order = ["comparison", "drug_interaction", "dosage", "timing", "mechanism", "side_effects"]

        for intent in priority_order:
            if intent in self.intent_patterns:
                patterns = self.intent_patterns[intent]
                for pattern in patterns:
                    if pattern.search(query_lower):
                        return intent, [f"{intent}:{pattern.pattern[:50]}"]

        return "unknown", []

    def _calculate_confidence(
        self, intent: str, drugs: List[str], substances: List[str], dosage: Optional[Dict[str, str]]
    ) -> float:
        """
        Calculate confidence score (0.0-1.0).

        Logic:
        - Found clear intent? +0.4
        - Found at least 1 drug? +0.3
        - For interactions, found 2+ entities? +0.3
        - For dosage queries, found dosage info? +0.2
        """
        score = 0.0

        # Found clear intent?
        if intent != "unknown":
            score += 0.4

        # Found at least 1 drug?
        if len(drugs) > 0:
            score += 0.3

        # Intent-specific validation
        if intent == "drug_interaction":
            # Need 2+ entities (drug+drug, drug+substance, etc.)
            total_entities = len(drugs) + len(substances)
            if total_entities >= 2:
                score += 0.3
        elif intent == "dosage":
            # Bonus if we found dosage info
            if dosage:
                score += 0.2
            else:
                score += 0.15  # Still reasonable without exact dosage
        else:
            # Other intents just need to be recognized
            if intent != "unknown":
                score += 0.3

        return min(score, 1.0)  # Cap at 1.0


# Convenience function
def extract_patient_entities(query: str) -> PatientQueryEntities:
    """
    Extract entities from a patient query.

    Usage:
        entities = extract_patient_entities("Can I take Tylenol and Advil together?")
        print(entities.intent)  # "drug_interaction"
        print(entities.drugs)   # ["acetaminophen", "ibuprofen"]
        print(entities.confidence)  # 1.0
    """
    extractor = PatientEntityExtractor()
    return extractor.extract(query)
