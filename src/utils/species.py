"""Species utilities for consistent species matching across modules."""
import re
from typing import Dict, Iterable, List, Optional, Set


def tokenize_species_string(text: str) -> Set[str]:
    """Normalize species strings by tokenizing on non-alphanumeric characters."""
    if not text:
        return set()
    return set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))


def infer_species_from_text(
    paper_or_metadata: Dict[str, any],
    mesh_terms: Optional[Iterable[str]] = None,
    *,
    strict_species_inference: bool = True,
) -> List[str]:
    """
    Infer species from text using consistent logic across modules.

    Args:
        paper_or_metadata: Dictionary containing paper data or metadata
        mesh_terms: Optional MeSH terms to consider
        strict_species_inference: Whether to use strict inference rules for humans

    Returns:
        List of inferred species
    """
    # Species keywords and mapping
    _SPECIES_KEYWORDS = {
        "human": {"human", "humans"},  # Removed generic "patient", "patients"
        "mouse": {"mouse", "mice"},
        "rat": {"rat", "rats"},
        "dog": {"dog", "dogs", "canine", "canines"},
        "monkey": {"monkey", "monkeys", "nonhuman primate", "macaque"},
    }

    _CLINICAL_STUDY_TAGS = {
        "clinical trial",
        "randomized controlled trial",
        "controlled clinical trial",
        "phase i",
        "phase ii",
        "phase iii",
        "phase iv",
        "rct",
        "clinical study",
        "human study",
        "patient study",
        "clinical investigation",
    }

    _NEGATION_TERMS = {
        "in vitro",
        "cell culture",
        "cultured cells",
        "tissue culture",
        "cell line",
        "isolated cells",
        "artificial",
    }

    combined_sources = [
        paper_or_metadata.get("title"),
        paper_or_metadata.get("abstract"),
        paper_or_metadata.get("summary"),
    ]
    if mesh_terms is None:
        mesh_terms = paper_or_metadata.get("mesh_terms") or []
    combined_sources.extend(mesh_terms)
    combined_text = " ".join(str(value) for value in combined_sources if value)

    # Tokenize text using non-alphanumeric delimiters for stricter matching
    tokens = tokenize_species_string(combined_text)

    # Check for negation terms that indicate non-human contexts
    has_negation = any(term in combined_text.lower() for term in _NEGATION_TERMS)

    matches: List[str] = []
    for species, keywords in _SPECIES_KEYWORDS.items():
        if species == "human" and strict_species_inference:
            # For humans, require either MeSH "Humans" or clinical study context
            mesh_tokens = tokenize_species_string(" ".join(str(term) for term in mesh_terms))
            study_types = paper_or_metadata.get("study_types", []) or paper_or_metadata.get("tags", [])
            study_context = " ".join(str(tag).lower() for tag in study_types)

            # Check MeSH terms for "Humans"
            if "humans" in mesh_tokens:
                if not has_negation:
                    matches.append(species)
            # Check for clinical study tags and human-related keywords together
            elif any(clinical_tag in study_context for clinical_tag in _CLINICAL_STUDY_TAGS):
                if any(keyword in tokens for keyword in keywords) and not has_negation:
                    matches.append(species)
            # Allow "patient"/"patients" only with clinical context
            elif any(clinical_tag in study_context for clinical_tag in _CLINICAL_STUDY_TAGS):
                patient_tokens = {"patient", "patients"}
                if patient_tokens.intersection(tokens) and not has_negation:
                    matches.append(species)
        else:
            # For non-human species, use standard matching
            if any(keyword in tokens for keyword in keywords):
                if not (has_negation and species == "human"):  # Block only human when in vitro
                    if species not in matches:
                        matches.append(species)

    return matches
