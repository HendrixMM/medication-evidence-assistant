"""Study ranking and filtering utilities for pharmaceutical literature."""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Iterable, Literal

logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"(?<![a-z0-9-])[a-z0-9]+(?:-[a-z0-9]+)*(?![a-z0-9-])")

# Optional dependencies for enhanced MinHash performance
try:
    pass

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from datasketch import MinHash, MinHashLSH

    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False


# Helper function to parse environment variables as truthy values
def _env_true(env_var: str, default: bool = False) -> bool:
    """Parse environment variable as boolean."""
    value = os.getenv(env_var, str(default).lower()).lower()
    return value in ("true", "1", "t", "yes", "y")


# Configuration for MinHash diversity filtering
ENABLE_MINHASH_DIVERSITY = _env_true("ENABLE_MINHASH_DIVERSITY", True)
MINHASH_NUM_PERMUTATIONS = int(os.getenv("MINHASH_NUM_PERMUTATIONS", "128"))
MINHASH_MIN_INPUT_SIZE = int(os.getenv("MINHASH_MIN_INPUT_SIZE", "500"))
MINHASH_NUM_BANDS = int(os.getenv("MINHASH_NUM_BANDS", "16"))
MINHASH_LSH_THRESHOLD = float(os.getenv("MINHASH_LSH_THRESHOLD", "0.8"))
MINHASH_OPTIMIZE_MEMORY = _env_true("MINHASH_OPTIMIZE_MEMORY", True)
MINHASH_CACHE_SIGNATURES = _env_true("MINHASH_CACHE_SIGNATURES", False)

# Validate MinHash configuration with fallbacks
if MINHASH_NUM_PERMUTATIONS < 64 or MINHASH_NUM_PERMUTATIONS > 256:
    logger.warning("MINHASH_NUM_PERMUTATIONS must be between 64 and 256, using default 128")
    MINHASH_NUM_PERMUTATIONS = 128
if MINHASH_MIN_INPUT_SIZE < 100:
    logger.warning("MINHASH_MIN_INPUT_SIZE must be at least 100, using default 500")
    MINHASH_MIN_INPUT_SIZE = 500
if MINHASH_NUM_BANDS < 4 or MINHASH_NUM_BANDS > MINHASH_NUM_PERMUTATIONS:
    logger.warning("MINHASH_NUM_BANDS must be between 4 and NUM_PERMUTATIONS, using default 16")
    MINHASH_NUM_BANDS = 16
if not (0.0 < MINHASH_LSH_THRESHOLD < 1.0):
    logger.warning("MINHASH_LSH_THRESHOLD must be between 0 and 1, using default 0.8")
    MINHASH_LSH_THRESHOLD = 0.8


DEFAULT_WEIGHTS = {
    "quality": 0.4,
    "recency": 0.2,
    "sample_size": 0.15,
    "species": 0.1,
    "pharma_relevance": 0.15,
}

_HIGH_QUALITY_TAGS = {
    "systematic review": 0.95,
    "meta-analysis": 0.95,
    "meta analysis": 0.95,  # Common variant without hyphen
    "randomized controlled trial": 0.9,
    "rct": 0.9,  # Common abbreviation
    "clinical trial": 0.8,
    "phase iv clinical trial": 0.88,
    "phase 4 clinical trial": 0.88,
    "phase iii clinical trial": 0.88,
    "phase 3 clinical trial": 0.88,
    "phase ii clinical trial": 0.82,
    "phase 2 clinical trial": 0.82,
    "phase i clinical trial": 0.78,
    "phase 1 clinical trial": 0.78,
    "observational study": 0.7,
    "case-control studies": 0.7,
    "case control study": 0.7,
    "cohort studies": 0.7,
    "cohort study": 0.7,
}

_LOWER_QUALITY_TAGS = {
    "case reports": 0.4,
    "animal study": 0.4,
    "in vitro": 0.35,
    "letter": 0.3,
    "editorial": 0.25,
}

_SPECIES_PRIORITIES = {
    "human": 1.0,
    "humans": 1.0,
    "rat": 0.6,
    "rats": 0.6,
    "mouse": 0.55,
    "mice": 0.55,
    "dog": 0.5,
    "dogs": 0.5,
    "in vitro": 0.35,
}

_PHARMA_RELEVANCE_KEYWORDS = {
    "pharmacokinetics": 0.3,
    "pharmacodynamics": 0.25,
    "drug interaction": 0.35,
    "cytochrome": 0.2,
    "cyp": 0.2,
    "clearance": 0.15,
    "half-life": 0.2,
    "auc": 0.2,
    "exposure": 0.1,
}

_KNOWN_CYP_ENZYMES = {
    "cyp1a2",
    "cyp2b6",
    "cyp2c8",
    "cyp2c9",
    "cyp2c19",
    "cyp2d6",
    "cyp2e1",
    "cyp3a4",
    "cyp3a5",
    "cyp3a7",
}

_SAMPLE_SIZE_PATTERNS = [
    # Prefer explicit declarations such as "n=120"
    (
        "n_equals",
        re.compile(r"\bn\s*=\s*(?P<count>\d{1,3}(?:,\d{3})*|\d{1,5})\b", re.IGNORECASE),
    ),
    # Accept "patients = 85" or "participants: 60"
    (
        "term_assignment",
        re.compile(
            r"\b(?:patients|subjects|participants)\s*(?:[:=]\s*)?(?P<count>\d{1,3}(?:,\d{3})*|\d{1,5})\b",
            re.IGNORECASE,
        ),
    ),
    # Handle trailing descriptors such as "120 patients"
    (
        "term_trailing",
        re.compile(
            r"\b(?P<count>\d{1,3}(?:,\d{3})*|\d{1,5})\s+(?:patients|subjects|participants)\b",
            re.IGNORECASE,
        ),
    ),
]


@dataclass
class RankingBreakdown:
    quality: float
    recency: float
    sample_size: float
    species: float
    pharma_relevance: float

    def to_dict(self) -> dict[str, float]:
        return {
            "quality": self.quality,
            "recency": self.recency,
            "sample_size": self.sample_size,
            "species": self.species,
            "pharma_relevance": self.pharma_relevance,
        }


class StudyRankingFilter:
    """Ranks and filters PubMed study metadata for pharmaceutical relevance."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        recency_decay_years: float | None = None,
        diversity_threshold: float = 0.92,
        max_pairs: int | None = 5000,
    ) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self._normalise_weights()
        decay_value = float(recency_decay_years or 10.0)
        if decay_value <= 0:
            raise ValueError("recency_decay_years must be greater than zero")
        self.recency_decay_years = decay_value
        self.diversity_threshold = float(diversity_threshold)
        if not (0.0 < self.diversity_threshold < 1.0):
            raise ValueError("diversity_threshold must be between 0 and 1")
        if max_pairs is not None and max_pairs <= 0:
            max_pairs = None
        self.max_pairs = max_pairs

    def rank_studies(
        self,
        papers: list[dict[str, Any]],
        query: str = "",
        *,
        entities: list[str] | None = None,
        intent: str = "",
    ) -> list[dict[str, Any]]:
        """Attach ranking metadata to each paper and sort in-place."""
        if not papers:
            return papers

        ranked: list[dict[str, Any]] = []
        diversified = self.apply_diversity_filter(papers)
        for paper in diversified:
            tags = paper.get("study_types") or paper.get("tags") or []
            quality = self._calculate_study_quality_score(tags)
            recency = self._calculate_recency_score(paper.get("publication_year") or paper.get("year"))
            sample_size = self._calculate_sample_size_score(paper)
            species_source = " ".join(str(paper.get(key, "")) for key in ("species", "mesh_terms", "title", "abstract"))
            species = self._calculate_species_preference_score(species_source)
            pharma_relevance = self._calculate_pharmaceutical_relevance_score(
                paper, query, entities=entities or [], intent=intent
            )

            breakdown = RankingBreakdown(
                quality=quality,
                recency=recency,
                sample_size=sample_size,
                species=species,
                pharma_relevance=pharma_relevance,
            )
            score = self._weighted_score(breakdown)

            paper = dict(paper)  # shallow copy to avoid mutating input
            paper["ranking_score"] = round(score, 4)
            paper["ranking_breakdown"] = breakdown.to_dict()
            ranked.append(paper)

        ranked.sort(key=lambda item: item.get("ranking_score", 0.0), reverse=True)
        return ranked

    def apply_diversity_filter(
        self,
        papers: Iterable[dict[str, Any]],
        *,
        threshold: float | None = None,
        max_pairs: int | None = None,
        method: Literal["signature", "minhash"] = "signature",
    ) -> list[dict[str, Any]]:
        """Drop near duplicates based on title/abstract similarity with optimizations for large sets.

        Args:
            papers: Iterable of paper dictionaries
            threshold: Similarity threshold (default: from constructor)
            max_pairs: Maximum number of pairwise comparisons (adaptive by default)
            method: Detection method - "signature" (default) or "minhash" for large sets

        Returns:
            List of papers with near duplicates removed
        """
        threshold = self.diversity_threshold if threshold is None else float(threshold)
        papers_list = list(papers)
        input_size = len(papers_list)

        # Use minhash method for large sets to improve performance
        # Check if MinHash is enabled, datasketch is available, and input size meets minimum requirement
        use_minhash = (
            method == "minhash" and ENABLE_MINHASH_DIVERSITY and HAS_DATASKETCH and input_size >= MINHASH_MIN_INPUT_SIZE
        )

        if use_minhash:
            return self._apply_minhash_diversity(papers_list, threshold, max_pairs)

        # Adaptive max_pairs based on input size
        if max_pairs is None:
            if input_size <= 100:
                max_pairs = None  # No limit for small sets
            elif input_size <= 500:
                max_pairs = 5000
            elif input_size <= 1000:
                max_pairs = 10000
            else:
                max_pairs = 15000  # Cap for very large sets
        else:
            max_pairs = None if max_pairs <= 0 else int(max_pairs)

        filtered: list[dict[str, Any]] = []
        seen_texts: list[str] = []
        seen_signatures: list[set[str]] = []
        seen_hashes: list[str] = []  # Fast prefilter using hash
        comparisons = 0

        for paper in papers_list:
            full_text = " ".join(str(paper.get(key, "")) for key in ("title", "abstract", "summary")).strip()
            normalised = re.sub(r"\s+", " ", full_text.lower())
            if not normalised:
                filtered.append(paper)
                continue

            # Fast prefilter using md5 hash (deterministic across Python runs)
            text_hash = hashlib.md5(normalised.encode("utf-8")).hexdigest()
            if text_hash in seen_hashes:
                # Potential exact duplicate, skip detailed comparison
                continue

            is_duplicate = False
            signature = set(normalised.split()[:50])

            if max_pairs is not None and comparisons >= max_pairs:
                seen_texts.append(normalised)
                seen_signatures.append(signature)
                seen_hashes.append(text_hash)
                filtered.append(paper)
                continue

            # Fast signature-based prefilter
            for i, (seen, seen_sig) in enumerate(zip(seen_texts, seen_signatures)):
                if max_pairs is not None and comparisons >= max_pairs:
                    break

                # Quick signature overlap check (fast prefilter)
                if seen_sig and signature:
                    overlap_ratio = len(signature.intersection(seen_sig)) / len(signature.union(seen_sig))
                    if overlap_ratio < 0.3:  # Skip if very little overlap
                        continue

                if not self._lengths_similar(normalised, seen):
                    continue

                comparisons += 1
                similarity = SequenceMatcher(None, normalised, seen).ratio()
                if similarity >= threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            seen_texts.append(normalised)
            seen_signatures.append(signature)
            seen_hashes.append(text_hash)
            filtered.append(paper)

        return filtered

    def _apply_minhash_diversity(
        self,
        papers: list[dict[str, Any]],
        threshold: float,
        max_pairs: int | None = None,
    ) -> list[dict[str, Any]]:
        """Apply minhash-based diversity filtering for large document sets.

        Uses locality-sensitive hashing to cluster similar documents before
        detailed comparison, significantly reducing pairwise comparisons.
        Uses configurable parameters from environment for optimization.
        Optionally uses datasketch library if available for better performance.

        Args:
            papers: List of paper dictionaries
            threshold: Similarity threshold for considering duplicates
            max_pairs: Maximum number of pairwise comparisons

        Returns:
            List of papers with near duplicates removed
        """
        if not papers:
            return []

        # Use optimized datasketch implementation if available
        if HAS_DATASKETCH and ENABLE_MINHASH_DIVERSITY:
            try:
                return self._apply_minhash_datasketch(papers, threshold, max_pairs)
            except Exception as e:
                logger.debug("datasketch MinHash failed, falling back to pure Python: %s", e)

        # Minhash parameters from configuration
        num_permutations = MINHASH_NUM_PERMUTATIONS
        num_bands = MINHASH_NUM_BANDS
        rows_per_band = num_permutations // num_bands

        # Check for optional optimization features
        use_lsh_threshold = MINHASH_LSH_THRESHOLD < threshold

        # Generate minhash signatures for all documents
        minhash_signatures = []
        text_to_paper = []

        # Pre-compile regex for better performance
        whitespace_regex = re.compile(r"\s+")

        for paper in papers:
            full_text = " ".join(str(paper.get(key, "")) for key in ("title", "abstract", "summary")).strip()
            normalised = whitespace_regex.sub(" ", full_text.lower())

            if not normalised:
                minhash_signatures.append(None)
                text_to_paper.append(paper)
                continue

            # Create minhash signature with optimized hashing
            words = set(normalised.split())
            if not words:
                minhash_signatures.append(None)
                text_to_paper.append(paper)
                continue

            # Optimized hash-based minhash with early exit for memory optimization
            signature = []
            if MINHASH_OPTIMIZE_MEMORY and len(words) > 1000:
                # For large documents, use sampling to improve performance
                words = set(list(words)[:1000])

            for i in range(num_permutations):
                min_hash = float("inf")
                # Use a more efficient hash combination
                hash_input = f"{i}_".encode()
                for word in words:
                    combined = hash_input + word.encode()
                    hash_val = int(hashlib.md5(combined).hexdigest()[:8], 16)
                    min_hash = min(min_hash, hash_val)
                signature.append(min_hash)

            minhash_signatures.append(signature)
            text_to_paper.append(paper)

        # Banding for LSH with optimized memory usage
        bands = {}
        if MINHASH_OPTIMIZE_MEMORY:
            # Use dictionary with estimated capacity for better performance
            bands = {}
            estimated_buckets = len(papers) // 10  # Estimate bucket count
            if estimated_buckets > 1000:
                # For very large datasets, consider memory usage
                logger.debug("Using optimized memory mode for MinHash LSH with %d documents", len(papers))

        for idx, signature in enumerate(minhash_signatures):
            if signature is None:
                continue

            paper = text_to_paper[idx]
            for band_idx in range(num_bands):
                start = band_idx * rows_per_band
                end = start + rows_per_band
                band = tuple(signature[start:end])

                # Create band key with threshold optimization
                if use_lsh_threshold:
                    # Apply additional threshold to reduce false positives
                    band_key = (band_idx, band, hash(str(band)) % 1000 < int(MINHASH_LSH_THRESHOLD * 1000))
                else:
                    band_key = (band_idx, band)

                if band_key not in bands:
                    bands[band_key] = []
                bands[band_key].append((idx, paper))

        # Find candidate pairs from LSH buckets with size optimization
        candidate_pairs = set()
        large_buckets = 0

        for band_key, bucket in bands.items():
            if len(bucket) > 1:
                # Skip extremely large buckets to avoid O(n²) explosion
                if len(bucket) > 100:
                    large_buckets += 1
                    if large_buckets <= 5:  # Log first few occurrences
                        logger.warning("MinHash LSH bucket too large (%d items), skipping some pairs", len(bucket))
                    continue

                # All pairs in this bucket are candidates
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        pair = tuple(sorted([bucket[i][0], bucket[j][0]]))
                        candidate_pairs.add(pair)

        # Verify candidate pairs with SequenceMatcher
        comparisons = 0
        removed_indices = set()

        for idx1, idx2 in candidate_pairs:
            if max_pairs and comparisons >= max_pairs:
                break

            if idx1 in removed_indices or idx2 in removed_indices:
                continue

            sig1 = minhash_signatures[idx1]
            sig2 = minhash_signatures[idx2]

            if sig1 is None or sig2 is None:
                continue

            # Calculate actual similarity
            text1 = (
                " ".join(str(text_to_paper[idx1].get(key, "")) for key in ("title", "abstract", "summary"))
                .strip()
                .lower()
            )
            text2 = (
                " ".join(str(text_to_paper[idx2].get(key, "")) for key in ("title", "abstract", "summary"))
                .strip()
                .lower()
            )

            similarity = SequenceMatcher(None, text1, text2).ratio()
            comparisons += 1

            if similarity >= threshold:
                # Remove the second document
                removed_indices.add(idx2)

        # Return filtered list
        result = []
        for idx, paper in enumerate(text_to_paper):
            if idx not in removed_indices:
                result.append(paper)

        return result

    def _apply_minhash_datasketch(
        self,
        papers: list[dict[str, Any]],
        threshold: float,
        max_pairs: int | None = None,
    ) -> list[dict[str, Any]]:
        """Apply MinHash using datasketch library for optimized performance.

        Args:
            papers: List of paper dictionaries
            threshold: Similarity threshold for considering duplicates
            max_pairs: Maximum number of pairwise comparisons (not used in datasketch)

        Returns:
            List of papers with near duplicates removed
        """
        if not papers:
            return []

        # Create LSH index with configurable parameters
        lsh = MinHashLSH(num_perm=MINHASH_NUM_PERMUTATIONS, num_bands=MINHASH_NUM_BANDS, threshold=threshold)

        # Track minhashes and papers for later retrieval
        minhashes = []
        paper_map = {}

        # Process each paper
        whitespace_regex = re.compile(r"\s+")

        for idx, paper in enumerate(papers):
            # Extract and normalize text
            full_text = " ".join(str(paper.get(key, "")) for key in ("title", "abstract", "summary")).strip()
            normalised = whitespace_regex.sub(" ", full_text.lower())

            if not normalised:
                continue

            # Create MinHash
            mh = MinHash(num_perm=MINHASH_NUM_PERMUTATIONS)
            words = set(normalised.split())

            # Optimize large documents
            if MINHASH_OPTIMIZE_MEMORY and len(words) > 1000:
                words = set(list(words)[:1000])

            # Add words to MinHash
            for word in words:
                mh.update(word.encode("utf-8"))

            # Insert into LSH
            try:
                lsh.insert(str(idx), mh)
                minhashes.append(mh)
                paper_map[str(idx)] = (idx, paper)
            except Exception as e:
                logger.debug("Failed to insert paper %d into LSH: %s", idx, e)

        # Find duplicates
        removed_indices = set()
        comparisons = 0

        for idx_str, (orig_idx, paper) in paper_map.items():
            if orig_idx in removed_indices:
                continue

            # Query for near duplicates
            try:
                result = lsh.query(minhashes[int(idx_str)])
                for dup_idx_str in result:
                    dup_idx = paper_map.get(dup_idx_str, (None, None))[0]
                    if dup_idx is not None and dup_idx != orig_idx and dup_idx not in removed_indices:
                        # Verify with actual similarity to avoid false positives
                        text1 = (
                            " ".join(str(paper.get(key, "")) for key in ("title", "abstract", "summary"))
                            .strip()
                            .lower()
                        )
                        text2 = (
                            " ".join(str(papers[dup_idx].get(key, "")) for key in ("title", "abstract", "summary"))
                            .strip()
                            .lower()
                        )

                        similarity = SequenceMatcher(None, text1, text2).ratio()
                        comparisons += 1

                        if similarity >= threshold:
                            removed_indices.add(dup_idx)

                            # Check max_pairs if specified
                            if max_pairs and comparisons >= max_pairs:
                                break
            except Exception as e:
                logger.debug("LSH query failed for paper %d: %s", orig_idx, e)

        # Return filtered papers
        result = []
        for idx, paper in enumerate(papers):
            if idx not in removed_indices:
                result.append(paper)

        logger.debug("Datasketch MinHash processed %d papers with %d comparisons", len(papers), comparisons)
        return result

    def filter_by_criteria(
        self,
        papers: Iterable[dict[str, Any]],
        *,
        year_range: list[int] | None = None,
        study_types: Iterable[str] | None = None,
        min_sample_size: int | None = None,
        min_ranking_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Filter ranked papers according to user-provided criteria."""

        def _within_year_range(paper: dict[str, Any]) -> bool:
            if not year_range or len(year_range) != 2:
                return True
            year = paper.get("publication_year") or paper.get("year")
            if not isinstance(year, int):
                return True
            start, end = year_range
            if start is not None and year < start:
                return False
            if end is not None and year > end:
                return False
            return True

        def _matches_study_type(paper: dict[str, Any]) -> bool:
            if not study_types:
                return True
            types_available = [str(value).lower() for value in (paper.get("study_types") or paper.get("tags") or [])]
            allowed = {str(st).lower() for st in study_types}
            return bool(allowed.intersection(types_available))

        def _meets_sample_size(paper: dict[str, Any]) -> bool:
            if min_sample_size is None or min_sample_size <= 0:
                return True
            sample_size = paper.get("sample_size")
            if isinstance(sample_size, (int, float)):
                return sample_size >= min_sample_size
            abstract = paper.get("abstract") or ""
            estimated = self._estimate_sample_size_from_text(abstract)
            return estimated is None or estimated >= min_sample_size

        def _meets_ranking(paper: dict[str, Any]) -> bool:
            if min_ranking_score is None:
                return True
            score = paper.get("relevance_score")
            if score is None:
                score = paper.get("ranking_score", 0.0)
            return score >= float(min_ranking_score)

        filtered: list[dict[str, Any]] = []
        for paper in papers:
            if not (
                _within_year_range(paper)
                and _matches_study_type(paper)
                and _meets_sample_size(paper)
                and _meets_ranking(paper)
            ):
                continue
            filtered.append(paper)
        return filtered

    def get_ranking_explanation(self, paper: dict[str, Any], *, verbose: bool = False) -> str:
        """Return a human-readable explanation for the ranking score."""
        breakdown = paper.get("ranking_breakdown") or {}
        score = paper.get("ranking_score", 0.0)
        parts = [f"overall score {score:.2f}"]
        for key in ("quality", "recency", "sample_size", "species", "pharma_relevance"):
            value = breakdown.get(key)
            if value is None:
                continue
            parts.append(f"{key.replace('_', ' ')} {float(value):.2f}")

        if verbose:
            # Extract basic details from existing paper metadata rather than
            # referencing non-existent ranking_details
            drug_names = paper.get("drug_names", [])
            if drug_names:
                drug_list = [str(drug) for drug in drug_names[:3]]
                if drug_list:
                    parts.append("drugs " + ", ".join(drug_list))

            mesh_terms = paper.get("mesh_terms", [])
            if mesh_terms:
                mesh_list = [str(term) for term in mesh_terms[:3]]
                if mesh_list:
                    parts.append("mesh " + ", ".join(mesh_list))

            cyp_enzymes = paper.get("cyp_enzymes", [])
            if cyp_enzymes:
                cyp_list = [str(enzyme) for enzyme in cyp_enzymes[:3]]
                if cyp_list:
                    parts.append("cyp " + ", ".join(cyp_list))

            # Add year and study type context when available
            year = paper.get("publication_year") or paper.get("year")
            if year:
                parts.append(f"year {year}")

            # Always include study type if available
            study_type = paper.get("study_type")
            if study_type:
                parts.append(f"type {study_type}")
            else:
                # If no primary study_type, use first one or two from study_types
                study_types = paper.get("study_types", [])
                if study_types and isinstance(study_types, list):
                    if len(study_types) == 1:
                        parts.append(f"type {study_types[0]}")
                    elif len(study_types) >= 2:
                        parts.append(f"type {study_types[0]}, {study_types[1]}")
                    else:
                        parts.append(f"type {study_types[0]}")

        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _normalise_weights(self) -> None:
        total = sum(abs(value) for value in self.weights.values()) or 1.0
        for key, value in list(self.weights.items()):
            self.weights[key] = max(0.0, float(value)) / total

    def _weighted_score(self, breakdown: RankingBreakdown) -> float:
        return sum(self.weights[name] * getattr(breakdown, name) for name in self.weights if hasattr(breakdown, name))

    def _calculate_study_quality_score(self, tags: Any) -> float:
        if not tags:
            return 0.5
        if isinstance(tags, str):
            tags_iterable = [tags]
        else:
            tags_iterable = list(tags)
        scores: list[float] = []
        for tag in tags_iterable:
            lower = str(tag).lower()

            # First try exact matching
            if lower in _HIGH_QUALITY_TAGS:
                scores.append(_HIGH_QUALITY_TAGS[lower])
            elif lower in _LOWER_QUALITY_TAGS:
                scores.append(_LOWER_QUALITY_TAGS[lower])
            else:
                # Try pattern-based matching for unrecognized tags
                score = self._match_tag_with_patterns(lower)
                scores.append(score)

        return sum(scores) / len(scores)

    def _match_tag_with_patterns(self, tag: str) -> float:
        """Use pattern matching to identify study types from non-standard tags."""
        # High-quality patterns
        if (
            re.search(r"\brandomized.*trial\b", tag, re.IGNORECASE)
            or re.search(r"\brandomised.*trial\b", tag, re.IGNORECASE)
            or re.search(r"\brct\b", tag, re.IGNORECASE)
        ):
            return 0.9
        elif re.search(r"\bphase\s+([ivx]+|\d+)", tag, re.IGNORECASE):
            # Extract phase number for scoring
            phase_match = re.search(r"\bphase\s+([ivx]+|\d+)", tag, re.IGNORECASE)
            if phase_match:
                phase = phase_match.group(1).lower()
                # Normalize roman numerals to digits
                roman_map = {"iv": "4", "iii": "3", "ii": "2", "i": "1"}
                normalized_phase = roman_map.get(phase, phase)

                if normalized_phase in ["4", "iv"]:
                    return 0.88
                elif normalized_phase in ["3", "iii"]:
                    return 0.88
                elif normalized_phase in ["2", "ii"]:
                    return 0.82
                elif normalized_phase in ["1", "i"]:
                    return 0.78
        elif re.search(r"\bmeta\s*-?\s*analysis\b", tag, re.IGNORECASE):
            return 0.95
        elif re.search(r"\bsystematic\s+review\b", tag, re.IGNORECASE):
            return 0.95
        elif re.search(r"\bcohort\b", tag, re.IGNORECASE):
            return 0.7
        elif re.search(r"\bcase\s*-?\s*control\b", tag, re.IGNORECASE):
            return 0.7
        elif re.search(r"\bobservational\b", tag, re.IGNORECASE):
            return 0.7
        elif re.search(r"\bobservational\s+(?:study|studies)\b", tag, re.IGNORECASE):
            return 0.7

        # Lower-quality patterns
        elif re.search(r"\bcase\s+report\b", tag):
            return 0.4
        elif re.search(r"\banimal\b", tag):
            return 0.4
        elif re.search(r"\bin\s+vitro\b", tag):
            return 0.35
        elif tag in ["letter", "editorial", "comment"]:
            return 0.3

        # Default score for unknown patterns
        return 0.6

    def _calculate_recency_score(self, year: Any) -> float:
        current_year = datetime.utcnow().year
        try:
            year_int = int(year)
        except (TypeError, ValueError):
            return 0.5
        age = max(0, current_year - year_int)
        # Exponential decay favouring recent publications, configurable via recency_decay_years
        return math.exp(-age / self.recency_decay_years)

    def _calculate_sample_size_score(self, paper: dict[str, Any]) -> float:
        sample_size = paper.get("sample_size")
        if isinstance(sample_size, (int, float)):
            sample_size = int(sample_size)
        elif isinstance(sample_size, str):
            sample_size = self._parse_sample_size_candidate(sample_size.replace(",", ""))
        else:
            sample_size = None

        if sample_size is None:
            abstract = paper.get("abstract") or paper.get("summary") or ""
            sample_size = self._estimate_sample_size_from_text(abstract)

        if sample_size is None:
            return 0.55
        if sample_size >= 1000:
            return 1.0
        if sample_size >= 300:
            return 0.9
        if sample_size >= 100:
            return 0.8
        if sample_size >= 50:
            return 0.65
        if sample_size >= 20:
            return 0.5
        return 0.35

    def _estimate_sample_size_from_text(self, text: str) -> int | None:
        if not text:
            return None

        # Keywords that indicate valid sample size context
        context_keywords = [
            "patients",
            "subjects",
            "participants",
            "enrolled",
            "included",
            "sample",
            "cohort",
            "population",
            "n=",
            "n =",
            "total",
        ]

        counts_by_priority: dict[str, list[int]] = {name: [] for name, _ in _SAMPLE_SIZE_PATTERNS}
        for name, pattern in _SAMPLE_SIZE_PATTERNS:
            for match in pattern.finditer(text):
                raw_value = match.group("count")
                # Extract context around the match for year detection
                match_start = match.start()
                match_end = match.end()
                context_window = 50  # Characters before and after
                context_start = max(0, match_start - context_window)
                context_end = min(len(text), match_end + context_window)
                context = text[context_start:context_end]

                parsed = self._parse_sample_size_candidate(raw_value, context)
                if parsed is not None:
                    # Check proximity to keywords for better scoring
                    context.lower()

                    # Score based on keyword proximity
                    keyword_score = 0
                    for keyword in context_keywords:
                        if keyword in context:
                            keyword_score += 1

                    # Only accept if there's at least one keyword in context
                    if keyword_score > 0:
                        counts_by_priority[name].append(parsed)

        for priority_name in ("n_equals", "term_assignment", "term_trailing"):
            values = counts_by_priority.get(priority_name, [])
            if values:
                return max(values)
        return None

    def _parse_sample_size_candidate(self, raw_value: str | None, context: str = "") -> int | None:
        if not raw_value:
            return None
        normalized = raw_value.replace(",", "")
        try:
            value = int(normalized)
        except ValueError:
            return None

        # Filter out years (already present)
        if len(normalized) == 4 and 1900 <= value <= 2100:
            # Additional check for date context words
            date_context_keywords = [
                "year",
                "years",
                "period",
                "duration",
                "follow-up",
                "followup",
                "study period",
                "from",
                "to",
                "between",
                "during",
                "until",
            ]
            context_lower = context.lower()

            # If near date context words, definitely treat as year, not sample size
            if any(keyword in context_lower for keyword in date_context_keywords):
                return None

        # Add plausibility checks - discard unrealistic sample sizes
        # Most studies have sample sizes between 1 and 100,000
        if value < 1:
            return None
        if value > 100000:  # High percentile cutoff
            return None

        return value

    def _calculate_species_preference_score(self, text: str) -> float:
        if not text:
            return 0.5

        # Tokenize text for stricter matching
        import re

        tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
        lower_text = text.lower()

        # Check for negation terms including non-human variants
        negation_patterns = [
            r"\bin vitro\b",
            r"\bcell culture\b",
            r"\bcultured cells\b",
            r"\btissue culture\b",
            r"\bcell line\b",
            r"\bnon-?human\b",  # Handles both 'non-human' and 'nonhuman'
        ]

        has_negation = any(re.search(pattern, lower_text) for pattern in negation_patterns)

        # Special handling for non-human detection
        non_human_match = re.search(r"\bnon-?human\b", lower_text)

        # Check for exact token matches to reduce false positives
        for species, weight in _SPECIES_PRIORITIES.items():
            if species in tokens:
                # Down-rank human scores if negation terms present OR if non-human detected
                if species in ("human", "humans"):
                    if has_negation or non_human_match:
                        return weight * 0.3  # Significantly reduce score
                return weight
        return 0.45

    def _calculate_pharmaceutical_relevance_score(
        self,
        paper: dict[str, Any],
        query: str = "",
        entities: list[str] | None = None,
        intent: str = "",
    ) -> float:
        text_fields = [
            paper.get("title"),
            paper.get("abstract"),
            paper.get("summary"),
            " ".join(str(term) for term in (paper.get("mesh_terms") or [])),
            " ".join(str(term) for term in (paper.get("keywords") or [])),
        ]
        combined_text = " ".join(fragment for fragment in text_fields if fragment)
        lower_text = combined_text.lower()
        if not lower_text:
            lower_text = ""

        score = 0.0
        for keyword, weight in _PHARMA_RELEVANCE_KEYWORDS.items():
            if keyword in lower_text:
                score += weight
        if query:
            lower_query = query.lower()
            for keyword, weight in _PHARMA_RELEVANCE_KEYWORDS.items():
                if keyword in lower_query:
                    score += weight * 0.5

        query_terms = {token for token in re.split(r"[^a-z0-9]+", query.lower()) if len(token) >= 4}

        drug_terms: set[str] = set()
        for entry in paper.get("drug_names", []) or []:
            if isinstance(entry, dict):
                name = entry.get("name")
            else:
                name = entry
            if name:
                drug_terms.add(str(name).lower())

        if drug_terms:
            score += min(0.2, 0.05 * len(drug_terms))

        if query_terms:
            overlap = {term for term in query_terms if term in lower_text or term in drug_terms}
            if overlap:
                score += min(0.2, 0.05 * len(overlap))

        mesh_terms = {str(term).lower() for term in (paper.get("mesh_terms") or []) if term}
        if query_terms and mesh_terms:
            mesh_overlap = query_terms.intersection(mesh_terms)
            if mesh_overlap:
                score += min(0.15, 0.05 * len(mesh_overlap))

        cyp_entries = {str(entry).lower().replace(" ", "") for entry in (paper.get("cyp_enzymes") or []) if entry}
        if cyp_entries:
            overlap = cyp_entries & _KNOWN_CYP_ENZYMES
            if query_terms:
                overlap |= {cyp for cyp in cyp_entries if any(term in cyp for term in query_terms)}
            if overlap:
                score += min(0.2, 0.1 * len(overlap))

        haystack = lower_text
        haystack_tokens = self._text_tokens(haystack)
        normalized_entities = list(
            dict.fromkeys(
                entity.strip().lower()
                for entity in entities or []
                if isinstance(entity, str) and entity.strip()
            )
        )
        entity_matches = [
            entity
            for entity in normalized_entities
            if self._contains_token_phrase(haystack_tokens, entity)
        ]
        if len(entity_matches) >= 2:
            score += 0.25
        elif len(entity_matches) == 1:
            score += 0.1

        if intent == "drug_interaction" and "interaction" in haystack:
            score += 0.15
        if intent == "timing" and any(
            term in haystack for term in ("pharmacokinetics", "exposure", "auc", "half-life")
        ):
            score += 0.15

        return min(score, 1.0)

    @staticmethod
    def _text_tokens(text: str) -> tuple[str, ...]:
        return tuple(TOKEN_PATTERN.findall(text.lower()))

    def _contains_token_phrase(self, haystack_tokens: tuple[str, ...], phrase: str) -> bool:
        phrase_tokens = self._text_tokens(phrase)
        if not phrase_tokens:
            return False
        if len(phrase_tokens) == 1:
            return phrase_tokens[0] in haystack_tokens

        phrase_length = len(phrase_tokens)
        for start in range(0, len(haystack_tokens) - phrase_length + 1):
            if haystack_tokens[start : start + phrase_length] == phrase_tokens:
                return True
        return False

    @staticmethod
    def _lengths_similar(lhs: str, rhs: str) -> bool:
        max_len = max(len(lhs), len(rhs), 1)
        diff = abs(len(lhs) - len(rhs))
        return diff / max_len <= 0.6


__all__ = ["StudyRankingFilter"]
