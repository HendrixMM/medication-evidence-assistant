"""Meta-summary generation and comparative analysis engine for pharmaceutical research."""
from __future__ import annotations

import logging
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

try:
    from .pharmaceutical_processor import PharmaceuticalProcessor
except ImportError:
    try:
        from pharmaceutical_processor import PharmaceuticalProcessor  # type: ignore
    except ImportError:
        PharmaceuticalProcessor = None

from pydantic import BaseModel, Field, ValidationError

try:
    from .paper_schema import Paper
except ImportError:  # pragma: no cover - module executed outside package context
    from paper_schema import Paper  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class KeyFinding:
    """Structured representation of a key finding from a paper."""

    paper_id: str
    finding: str
    evidence_level: str
    confidence: float
    drug_entities: list[str]
    pk_parameters: dict[str, Any]
    study_type: str


class BulletPoint(BaseModel):
    """Structured representation of a summary bullet point."""

    text: str = Field(..., description="The bullet point text")
    finding_type: str = Field(default="general", description="Type of finding (key, comparative, gap)")
    confidence_score: float | None = Field(default=None, description="Confidence score for this bullet point")
    source_count: int | None = Field(default=None, description="Number of sources supporting this point")


class ComparativeAnalysis(BaseModel):
    """Structured representation of comparative analysis results."""

    convergent_findings: list[dict[str, Any]] = Field(
        default_factory=list, description="Findings that converge across papers"
    )
    divergent_findings: list[dict[str, Any]] = Field(
        default_factory=list, description="Findings that diverge across papers"
    )
    dose_response_patterns: list[dict[str, Any]] = Field(
        default_factory=list, description="Identified dose-response patterns"
    )
    population_differences: list[dict[str, Any]] = Field(
        default_factory=list, description="Population-specific differences"
    )
    methodological_variations: list[dict[str, Any]] = Field(
        default_factory=list, description="Methodological variations across studies"
    )
    sample_size_comparison: dict[str, Any] = Field(
        default_factory=dict, description="Sample size comparison statistics"
    )
    contradictions: list[dict[str, Any]] = Field(default_factory=list, description="Identified contradictions")


class MetaSummary(BaseModel):
    """Structured representation of a meta-summary with comprehensive analysis."""

    query: str = Field(..., description="Original query that generated this summary")
    total_papers: int = Field(..., description="Total number of papers analyzed")
    key_findings: list[dict[str, Any]] = Field(default_factory=list, description="Key findings extracted from papers")
    bullet_points: list[str] = Field(default_factory=list, description="Summary bullet points")
    comparative_analysis: ComparativeAnalysis = Field(
        default_factory=ComparativeAnalysis, description="Comparative analysis results"
    )
    evidence_synthesis: dict[str, Any] = Field(default_factory=dict, description="Evidence level synthesis")
    confidence_scores: dict[str, float] = Field(
        default_factory=dict, description="Confidence scores for various aspects"
    )
    research_gaps: list[str] = Field(default_factory=list, description="Identified research gaps")
    citations: list[str] = Field(default_factory=list, description="Formatted citations")
    summary_statistics: dict[str, Any] = Field(default_factory=dict, description="Summary statistics")
    error: str | None = Field(default=None, description="Error message if generation failed")


class BulletGenerationStrategy:
    """Unified strategy for generating bullet points with normalization and similarity deduping."""

    def __init__(
        self,
        max_points: int = 7,
        min_points: int = 3,
        include_bullet_glyph: bool = False,
        normalization_enabled: bool = True,
        similarity_threshold: float = 0.75,
    ):
        """Initialize the bullet generation strategy.

        Args:
            max_points: Maximum number of bullet points (3-7 constraint)
            min_points: Minimum number of bullet points (3-7 constraint)
            include_bullet_glyph: Whether to prefix bullets with glyph
            normalization_enabled: Enable text normalization for deduplication
            similarity_threshold: Threshold for similarity-based deduplication
        """
        self.max_points = max(min_points, min(max_points, 7))
        self.min_points = max(3, min_points)  # Ensure minimum of 3
        self.bullet_prefix = "• " if include_bullet_glyph else ""
        self.normalization_enabled = normalization_enabled
        self.similarity_threshold = similarity_threshold

        # State for deduplication
        self.normalized_texts: list[str] = []
        self.normalized_signatures: list[set[str]] = []

    def generate_bullets(
        self, key_findings: list[Any], comparative_analysis: dict[str, Any], research_gaps: list[str]
    ) -> list[str]:
        """Generate bullet points using unified strategy."""
        bullet_points: list[str] = []

        # Reset state
        self.normalized_texts = []
        self.normalized_signatures = []

        try:
            findings = key_findings or []
            comparative_analysis = comparative_analysis or {}
            research_gaps = research_gaps or []

            # Generate candidates from different sources
            key_candidates = self._build_key_candidates(findings)
            comparative_candidates = self._build_comparative_candidates(comparative_analysis)
            gap_candidates = self._build_gap_candidates(research_gaps)

            # Process candidates with priority ordering
            supplemental_candidates: list[str] = []

            def reserve_category(candidates: list[str]) -> None:
                if not candidates:
                    return
                if len(bullet_points) >= self.max_points:
                    supplemental_candidates.extend(candidates)
                    return

                added = False
                for idx, candidate in enumerate(candidates):
                    if self._try_add_point(candidate, bullet_points):
                        supplemental_candidates.extend(candidates[idx + 1 :])
                        added = True
                        break
                if not added:
                    supplemental_candidates.extend(candidates)

            # Process in priority order
            reserve_category(key_candidates)
            reserve_category(comparative_candidates)
            reserve_category(gap_candidates)

            # Add supplemental candidates if space
            for candidate in supplemental_candidates:
                self._try_add_point(candidate, bullet_points)
                if len(bullet_points) >= self.max_points:
                    break

            # Ensure minimum bullet count
            if len(bullet_points) < self.min_points:
                self._add_fallback_bullets(bullet_points, len(findings))

            # Truncate to max_points if needed
            if len(bullet_points) > self.max_points:
                bullet_points = bullet_points[: self.max_points]

            return bullet_points

        except Exception as e:
            logger.error(f"Error in unified bullet generation strategy: {e}")
            return [f"{self.bullet_prefix}Error generating summary points: {str(e)}"]

    def _normalize_tokens(self, text: str) -> set[str]:
        """Normalize text to tokens for similarity comparison."""
        if not self.normalization_enabled:
            return {text.lower()}

        tokens = re.split(r"[^a-z0-9]+", text.lower())
        keywords = {token for token in tokens if len(token) > 2}
        return keywords or {text.lower()}

    def _try_add_point(self, point: str, bullet_points: list[str]) -> bool:
        """Try to add a point with deduplication."""
        if len(bullet_points) >= self.max_points or not point:
            return False

        clean_point = (
            point[len(self.bullet_prefix) :] if self.bullet_prefix and point.startswith(self.bullet_prefix) else point
        )
        clean_point = clean_point.strip()
        if not clean_point:
            return False

        lowered = clean_point.lower()
        if lowered in self.normalized_texts:
            return False

        # Similarity-based deduplication
        if self.normalization_enabled:
            candidate_signature = self._normalize_tokens(clean_point)
            for existing_signature in self.normalized_signatures:
                union = candidate_signature | existing_signature
                if not union:
                    continue
                similarity = len(candidate_signature & existing_signature) / len(union)
                if similarity >= self.similarity_threshold:
                    return False

            self.normalized_signatures.append(candidate_signature)

        self.normalized_texts.append(lowered)
        bullet_points.append(point)
        return True

    def _build_key_candidates(self, findings: list[Any]) -> list[str]:
        """Build candidates from key findings with evidence weighting."""
        candidates: list[str] = []

        # Sort by evidence quality and confidence
        evidence_weights = {
            "Level 1": 1.0,
            "Level 2": 0.9,
            "Level 3": 0.7,
            "Level 4": 0.6,
            "Level 5": 0.4,
            "Level 6": 0.2,
        }

        weighted_findings = []
        for finding in findings:
            evidence_weight = evidence_weights.get(getattr(finding, "evidence_level", ""), 0.3)
            confidence_weight = getattr(finding, "confidence", 0.5)
            combined_weight = evidence_weight * confidence_weight
            weighted_findings.append((finding, combined_weight))

        # Sort by combined weight
        weighted_findings.sort(key=lambda x: x[1], reverse=True)

        for finding, weight in weighted_findings:
            context_bits: list[str] = []
            if getattr(finding, "evidence_level", None):
                context_bits.append(str(finding.evidence_level))
            if getattr(finding, "study_type", None):
                context_bits.append(str(finding.study_type))
            context = f" ({', '.join(context_bits)})" if context_bits else ""

            bullet = f"{self.bullet_prefix}{finding.finding}{context}"
            if getattr(finding, "drug_entities", None):
                top_drugs = ", ".join(finding.drug_entities[:2])
                bullet += f" - Drugs: {top_drugs}"
            candidates.append(bullet)

        return candidates

    def _build_comparative_candidates(self, comparative_analysis: dict[str, Any]) -> list[str]:
        """Build candidates from comparative analysis."""
        candidates: list[str] = []

        # Convergent findings (sorted by weighted score)
        convergent_findings = comparative_analysis.get("convergent_findings", [])
        convergent_findings_sorted = sorted(convergent_findings, key=lambda x: x.get("weighted_score", 0), reverse=True)

        for conv in convergent_findings_sorted:
            drug = conv.get("drug", "unknown target")
            finding = conv.get("convergent_finding", "aligned outcomes identified")
            studies = conv.get("papers_count")
            weight = conv.get("finding_weight", 0.5)
            count_note = f" (evidence score: {weight:.2f}, {studies} studies)" if studies else ""
            candidates.append(f"{self.bullet_prefix}Convergent finding for {drug}: {finding}{count_note}")

        # Divergent findings
        for divergent in comparative_analysis.get("divergent_findings", []):
            entity = divergent.get("entity", "target")
            directions = ", ".join(divergent.get("directions", [])) or "mixed directions"
            study_count = len(divergent.get("papers", []))
            candidates.append(
                f"{self.bullet_prefix}Divergent evidence for {entity}: {directions} across {study_count} studies"
            )

        # Contradictions (transparency)
        for contradiction in comparative_analysis.get("contradictions", []):
            drug = contradiction.get("drug", "target")
            detail = contradiction.get("details", "conflicting findings identified")
            candidates.append(f"{self.bullet_prefix}[Note] {drug}: {detail}")

        # Sample size patterns
        sample_summary = (comparative_analysis.get("sample_size_comparison") or {}).get("aggregate", {})
        if sample_summary.get("min") and sample_summary.get("max"):
            median = sample_summary.get("median")
            spread = f"{sample_summary['min']} to {sample_summary['max']} participants"
            median_note = f" (median {median})" if median else ""
            candidates.append(f"{self.bullet_prefix}Sample sizes span {spread}{median_note} across the evidence base")

        return candidates

    def _build_gap_candidates(self, research_gaps: list[str]) -> list[str]:
        """Build candidates from research gaps."""
        return [f"{self.bullet_prefix}Research gap: {gap}" for gap in research_gaps if gap]

    def _add_fallback_bullets(self, bullet_points: list[str], findings_count: int) -> None:
        """Add fallback bullets to meet minimum count requirement."""
        safe_fallback_notes = [
            f"{self.bullet_prefix}[Meta-analysis] Evidence synthesis based on {findings_count} weighted findings",
            f"{self.bullet_prefix}[Methodology] Analysis employed evidence quality weighting (study design, sample size, consistency)",
            f"{self.bullet_prefix}[Quality] Current synthesis represents best available evidence with transparency scoring",
            f"{self.bullet_prefix}[Evidence] Quality-weighted analysis performed across available literature",
            f"{self.bullet_prefix}[Future research] Additional high-quality studies would strengthen confidence in conclusions",
        ]

        for fallback_note in safe_fallback_notes:
            if len(bullet_points) >= self.min_points:
                break
            self._try_add_point(fallback_note, bullet_points)

        # Final fallback for edge cases
        fallback_counter = 1
        while len(bullet_points) < self.min_points:
            if len(bullet_points) >= self.max_points:
                break
            fallback_point = f"{self.bullet_prefix}[Meta-note] Evidence synthesis placeholder {fallback_counter}"
            if not self._try_add_point(fallback_point, bullet_points):
                bullet_points.append(fallback_point)
            fallback_counter += 1


class SynthesisEngine:
    """Generate meta-summaries and comparative analysis from pharmaceutical research papers.

    Provides methods to synthesize findings across multiple papers, perform comparative
    analysis, and generate structured summaries with evidence levels and citations.

    Example usage:
        engine = SynthesisEngine()
        papers = [{"content": "...", "metadata": {...}}, ...]
        summary = engine.generate_meta_summary(papers, "drug interactions", max_bullet_points=5)
    """

    def __init__(self, pharma_processor: PharmaceuticalProcessor | None = None):
        """Initialize synthesis engine with optional pharmaceutical processor.

        Args:
            pharma_processor: Optional PharmaceuticalProcessor for entity extraction.
                             If None, will attempt to create one if available.
        """
        self.pharma_processor = pharma_processor
        if self.pharma_processor is None and PharmaceuticalProcessor is not None:
            try:
                self.pharma_processor = PharmaceuticalProcessor()
            except Exception as e:
                logger.warning(f"Failed to initialize PharmaceuticalProcessor: {e}")

        # Study type patterns for classification (order matters - most specific first)
        self.study_type_patterns = {
            "meta_analysis": [r"meta-analysis", r"systematic review", r"pooled analysis"],
            "clinical_trial": [
                r"clinical trial",
                r"randomized.*trial",
                r"rct\b",
                r"phase\s+(?:i|ii|iii|iv|v|1|2|3|4|5)\s+(?:trial|study|clinical)",
                r"phase\s+(?:i|ii|iii|iv|v|1|2|3|4|5).*(?:trial|study|investigation)",
                r"phase\s+(?:i|ii|iii|iv|v|1|2|3|4|5)\b",
            ],
            "case_report": [r"case report", r"case series"],
            "observational": [r"cohort(?!\s+trial)", r"case-control", r"cross-sectional", r"observational(?!\s+trial)"],
            "in_vitro": [r"in vitro", r"cell culture", r"laboratory", r"microsomal"],
            "review": [r"review", r"narrative", r"expert opinion"],
        }

        # Evidence level mapping
        self.evidence_levels = {
            "meta_analysis": "Level 1",
            "clinical_trial": "Level 2",
            "observational": "Level 3",
            "case_report": "Level 4",
            "in_vitro": "Level 5",
            "review": "Level 6",
        }

    def _prepare_papers(self, papers: list[Any]) -> list[dict[str, Any]]:
        """Validate and normalise paper inputs using the Paper schema."""
        if not papers:
            return []
        prepared: list[dict[str, Any]] = []

        for index, paper in enumerate(papers):
            try:
                paper_model = paper if isinstance(paper, Paper) else Paper.model_validate(paper)
            except ValidationError as exc:
                logger.warning("Skipping paper at index %s due to schema validation error: %s", index, exc)
                continue

            prepared.append(paper_model.as_dict())

        return prepared

    def _ensure_prepared_papers(self, papers: list[Any]) -> list[dict[str, Any]]:
        """Return schema-validated papers, coercing input when necessary."""
        if not papers:
            return []
        if all(isinstance(paper, dict) and paper.get("__paper_schema_validated__") for paper in papers):
            return papers
        return self._prepare_papers(papers)

    def generate_meta_summary(
        self,
        papers: list[dict],
        query: str,
        max_bullet_points: int = 7,
        min_bullet_points: int = 3,
        include_bullet_glyph: bool = False,
    ) -> dict[str, Any]:
        """Generate comprehensive meta-summary from multiple papers.

        Guarantees 3-7 bullet points in the summary through post-processing to ensure
        minimum coverage and avoid redundancy.

        Args:
            papers: Iterable of `Paper` objects or dictionaries with content and metadata
            query: Query context for focused analysis
            max_bullet_points: Maximum number of bullet points in summary (capped at 7)
            min_bullet_points: Minimum number of bullet points in summary (defaults to 3)
            include_bullet_glyph: When True, prefix bullet points with a glyph (e.g., •)

        Returns:
            Dictionary containing structured meta-summary with findings, analysis, and citations
        """
        prepared_papers: list[dict[str, Any]] = []
        try:
            prepared_papers = self._ensure_prepared_papers(papers)
            logger.info(
                "Generating meta-summary for %s schema-validated papers with query: %s",
                len(prepared_papers),
                query,
            )

            if not prepared_papers:
                return {
                    "query": query,
                    "total_papers": 0,
                    "key_findings": [],
                    "bullet_points": [],
                    "comparative_analysis": {},
                    "evidence_synthesis": {},
                    "confidence_scores": {},
                    "research_gaps": [],
                    "citations": [],
                    "summary_statistics": {
                        "study_types": {},
                        "drug_mentions": {},
                        "evidence_distribution": {},
                    },
                }

            # Extract and structure key findings
            key_findings = self._extract_key_findings(prepared_papers)
            logger.debug(f"Extracted {len(key_findings)} key findings")

            # Perform comparative analysis
            comparative_analysis = self._perform_comparative_analysis(prepared_papers, query)

            # Identify research gaps ahead of summary synthesis so bullet points can reference them
            research_gaps = self._identify_research_gaps(prepared_papers, query)

            # Synthesize evidence levels
            evidence_synthesis = self._synthesize_evidence_levels(prepared_papers)

            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(prepared_papers)

            # Generate bullet points with minimum coverage guarantees
            bullet_point_limit = max(min_bullet_points, min(max_bullet_points, 7))
            bullet_points = self._generate_bullet_points(
                key_findings,
                comparative_analysis,
                research_gaps,
                bullet_point_limit,
                min_bullet_points,
                include_bullet_glyph=include_bullet_glyph,
            )

            # Format citations
            citations = self._format_citations(prepared_papers)

            meta_summary = {
                "query": query,
                "total_papers": len(prepared_papers),
                "key_findings": [
                    {
                        "finding": finding.finding,
                        "evidence_level": finding.evidence_level,
                        "confidence": finding.confidence,
                        "paper_id": finding.paper_id,
                        "drug_entities": finding.drug_entities,
                        "study_type": finding.study_type,
                    }
                    for finding in key_findings
                ],
                "bullet_points": bullet_points,
                "comparative_analysis": (
                    comparative_analysis.model_dump()
                    if hasattr(comparative_analysis, "model_dump")
                    else comparative_analysis
                ),
                "evidence_synthesis": evidence_synthesis,
                "confidence_scores": confidence_scores,
                "research_gaps": research_gaps,
                "citations": citations,
                "summary_statistics": {
                    "study_types": self._count_study_types(prepared_papers),
                    "drug_mentions": self._count_drug_mentions(key_findings),
                    "evidence_distribution": self._count_evidence_levels(key_findings),
                },
            }

            logger.info("Meta-summary generation completed successfully")
            return meta_summary

        except Exception as e:
            logger.error(f"Error generating meta-summary: {e}")
            return {
                "error": str(e),
                "query": query,
                "total_papers": len(prepared_papers) if prepared_papers else (len(papers) if papers else 0),
                "key_findings": [],
                "bullet_points": [],
                "comparative_analysis": {},
                "evidence_synthesis": {},
                "confidence_scores": {},
                "research_gaps": [],
                "citations": [],
                "summary_statistics": {"study_types": {}, "drug_mentions": {}, "evidence_distribution": {}},
            }

    def _extract_key_findings(self, papers: list[dict]) -> list[KeyFinding]:
        """Extract structured key findings from papers."""
        papers = self._ensure_prepared_papers(papers)
        findings = []

        for i, paper in enumerate(papers):
            try:
                paper_id = paper.get("metadata", {}).get("pmid") or paper.get("id") or f"paper_{i}"
                content = paper.get("page_content") or paper.get("content", "")
                metadata = paper.get("metadata", {})

                # Classify study type
                study_type = self._classify_study_type(content)
                evidence_level = self.evidence_levels.get(study_type, "Level 6")

                # Extract drug entities if pharmaceutical processor is available
                drug_entities = []
                pk_parameters = {}
                if self.pharma_processor:
                    try:
                        drug_candidates = self.pharma_processor.extract_drug_names(content)
                        drug_entities = [d["name"] for d in drug_candidates if d["confidence"] > 0.8]
                        pk_parameters = self.pharma_processor.extract_pharmacokinetic_parameters(content)
                    except Exception as e:
                        logger.warning(f"Error extracting pharmaceutical entities: {e}")

                # Extract key findings from content
                finding_text = self._extract_main_finding(content, metadata)
                confidence = self._calculate_finding_confidence(content, study_type, metadata)

                finding = KeyFinding(
                    paper_id=paper_id,
                    finding=finding_text,
                    evidence_level=evidence_level,
                    confidence=confidence,
                    drug_entities=drug_entities,
                    pk_parameters=pk_parameters,
                    study_type=study_type,
                )
                findings.append(finding)

            except Exception as e:
                logger.warning(f"Error processing paper {i}: {e}")
                continue

        return sorted(findings, key=lambda x: (-len(x.drug_entities), -x.confidence))

    def _perform_comparative_analysis(self, papers: list[dict], query: str) -> dict[str, Any]:
        """Perform comparative analysis across papers with query-aware weighting."""
        try:
            papers = self._ensure_prepared_papers(papers)

            analysis = {
                "convergent_findings": [],
                "divergent_findings": [],
                "dose_response_patterns": [],
                "population_differences": [],
                "methodological_variations": [],
                "sample_size_comparison": {"per_paper": [], "aggregate": {}},
                "contradictions": [],
            }

            # Extract query terms for relevance scoring
            query_terms = self._extract_query_terms(query)

            # Group papers by similar drug entities or outcomes
            drug_groups = defaultdict(list)
            outcome_groups = defaultdict(list)
            paper_summaries: dict[str, dict[str, Any]] = {}
            processed_divergence_keys: set[str] = set()

            for paper in papers:
                content = paper.get("page_content") or paper.get("content", "")
                metadata = paper.get("metadata", {})
                paper_id = metadata.get("pmid") or paper.get("id") or metadata.get("title", "unknown")

                # Extract drug mentions for grouping
                drugs: list[dict[str, Any]] = []
                if self.pharma_processor:
                    drugs = self.pharma_processor.extract_drug_names(content)
                    for drug in drugs:
                        if drug["confidence"] > 0.8:
                            drug_groups[drug["name"].lower()].append(paper)

                # Extract outcome mentions
                outcomes = self._extract_outcomes(content)
                for outcome in outcomes:
                    outcome_groups[outcome].append(paper)

                study_type = self._classify_study_type(content)
                methodology_descriptors = self._extract_methodology_descriptors(content)
                sample_sizes = self._extract_sample_sizes(content)
                primary_finding = self._extract_main_finding(content, metadata)
                direction = self._infer_finding_direction(primary_finding)
                confidence_intervals = self._extract_confidence_intervals(content)
                population_markers = self._identify_population_markers(content)

                paper_summaries[paper_id] = {
                    "paper_id": paper_id,
                    "study_type": study_type,
                    "methodology_notes": methodology_descriptors,
                    "sample_sizes": sample_sizes,
                    "primary_finding": primary_finding,
                    "direction": direction,
                    "confidence_intervals": confidence_intervals,
                    "drug_mentions": [drug["name"].lower() for drug in drugs if drug.get("confidence", 0) > 0.5]
                    if self.pharma_processor and drugs
                    else [],
                    "population_markers": population_markers,
                }

            # Calculate weighted scoring for findings based on evidence quality and sample size first
            # This needs to be done before convergent findings analysis
            per_paper_samples = []
            all_sample_sizes: list[int] = []
            for summary in paper_summaries.values():
                if summary["sample_sizes"]:
                    per_paper_samples.append(
                        {
                            "paper_id": summary["paper_id"],
                            "sample_sizes": summary["sample_sizes"],
                            "largest_cohort": max(summary["sample_sizes"]),
                            "smallest_cohort": min(summary["sample_sizes"]),
                        }
                    )
                    all_sample_sizes.extend(summary["sample_sizes"])

            weighting_summary = self._calculate_finding_weights(paper_summaries, per_paper_samples)

            # Identify convergent findings (multiple papers, similar results) with query-aware weighting
            for drug, drug_papers in drug_groups.items():
                if len(drug_papers) >= 2:
                    findings_with_relevance = []
                    for paper in drug_papers:
                        content = paper.get("page_content") or paper.get("content", "")
                        finding = self._extract_main_finding(content, paper.get("metadata", {}))
                        relevance_score = self._calculate_query_relevance(content + " " + finding, query_terms)
                        findings_with_relevance.append(
                            {"finding": finding, "relevance_score": relevance_score, "content": content}
                        )

                    # Sort by relevance score so query-relevant findings surface first
                    findings_with_relevance.sort(key=lambda x: x["relevance_score"], reverse=True)
                    findings = [f["finding"] for f in findings_with_relevance]

                    if self._are_findings_convergent(findings):
                        # Calculate weighted score for this convergent finding
                        paper_ids = [paper.get("metadata", {}).get("pmid", "unknown") for paper in drug_papers]
                        finding_weight = self._calculate_convergent_finding_weight(
                            paper_ids, paper_summaries, weighting_summary
                        )

                        analysis["convergent_findings"].append(
                            {
                                "drug": drug,
                                "papers_count": len(drug_papers),
                                "convergent_finding": self._synthesize_convergent_finding(findings),
                                "query_relevance_score": sum(f["relevance_score"] for f in findings_with_relevance)
                                / len(findings_with_relevance),
                                "finding_weight": finding_weight,
                                "weighted_score": finding_weight
                                * (
                                    sum(f["relevance_score"] for f in findings_with_relevance)
                                    / len(findings_with_relevance)
                                ),
                            }
                        )

            # Identify dose-response patterns
            for drug, drug_papers in drug_groups.items():
                dose_responses = self._extract_dose_response_data(drug_papers, drug)
                if dose_responses:
                    analysis["dose_response_patterns"].append({"drug": drug, "patterns": dose_responses})

            # Capture methodological variations per paper
            analysis["methodological_variations"] = [
                {
                    "paper_id": summary["paper_id"],
                    "study_type": summary["study_type"],
                    "methodology_descriptors": summary["methodology_notes"],
                }
                for summary in paper_summaries.values()
            ]

            # Summarize sample sizes (already calculated above)
            aggregate: dict[str, Any] = {}
            if all_sample_sizes:
                aggregate = {
                    "min": min(all_sample_sizes),
                    "max": max(all_sample_sizes),
                    "median": statistics.median(all_sample_sizes),
                    "mean": round(statistics.mean(all_sample_sizes), 2),
                    "papers_with_sample_sizes": len(per_paper_samples),
                }

            analysis["sample_size_comparison"] = {
                "per_paper": per_paper_samples,
                "aggregate": aggregate,
                "weighting_summary": weighting_summary,
            }

            divergent_records: list[dict[str, Any]] = []

            for drug, drug_papers in drug_groups.items():
                related_summaries = [
                    paper_summaries.get(
                        paper.get("metadata", {}).get("pmid")
                        or paper.get("id")
                        or paper.get("metadata", {}).get("title", "unknown")
                    )
                    for paper in drug_papers
                ]
                related_summaries = [s for s in related_summaries if s]
                directional_set = {s["direction"] for s in related_summaries if s["direction"] != "neutral"}

                if len(directional_set.intersection({"increase", "decrease"})) == 2:
                    key = f"drug::{drug}"
                    if key not in processed_divergence_keys:
                        processed_divergence_keys.add(key)
                        divergent_records.append(
                            {
                                "entity": drug,
                                "dimension": "drug",
                                "directions": sorted(directional_set),
                                "papers": [s["paper_id"] for s in related_summaries],
                                "sample_findings": [
                                    s["primary_finding"] for s in related_summaries[:3] if s.get("primary_finding")
                                ],
                            }
                        )

            for outcome, outcome_papers in outcome_groups.items():
                related_summaries = [
                    paper_summaries.get(
                        paper.get("metadata", {}).get("pmid")
                        or paper.get("id")
                        or paper.get("metadata", {}).get("title", "unknown")
                    )
                    for paper in outcome_papers
                ]
                related_summaries = [s for s in related_summaries if s]
                directional_set = {s["direction"] for s in related_summaries if s["direction"] != "neutral"}

                if len(directional_set.intersection({"increase", "decrease"})) == 2:
                    key = f"outcome::{outcome}"
                    if key not in processed_divergence_keys:
                        processed_divergence_keys.add(key)
                        divergent_records.append(
                            {
                                "entity": outcome,
                                "dimension": "outcome",
                                "directions": sorted(directional_set),
                                "papers": [s["paper_id"] for s in related_summaries],
                                "sample_findings": [
                                    s["primary_finding"] for s in related_summaries[:3] if s.get("primary_finding")
                                ],
                            }
                        )

            analysis["divergent_findings"] = divergent_records

            population_differences: list[dict[str, Any]] = []

            for drug, drug_papers in drug_groups.items():
                population_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for paper in drug_papers:
                    summary = paper_summaries.get(
                        paper.get("metadata", {}).get("pmid")
                        or paper.get("id")
                        or paper.get("metadata", {}).get("title", "unknown")
                    )
                    if not summary:
                        continue

                    markers = summary.get("population_markers") or []
                    for marker in markers:
                        population_map[marker].append(
                            {
                                "paper_id": summary["paper_id"],
                                "finding": summary.get("primary_finding"),
                                "direction": summary.get("direction"),
                            }
                        )

                if len(population_map) >= 2:
                    population_differences.append(
                        {"drug": drug, "populations": {label: entries[:3] for label, entries in population_map.items()}}
                    )

            analysis["population_differences"] = population_differences

            # Detect contradictions (directional conflicts or non-overlapping CIs)
            contradictions = []
            for drug, drug_papers in drug_groups.items():
                related_summaries = [
                    paper_summaries.get(
                        paper.get("metadata", {}).get("pmid")
                        or paper.get("id")
                        or paper.get("metadata", {}).get("title", "unknown")
                    )
                    for paper in drug_papers
                ]
                related_summaries = [s for s in related_summaries if s]

                directions = {s["direction"] for s in related_summaries if s["direction"] != "neutral"}
                if len(directions) > 1:
                    contradictions.append(
                        {
                            "drug": drug,
                            "type": "directional_conflict",
                            "details": f"Conflicting directions reported: {', '.join(sorted(directions))}",
                            "papers": [s["paper_id"] for s in related_summaries],
                        }
                    )

                cis = [ci for s in related_summaries for ci in s["confidence_intervals"]]
                conflicting_ci = self._find_conflicting_confidence_intervals(cis)
                if conflicting_ci:
                    contradictions.append(
                        {
                            "drug": drug,
                            "type": "confidence_interval_conflict",
                            "details": f"Non-overlapping confidence intervals detected: {conflicting_ci}",
                            "papers": [s["paper_id"] for s in related_summaries],
                        }
                    )

            analysis["contradictions"] = contradictions

            # Sort convergent findings by weighted score (evidence quality + sample size + relevance)
            analysis["convergent_findings"].sort(key=lambda x: x.get("weighted_score", 0), reverse=True)

            return analysis

        except Exception as e:
            logger.error(f"Error in comparative analysis: {e}")
            return {"error": str(e)}

    def _calculate_finding_weights(
        self, paper_summaries: dict[str, dict], per_paper_samples: list[dict]
    ) -> dict[str, Any]:
        """Calculate weighted scoring based on evidence level, sample size, and consistency."""
        try:
            # Evidence level weights (higher is better)
            evidence_weights = {
                "Level 1": 1.0,  # Systematic reviews/meta-analyses
                "Level 2": 0.9,  # RCTs
                "Level 3": 0.7,  # Cohort studies
                "Level 4": 0.6,  # Case-control studies
                "Level 5": 0.4,  # Case series/reports
                "Level 6": 0.2,  # Expert opinion
                "Unknown": 0.1,
            }

            # Sample size normalization (log scale to prevent huge studies from dominating)
            sample_size_weights = {}
            all_samples = []
            for sample_data in per_paper_samples:
                if sample_data["sample_sizes"]:
                    max_sample = max(sample_data["sample_sizes"])
                    all_samples.append(max_sample)
                    sample_size_weights[sample_data["paper_id"]] = max_sample

            # Normalize sample sizes to 0.1-1.0 range using log scale
            if all_samples:
                max_log_sample = math.log10(max(all_samples) + 1)
                min_log_sample = math.log10(min(all_samples) + 1)
                log_range = max_log_sample - min_log_sample if max_log_sample > min_log_sample else 1

                for paper_id, sample_size in sample_size_weights.items():
                    log_sample = math.log10(sample_size + 1)
                    normalized_weight = 0.1 + 0.9 * ((log_sample - min_log_sample) / log_range)
                    sample_size_weights[paper_id] = normalized_weight

            # Calculate consistency weights (how often findings agree)
            consistency_weights = {}
            for paper_id, summary in paper_summaries.items():
                # Higher consistency for findings that align with majority direction
                direction = summary.get("direction", "neutral")
                all_directions = [s.get("direction", "neutral") for s in paper_summaries.values()]
                direction_counts = Counter(all_directions)
                most_common_direction = direction_counts.most_common(1)[0][0] if direction_counts else "neutral"

                if direction == most_common_direction and direction != "neutral":
                    consistency_weights[paper_id] = 1.0
                elif direction == "neutral":
                    consistency_weights[paper_id] = 0.7
                else:
                    consistency_weights[paper_id] = 0.5

            # Combine weights for each paper
            paper_weights = {}
            for paper_id, summary in paper_summaries.items():
                evidence_level = summary.get("evidence_level", "Unknown")
                evidence_weight = evidence_weights.get(evidence_level, 0.1)
                sample_weight = sample_size_weights.get(paper_id, 0.5)  # Default if no sample size
                consistency_weight = consistency_weights.get(paper_id, 0.5)

                # Combined weight: evidence_quality * sample_size_factor * consistency_factor
                combined_weight = evidence_weight * sample_weight * consistency_weight
                paper_weights[paper_id] = {
                    "evidence_weight": evidence_weight,
                    "sample_size_weight": sample_weight,
                    "consistency_weight": consistency_weight,
                    "combined_weight": combined_weight,
                }

            return {
                "paper_weights": paper_weights,
                "evidence_distribution": dict(
                    Counter(s.get("evidence_level", "Unknown") for s in paper_summaries.values())
                ),
                "sample_size_stats": {
                    "papers_with_samples": len(sample_size_weights),
                    "range": [min(all_samples), max(all_samples)] if all_samples else [0, 0],
                    "median": statistics.median(all_samples) if all_samples else 0,
                },
                "consistency_analysis": dict(Counter(consistency_weights.values())),
                "weighting_methodology": "Combined score: f(evidence_level, sample_size, consistency)",
            }

        except Exception as e:
            logger.error(f"Error calculating finding weights: {e}")
            return {"error": str(e)}

    def _calculate_convergent_finding_weight(
        self, paper_ids: list[str], paper_summaries: dict[str, dict], weighting_summary: dict[str, Any]
    ) -> float:
        """Calculate the combined weight for a convergent finding based on its constituent papers."""
        try:
            paper_weights = weighting_summary.get("paper_weights", {})

            if not paper_weights:
                return 0.5  # Default weight if no weighting data

            # Get weights for papers in this convergent finding
            finding_weights = []
            for paper_id in paper_ids:
                if paper_id in paper_weights:
                    finding_weights.append(paper_weights[paper_id]["combined_weight"])
                else:
                    finding_weights.append(0.3)  # Default for missing papers

            if not finding_weights:
                return 0.5

            # Use weighted average (more papers with high quality contribute more)
            total_weight = sum(finding_weights)
            num_papers = len(finding_weights)

            # Bonus for multiple high-quality papers (up to 20% bonus)
            multi_paper_bonus = min(0.2, (num_papers - 1) * 0.05)

            # Final weight: average + multi-paper bonus, capped at 1.0
            final_weight = min(1.0, (total_weight / num_papers) + multi_paper_bonus)

            return round(final_weight, 3)

        except Exception as e:
            logger.error(f"Error calculating convergent finding weight: {e}")
            return 0.5

    def _generate_bullet_points(
        self,
        key_findings: list[Any],
        comparative_analysis: dict[str, Any],
        research_gaps: list[str],
        max_points: int,
        min_points: int,
        include_bullet_glyph: bool = False,
    ) -> list[str]:
        """Generate bullet points using unified strategy with normalization and similarity deduping."""
        try:
            # Use the unified bullet generation strategy
            bullet_generator = BulletGenerationStrategy(
                max_points=max_points,
                min_points=min_points,
                include_bullet_glyph=include_bullet_glyph,
                normalization_enabled=True,
                similarity_threshold=0.75,
            )

            return bullet_generator.generate_bullets(
                key_findings=key_findings, comparative_analysis=comparative_analysis, research_gaps=research_gaps
            )

        except Exception as e:
            logger.error(f"Error generating bullet points: {e}")
            bullet_prefix = "• " if include_bullet_glyph else ""
            return [f"{bullet_prefix}Error generating evidence-based summary"]

    def _synthesize_evidence_levels(self, papers: list[dict]) -> dict[str, Any]:
        """Synthesize evidence levels across papers."""
        try:
            papers = self._ensure_prepared_papers(papers)
            evidence_counts = Counter()
            total_papers = len(papers)

            for paper in papers:
                content = paper.get("page_content") or paper.get("content", "")
                study_type = self._classify_study_type(content)
                evidence_level = self.evidence_levels.get(study_type, "Level 6")
                evidence_counts[evidence_level] += 1

            # Calculate evidence strength score (weighted by evidence level)
            weights = {"Level 1": 5, "Level 2": 4, "Level 3": 3, "Level 4": 2, "Level 5": 1, "Level 6": 0.5}
            weighted_score = sum(evidence_counts[level] * weights.get(level, 0) for level in evidence_counts)
            max_possible_score = total_papers * 5
            evidence_strength = weighted_score / max_possible_score if max_possible_score > 0 else 0

            return {
                "evidence_distribution": dict(evidence_counts),
                "total_papers": total_papers,
                "evidence_strength_score": round(evidence_strength, 2),
                "primary_evidence_level": evidence_counts.most_common(1)[0][0] if evidence_counts else "Unknown",
                "high_quality_papers": evidence_counts.get("Level 1", 0) + evidence_counts.get("Level 2", 0),
            }

        except Exception as e:
            logger.error(f"Error synthesizing evidence levels: {e}")
            return {"error": str(e)}

    def _extract_methodology_descriptors(self, content: str) -> list[str]:
        """Identify notable methodology descriptors from paper content."""
        descriptors = []
        methodology_patterns = {
            "randomized": r"randomi[sz]ed",
            "double_blind": r"double[-\s]?blind",
            "placebo_controlled": r"placebo[-\s]?controlled",
            "crossover": r"crossover",
            "open_label": r"open[-\s]?label",
            "single_center": r"single[-\s]?center",
            "multi_center": r"multi[-\s]?center",
            "prospective": r"prospective",
            "retrospective": r"retrospective",
        }

        lowered = content.lower()
        for descriptor, pattern in methodology_patterns.items():
            if re.search(pattern, lowered):
                descriptors.append(descriptor.replace("_", " "))

        return descriptors

    def _extract_sample_sizes(self, content: str) -> list[int]:
        """Extract sample sizes from study descriptions."""
        sample_sizes: set[int] = set()
        patterns = [
            r"\b[ns]\s*[:=]\s*(\d{2,4})\b",
            r"\b(\d{2,4})\s+participants\b",
            r"\bcohort of\s+(\d{2,4})",
            r"\benrolled\s+(\d{2,4})\s+patients",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                try:
                    sample_sizes.add(int(match.group(1)))
                except (ValueError, IndexError):
                    continue

        return sorted(sample_sizes)

    def _infer_finding_direction(self, finding: str) -> str:
        """Infer whether a finding indicates increase, decrease, or neutral effect."""
        if not finding:
            return "neutral"

        finding_lower = finding.lower()
        increase_markers = ["increase", "higher", "elevated", "greater", "augment"]
        decrease_markers = ["decrease", "lower", "reduced", "diminish", "attenuate"]

        if any(marker in finding_lower for marker in increase_markers):
            return "increase"
        if any(marker in finding_lower for marker in decrease_markers):
            return "decrease"
        return "neutral"

    def _extract_confidence_intervals(self, content: str) -> list[tuple[float, float]]:
        """Extract numeric confidence intervals from text."""
        intervals: list[tuple[float, float]] = []
        patterns = [
            r"95%\s*(?:ci|confidence interval)[^\d]*(\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)",
            r"ci[:\s]*(\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                try:
                    lower = float(match.group(1))
                    upper = float(match.group(2))
                    if lower > upper:
                        lower, upper = upper, lower
                    intervals.append((lower, upper))
                except (ValueError, IndexError):
                    continue

        return intervals

    def _find_conflicting_confidence_intervals(
        self, intervals: list[tuple[float, float]]
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Identify pairs of non-overlapping confidence intervals."""
        if len(intervals) < 2:
            return None

        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        for i in range(len(sorted_intervals) - 1):
            current = sorted_intervals[i]
            next_interval = sorted_intervals[i + 1]
            if current[1] < next_interval[0] or next_interval[1] < current[0]:
                return current, next_interval

        return None

    def _identify_population_markers(self, content: str) -> list[str]:
        """Detect population-specific markers within study content."""
        population_patterns = {
            "pediatric": r"pediatric|children|childhood|adolescent",
            "geriatric": r"geriatric|elderly|older\s+adults|aged\s+\d+",
            "pregnancy": r"pregnan(?:cy|t)|maternal|prenatal",
            "renal_impairment": r"renal impairment|kidney failure|dialysis|chronic kidney",
            "hepatic_impairment": r"hepatic impairment|cirrhosis|liver failure",
            "genetic_variants": r"polymorphism|genotype|poor metabolizer|ultrarapid metabolizer",
        }

        lowered = content.lower()
        markers = [label for label, pattern in population_patterns.items() if re.search(pattern, lowered)]

        return markers

    def _identify_research_gaps(self, papers: list[dict], query: str) -> list[str]:
        """Identify research gaps based on paper analysis."""
        gaps = []

        try:
            papers = self._ensure_prepared_papers(papers)

            # Analyze study types for gaps
            study_types = [self._classify_study_type(p.get("page_content", "")) for p in papers]
            study_counts = Counter(study_types)

            if study_counts.get("clinical_trial", 0) < 2:
                gaps.append("Limited clinical trial data available")

            if study_counts.get("meta_analysis", 0) == 0:
                gaps.append("No meta-analyses identified")

            # Analyze drug coverage
            if self.pharma_processor:
                all_drugs = set()
                for paper in papers:
                    content = paper.get("page_content", "")
                    drugs = self.pharma_processor.extract_drug_names(content)
                    all_drugs.update(d["name"] for d in drugs if d["confidence"] > 0.8)

                if len(all_drugs) < 3:
                    gaps.append("Limited drug diversity in available studies")

            # Analyze population diversity
            population_keywords = ["pediatric", "elderly", "pregnancy", "renal", "hepatic"]
            population_mentions = 0
            for paper in papers:
                content = paper.get("page_content", "").lower()
                population_mentions += sum(1 for kw in population_keywords if kw in content)

            if population_mentions < len(papers) * 0.3:
                gaps.append("Limited special population data")

            return gaps[:5]  # Limit to top 5 gaps

        except Exception as e:
            logger.error(f"Error identifying research gaps: {e}")
            return ["Error analyzing research gaps"]

    def _calculate_confidence_scores(self, papers: list[dict]) -> dict[str, float]:
        """Calculate overall confidence scores for the synthesis."""
        try:
            papers = self._ensure_prepared_papers(papers)
            scores = {}

            # Sample size confidence
            total_papers = len(papers)
            if total_papers >= 10:
                scores["sample_size"] = 1.0
            elif total_papers >= 5:
                scores["sample_size"] = 0.8
            elif total_papers >= 3:
                scores["sample_size"] = 0.6
            else:
                scores["sample_size"] = 0.4

            # Study quality confidence
            study_types = [self._classify_study_type(p.get("page_content", "")) for p in papers]
            high_quality = sum(1 for st in study_types if st in ["meta_analysis", "clinical_trial"])
            scores["study_quality"] = min(1.0, high_quality / max(1, total_papers))

            # Consistency confidence (placeholder - would need more sophisticated analysis)
            scores["consistency"] = 0.7  # Default moderate confidence

            # Overall confidence
            scores["overall"] = sum(scores.values()) / len(scores)

            return {k: round(v, 2) for k, v in scores.items()}

        except Exception as e:
            logger.error(f"Error calculating confidence scores: {e}")
            return {"overall": 0.5, "error": str(e)}

    def _format_citations(self, papers: list[dict]) -> list[str]:
        """Format citations for the papers."""
        papers = self._ensure_prepared_papers(papers)
        citations = []

        for i, paper in enumerate(papers):
            try:
                metadata = paper.get("metadata", {})

                # Extract citation components with fallbacks
                title = metadata.get("title")
                authors = metadata.get("authors", [])
                journal = metadata.get("journal")
                year = metadata.get("year")
                pmid = metadata.get("pmid")
                doi = metadata.get("doi")

                # Handle missing metadata with fallback templates
                if not title:
                    title = f"Untitled Paper {i+1}"

                # Format author string
                if authors and isinstance(authors, list):
                    if len(authors) > 3:
                        author_str = f"{authors[0]} et al."
                    else:
                        author_str = ", ".join(str(author) for author in authors)
                else:
                    author_str = "Unknown Authors"

                # Format journal
                if not journal:
                    journal = "Unknown Journal"

                # Format year
                if not year:
                    year = "Unknown Year"

                # Build citation with robust fallbacks
                citation_parts = [author_str]

                if title:
                    citation_parts.append(title)

                citation_parts.append(journal)
                citation_parts.append(str(year))

                citation = ". ".join(citation_parts) + "."

                # Add identifier if available
                if pmid:
                    citation += f" PMID: {pmid}."
                elif doi:
                    citation += f" DOI: {doi}."
                else:
                    # Fallback identifier template when both PMID and DOI are missing
                    source_url = metadata.get("source_url") or metadata.get("url")
                    if source_url:
                        citation += f" URL: {source_url}."

                citations.append(citation)

            except Exception as e:
                logger.warning(f"Error formatting citation for paper {i}: {e}")
                citations.append(f"Citation formatting error for paper {i+1}")

        return citations

    # Helper methods

    def _classify_study_type(self, content: str) -> str:
        """Classify the study type based on content."""
        content_lower = content.lower()

        if re.search(r"\b(?:review|systematic review|meta-analysis|narrative review)\b", content_lower):
            if re.search(r"meta-analysis|systematic review|pooled analysis", content_lower):
                return "meta_analysis"
            return "review"

        for study_type, patterns in self.study_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return study_type

        return "review"  # Default fallback

    def _extract_main_finding(self, content: str, metadata: dict) -> str:
        """Extract the main finding from paper content."""
        # Look for conclusion or results sections
        conclusion_patterns = [
            r"conclusion[s]?[:\s]+(.*?)(?:\n\n|\Z)",
            r"results[:\s]+(.*?)(?:\n\n|\Z)",
            r"findings[:\s]+(.*?)(?:\n\n|\Z)",
        ]

        for pattern in conclusion_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                finding = match.group(1).strip()
                # Limit length
                if len(finding) > 200:
                    finding = finding[:200] + "..."
                return finding

        # Fallback to abstract or first sentences
        abstract = metadata.get("abstract", "")
        if abstract:
            sentences = abstract.split(". ")
            return ". ".join(sentences[:2]) + "." if len(sentences) > 1 else abstract

        # Last resort - first few sentences of content
        sentences = content.split(". ")
        return ". ".join(sentences[:2]) + "." if len(sentences) > 1 else "Finding extraction failed"

    def _calculate_finding_confidence(self, content: str, study_type: str, metadata: dict) -> float:
        """Calculate confidence score for a finding."""
        base_confidence = 0.5

        # Study type bonus
        type_bonus = {
            "meta_analysis": 0.4,
            "clinical_trial": 0.3,
            "observational": 0.2,
            "case_report": 0.1,
            "in_vitro": 0.1,
            "review": 0.05,
        }
        base_confidence += type_bonus.get(study_type, 0)

        # Content quality indicators
        if len(content) > 1000:
            base_confidence += 0.1

        # Metadata completeness
        if metadata.get("pmid"):
            base_confidence += 0.05
        if metadata.get("doi"):
            base_confidence += 0.05

        return min(1.0, base_confidence)

    def _extract_outcomes(self, content: str) -> list[str]:
        """Extract outcome measures from content."""
        outcome_patterns = [
            r"primary endpoint",
            r"secondary endpoint",
            r"efficacy",
            r"safety",
            r"adverse events",
            r"mortality",
            r"progression-free survival",
            r"overall survival",
        ]

        outcomes = []
        content_lower = content.lower()
        for pattern in outcome_patterns:
            if pattern in content_lower:
                outcomes.append(pattern.replace("_", " ").title())

        return outcomes

    def _are_findings_convergent(self, findings: list[str]) -> bool:
        """Check if findings are convergent (simplified heuristic)."""
        if len(findings) < 2:
            return False

        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "into",
            "onto",
            "were",
            "was",
            "are",
            "have",
            "has",
            "had",
            "also",
            "while",
            "between",
            "above",
            "below",
            "after",
            "before",
            "their",
            "there",
            "these",
            "those",
            "such",
            "than",
            "within",
            "without",
            "across",
            "about",
            "among",
            "using",
            "based",
        }
        domain_keywords = {
            "increase",
            "decrease",
            "reduction",
            "elevation",
            "improvement",
            "decline",
            "efficacy",
            "safety",
            "toxicity",
            "exposure",
            "clearance",
            "auc",
            "cmax",
            "response",
            "benefit",
        }

        filtered_tokens: list[set[str]] = []
        aggregate_counts: Counter[str] = Counter()

        for finding in findings:
            tokens = re.findall(r"\b[\w-]+\b", finding.lower())
            cleaned = {token for token in tokens if len(token) > 2 and token not in stopwords}
            if cleaned:
                filtered_tokens.append(cleaned)
                aggregate_counts.update(cleaned)

        if len(filtered_tokens) < 2:
            return False

        threshold = max(2, len(filtered_tokens) - 1)
        shared_tokens = {token for token, count in aggregate_counts.items() if count >= threshold}

        if shared_tokens & domain_keywords:
            return True

        # Require at least two overlapping content words when domain keywords are absent
        return len(shared_tokens - domain_keywords) >= 2

    def _synthesize_convergent_finding(self, findings: list[str]) -> str:
        """Synthesize convergent findings into a single statement."""
        if not findings:
            return "No convergent findings identified"

        # Simple approach - take the shortest finding as representative
        return min(findings, key=len)

    def _extract_dose_response_data(self, papers: list[dict], drug: str) -> list[dict]:
        """Extract dose-response data for a specific drug."""
        dose_data = []

        for paper in papers:
            content = paper.get("page_content", "")
            if self.pharma_processor and hasattr(self.pharma_processor, "extract_dosage_information"):
                try:
                    dosages = self.pharma_processor.extract_dosage_information(content)
                except Exception:
                    dosages = []
            else:
                dosages = []

            for dosage in dosages:
                if drug.lower() in content.lower():
                    dose_data.append(
                        {
                            "amount": dosage["amount"],
                            "unit": dosage["unit"],
                            "paper_id": paper.get("metadata", {}).get("pmid", "unknown"),
                        }
                    )

        return dose_data

    def _count_study_types(self, papers: list[dict]) -> dict[str, int]:
        """Count study types across papers."""
        papers = self._ensure_prepared_papers(papers)
        study_types = [self._classify_study_type(p.get("page_content", "")) for p in papers]
        return dict(Counter(study_types))

    def _count_drug_mentions(self, findings: list[KeyFinding]) -> dict[str, int]:
        """Count drug mentions across findings."""
        all_drugs = []
        for finding in findings:
            all_drugs.extend(finding.drug_entities)
        return dict(Counter(all_drugs))

    def _count_evidence_levels(self, findings: list[KeyFinding]) -> dict[str, int]:
        """Count evidence levels across findings."""
        evidence_levels = [f.evidence_level for f in findings]
        return dict(Counter(evidence_levels))

    def _extract_query_terms(self, query: str) -> list[str]:
        """Extract meaningful terms from query for relevance scoring."""
        if not query:
            return []

        # Simple tokenization and filtering
        import re

        tokens = re.findall(r"\b\w+\b", query.lower())

        # Remove common stopwords
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "into",
            "onto",
            "were",
            "was",
            "are",
            "have",
            "has",
            "had",
            "also",
            "while",
            "between",
            "above",
            "below",
            "after",
            "before",
            "their",
            "there",
            "these",
            "those",
            "such",
            "than",
            "within",
            "without",
            "across",
            "about",
            "among",
            "using",
            "based",
            "what",
            "how",
            "when",
            "where",
            "why",
            "which",
            "who",
            "whom",
        }

        meaningful_terms = [token for token in tokens if len(token) > 2 and token not in stopwords]
        return meaningful_terms[:10]  # Limit to top 10 terms

    def _calculate_query_relevance(self, text: str, query_terms: list[str]) -> float:
        """Calculate relevance score based on query term frequency and proximity."""
        if not query_terms or not text:
            return 0.0

        text_lower = text.lower()

        # Term frequency score
        term_matches = 0
        total_terms = len(query_terms)

        for term in query_terms:
            if term in text_lower:
                # Count occurrences and give diminishing returns for multiple mentions
                count = text_lower.count(term)
                term_matches += min(count, 3) / 3.0  # Cap at 3 mentions for scoring

        tf_score = term_matches / total_terms if total_terms > 0 else 0.0

        # Proximity bonus: check if query terms appear close together
        proximity_bonus = 0.0
        if len(query_terms) > 1:
            words = text_lower.split()
            word_positions = {}

            for i, word in enumerate(words):
                for term in query_terms:
                    if term in word:
                        if term not in word_positions:
                            word_positions[term] = []
                        word_positions[term].append(i)

            # Calculate proximity for terms that appear
            if len(word_positions) > 1:
                positions = []
                for term_positions in word_positions.values():
                    positions.extend(term_positions)

                if len(positions) > 1:
                    positions.sort()
                    # Bonus if terms appear within 20 words of each other
                    min_distance = min(positions[i + 1] - positions[i] for i in range(len(positions) - 1))
                    if min_distance <= 20:
                        proximity_bonus = 0.2 * (20 - min_distance) / 20

        return min(1.0, tf_score + proximity_bonus)


__all__ = ["SynthesisEngine", "KeyFinding", "BulletPoint", "ComparativeAnalysis", "MetaSummary"]
