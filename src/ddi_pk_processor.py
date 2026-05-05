"""Drug-drug interaction and pharmacokinetic analysis processor for pharmaceutical research."""
from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
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
except ImportError:  # pragma: no cover - support direct module execution
    from paper_schema import Paper  # type: ignore

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

logger = logging.getLogger(__name__)


@dataclass
class PKParameter:
    """Structured pharmacokinetic parameter data with provenance tracking."""

    parameter: str
    value: float | None
    unit: str
    confidence_interval: tuple[float, float] | None
    p_value: float | None
    study_id: str
    value_numeric: float | None = None
    unit_canonical: str | None = None
    original_text: str | None = None
    # Provenance tracking fields
    original_value: float | None = None
    original_unit: str | None = None
    conversion_factor: float | None = None
    normalization_notes: str | None = None


@dataclass
class DrugInteraction:
    """Structured drug interaction data."""

    primary_drug: str
    secondary_drug: str
    interaction_type: str
    severity: str
    mechanism: str
    clinical_effect: str
    evidence_level: str
    study_id: str


class InteractionReport(BaseModel):
    """Structured representation of a comprehensive drug interaction analysis report."""

    primary_drug: str = Field(..., description="Primary drug being analyzed")
    analyzed_papers: int = Field(..., description="Number of papers analyzed")
    secondary_drugs_analyzed: list[str] = Field(
        default_factory=list, description="Secondary drugs analyzed for interactions"
    )
    pk_parameters: dict[str, Any] = Field(default_factory=dict, description="Pharmacokinetic parameters extracted")
    pk_parameter_source: str = Field(default="text_parsing", description="Source of PK parameter data")
    cyp_interactions: dict[str, Any] = Field(default_factory=dict, description="CYP enzyme interaction analysis")
    auc_cmax_changes: list[dict[str, Any]] = Field(default_factory=list, description="AUC/Cmax changes identified")
    pk_pd_summary: dict[str, Any] = Field(default_factory=dict, description="Pharmacokinetic/pharmacodynamic summary")
    clinical_recommendations: list[dict[str, Any]] = Field(
        default_factory=list, description="Clinical recommendations (research-only)"
    )
    interaction_severities: dict[str, str] = Field(default_factory=dict, description="Interaction severity assessments")
    formatted_report: str = Field(default="", description="Formatted text report")
    summary_statistics: dict[str, Any] = Field(default_factory=dict, description="Summary statistics")
    error: str | None = Field(default=None, description="Error message if analysis failed")


DEFAULT_DDI_CONFIG: dict[str, Any] = {
    "severity_thresholds": {
        "increase": {"major": 5.0, "moderate": 2.0, "minor": 1.25},
        "decrease": {"major": 0.2, "moderate": 0.5, "minor": 0.8},
        "clearance_decrease": {"major": 0.2, "moderate": 0.5},
    },
    "keyword_groups": {
        "textual": {
            "contraindicated": [
                "contraindicated",
                "contraindication",
                "should not",
                "must not",
                "avoid",
                "prohibited",
                "black box",
                "severe",
            ],
            "major": [
                "major",
                "significant",
                "serious",
                "important",
                "clinically significant",
                "substantial",
                "pronounced",
                "marked",
            ],
            "moderate": ["moderate", "modest", "noticeable", "measurable", "clinically relevant"],
            "minor": ["minor", "slight", "small", "minimal", "negligible", "weak"],
        },
        "mechanistic": {
            "major": [
                "strong inhibitor",
                "potent inhibitor",
                "strong inducer",
                "marked inhibition",
                "pronounced induction",
            ],
            "moderate": ["moderate inhibitor", "moderate inducer", "significant inhibitor"],
        },
    },
}


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating the originals."""
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


class DDIPKProcessor:
    """Analyze drug-drug interactions and pharmacokinetic parameters from research papers.

    Provides comprehensive analysis of drug interactions, CYP enzyme effects,
    AUC/Cmax changes, and clinical recommendations based on pharmaceutical literature.

    Example usage:
        processor = DDIPKProcessor()
        analysis = processor.analyze_drug_interactions(papers, "warfarin", ["fluconazole"])
    """

    def __init__(
        self,
        pharma_processor: PharmaceuticalProcessor | None = None,
        config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
    ):
        """Initialize DDI/PK processor with optional pharmaceutical processor and configuration.

        Args:
            pharma_processor: Optional PharmaceuticalProcessor for entity extraction.
                             If None, will attempt to create one if available.
            config: Optional override dictionary for severity thresholds and keyword groups.
            config_path: Path to a JSON or YAML configuration file that augments defaults.
        """
        self.pharma_processor = pharma_processor
        if self.pharma_processor is None and PharmaceuticalProcessor is not None:
            try:
                self.pharma_processor = PharmaceuticalProcessor()
            except Exception as e:
                logger.warning(f"Failed to initialize PharmaceuticalProcessor: {e}")

        self.config = self._build_config(config=config, config_path=config_path)
        keyword_groups = self.config.get("keyword_groups", {})
        self.severity_keywords = keyword_groups.get("textual", {})
        self.mechanistic_keywords = keyword_groups.get("mechanistic", {})
        self.severity_thresholds = self.config.get("severity_thresholds", {})

        # CYP enzyme patterns and classifications
        self.cyp_patterns = {
            "substrate": [
                r"substrate.*cyp",
                r"metabolized.*by.*cyp",
                r"cyp.*substrate",
                r"primarily.*metabolized.*cyp",
                r"cyp.*mediated.*metabolism",
            ],
            "inhibitor": [
                r"inhibit.*cyp",
                r"cyp.*inhibitor",
                r"inhibition.*cyp",
                r"cyp.*inhibition",
                r"blocks.*cyp",
                r"cyp.*blocking",
            ],
            "inducer": [
                r"(induce|induces|inducing|induction).*cyp",
                r"cyp.*(induce|induces|inducing|induction)",
                r"(upregulate|upregulates|upregulating|upregulation).*cyp",
                r"cyp.*(upregulate|upregulates|upregulating|upregulation)",
                r"(increase|increases|increasing).*cyp.*activity",
            ],
        }

        # PK parameter patterns with units and statistical indicators
        self.pk_parameter_patterns = {
            "auc": {
                "patterns": [
                    r"auc.*?(\d+(?:\.\d+)?)\s*([%]?)\s*(increase|decrease|chang|fold)",
                    r"area.*under.*curve.*?(\d+(?:\.\d+)?)\s*([%]?)",
                    r"auc.*ratio.*?(\d+(?:\.\d+)?)",
                ],
                "units": ["ng⋅h/mL", "µg⋅h/mL", "mg⋅h/L", "fold", "%"],
            },
            "cmax": {
                "patterns": [
                    r"cmax.*?(\d+(?:\.\d+)?)\s*([%]?)\s*(increase|decrease|chang|fold)",
                    r"maximum.*concentration.*?(\d+(?:\.\d+)?)\s*([%]?)",
                    r"peak.*concentration.*?(\d+(?:\.\d+)?)\s*([%]?)",
                ],
                "units": ["ng/mL", "µg/mL", "mg/L", "fold", "%"],
            },
            "clearance": {
                "patterns": [
                    r"clearance.*?(\d+(?:\.\d+)?)\s*([%]?)\s*(increase|decrease|chang)",
                    r"cl/f.*?(\d+(?:\.\d+)?)\s*([%]?)",
                ],
                "units": ["L/h", "mL/min", "L/h/kg", "%"],
            },
            "half_life": {
                "patterns": [
                    r"half.*life.*?(\d+(?:\.\d+)?)\s*(h|hr|hours?|min|minutes?)",
                    r"t1/2.*?(\d+(?:\.\d+)?)\s*(h|hr|hours?|min|minutes?)",
                ],
                "units": ["h", "hr", "hours", "min", "minutes"],
            },
            "volume_distribution": {
                "patterns": [
                    r"volume.*distribution.*?(\d+(?:\.\d+)?)\s*(L|L/kg)",
                    r"\bvd\b.*?(\d+(?:\.\d+)?)\s*(L|L/kg)",
                    r"\bv_d\b.*?(\d+(?:\.\d+)?)\s*(L|L/kg)",
                ],
                "units": ["L", "L/kg"],
            },
        }

        # Clinical recommendation patterns
        self.recommendation_patterns = [
            r"recommend.*?(monitor|avoid|adjust|reduce|reducing|reduction|increase|increasing|elevation|caution)",
            r"suggest.*?(monitor|avoid|adjust|reduce|reducing|reduction|increase|increasing|elevation|caution)",
            r"advise.*?(monitor|avoid|adjust|reduce|reducing|reduction|increase|increasing|elevation|caution)",
            r"clinical.*?(monitor|avoid|adjust|reduce|reducing|reduction|increase|increasing|elevation|caution)",
            r"should.*?(monitor|avoid|adjust|reduce|reducing|reduction|increase|increasing|elevation|caution)",
        ]

    def _build_config(
        self,
        config: dict[str, Any] | None,
        config_path: str | Path | None,
    ) -> dict[str, Any]:
        """Merge default configuration with optional overrides."""
        merged = copy.deepcopy(DEFAULT_DDI_CONFIG)

        if config_path:
            try:
                file_config = self._load_config_file(Path(config_path))
                if file_config:
                    merged = _deep_merge_dicts(merged, file_config)
            except Exception as exc:
                logger.warning("Failed to load DDI/PK processor config from %s: %s", config_path, exc)

        if config:
            merged = _deep_merge_dicts(merged, config)

        return merged

    def _load_config_file(self, path: Path) -> dict[str, Any]:
        """Load configuration overrides from a JSON or YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        suffix = path.suffix.lower()
        with path.open("r", encoding="utf-8") as handle:
            if suffix == ".json":
                return json.load(handle)

            if suffix in {".yml", ".yaml"}:
                if yaml is None:
                    raise ImportError("PyYAML is required to load YAML configuration files.")
                return yaml.safe_load(handle) or {}

        raise ValueError(f"Unsupported configuration file format: {path.suffix}")

    def _prepare_papers(self, papers: list[Any]) -> list[dict[str, Any]]:
        """Validate paper payloads using the shared Paper schema."""
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
        """Return schema-prepared papers, coercing raw inputs when necessary."""
        if not papers:
            return []
        if all(isinstance(paper, dict) and paper.get("__paper_schema_validated__") for paper in papers):
            return papers
        return self._prepare_papers(papers)

    def _normalize_pk_measurement(
        self, parameter: str, value: float | None, unit: str | None
    ) -> tuple[float | None, str | None, float | None, str | None]:
        """Normalize PK units to canonical representations with provenance tracking.

        Returns:
            Tuple of (normalized_value, canonical_unit, conversion_factor, normalization_notes)
        """
        # Store original values for provenance

        if value is None:
            return None, unit, None, None

        if not unit:
            return value, None, None, "No unit provided"

        unit_clean = unit.strip().replace("−", "-")
        unit_key = (
            unit_clean.replace("μ", "u").replace("µ", "u").replace("⋅", "").replace("·", "").replace(" ", "").lower()
        )

        unit_key = re.sub(r"ml\^-?1", "perml", unit_key)
        unit_key = re.sub(r"l\^-?1", "perl", unit_key)
        unit_key = re.sub(r"h\^-?1", "perh", unit_key)
        unit_key = re.sub(r"ml-?1", "perml", unit_key)
        unit_key = re.sub(r"l-?1", "perl", unit_key)
        unit_key = re.sub(r"h-?1", "perh", unit_key)

        normalized_value = value
        canonical_unit: str | None = None
        conversion_factor: float | None = None
        normalization_notes: str | None = None

        if "geometricmeanratio" in unit_key or unit_key == "gmr":
            canonical_unit = "fold"
            conversion_factor = 1.0
            normalization_notes = "Geometric mean ratio converted to fold representation"

        if parameter == "cmax":
            if unit_key in {"ng/ml", "ngperml"}:
                canonical_unit = "ng/mL"
                conversion_factor = 1.0
                normalization_notes = "Already in canonical ng/mL unit"
            elif unit_key in {"ug/ml", "ugperml"}:
                canonical_unit = "ng/mL"
                conversion_factor = 1000.0
                normalized_value = value * 1000
                normalization_notes = f"Converted from µg/mL to ng/mL (×{conversion_factor})"
            elif unit_key in {"mg/l", "mgperl"}:
                canonical_unit = "ng/mL"
                conversion_factor = 1000.0
                normalized_value = value * 1000
                normalization_notes = f"Converted from mg/L to ng/mL (×{conversion_factor})"

        elif parameter == "auc":
            if "fold" in unit_key:
                canonical_unit = "fold"
                conversion_factor = 1.0
                normalization_notes = "Fold change value, no conversion needed"
            elif "%" in unit_key or "percent" in unit_key:
                canonical_unit = "fraction"
                conversion_factor = 0.01
                normalized_value = value / 100
                normalization_notes = f"Converted from percentage to fraction (×{conversion_factor})"
            elif "ng" in unit_key and ("h/ml" in unit_key or "hperml" in unit_key or "hml" in unit_key):
                canonical_unit = "ng*h/mL"
                conversion_factor = 1.0
                normalization_notes = "Already in canonical ng*h/mL unit"
            elif "ug" in unit_key and ("h/ml" in unit_key or "hperml" in unit_key or "hml" in unit_key):
                canonical_unit = "ng*h/mL"
                conversion_factor = 1000.0
                normalized_value = value * 1000
                normalization_notes = f"Converted from µg*h/mL to ng*h/mL (×{conversion_factor})"
            elif "mg" in unit_key and ("h/l" in unit_key or "hperl" in unit_key or "hl" in unit_key):
                canonical_unit = "ng*h/mL"
                conversion_factor = 1000.0
                normalized_value = value * 1000
                normalization_notes = f"Converted from mg*h/L to ng*h/mL (×{conversion_factor})"

        elif parameter == "half_life":
            if unit_key in {"h", "hr", "hrs", "hour", "hours"}:
                canonical_unit = "hours"
                conversion_factor = 1.0
                normalization_notes = "Already in canonical hours unit"
            elif unit_key in {"min", "mins", "minute", "minutes"}:
                canonical_unit = "hours"
                conversion_factor = 1 / 60.0
                normalized_value = value / 60
                normalization_notes = f"Converted from minutes to hours (×{conversion_factor})"

        elif parameter == "clearance":
            if unit_key in {"l/h", "lperh"}:
                canonical_unit = "L/h"
                conversion_factor = 1.0
                normalization_notes = "Already in canonical L/h unit"
            elif unit_key in {"ml/min", "mlpermin"}:
                canonical_unit = "L/h"
                conversion_factor = 0.06
                normalized_value = value * 0.06
                normalization_notes = f"Converted from mL/min to L/h (×{conversion_factor})"
            elif unit_key in {"l/h/kg", "lperhkg"}:
                canonical_unit = "L/h/kg"
                conversion_factor = 1.0
                normalization_notes = "Body weight normalized clearance, no conversion needed"
            elif "%" in unit_key or "percent" in unit_key:
                canonical_unit = "fraction"
                conversion_factor = 0.01
                normalized_value = value / 100
                normalization_notes = f"Converted from percentage to fraction (×{conversion_factor})"

        if canonical_unit is None:
            canonical_unit = unit_clean or None
            normalized_value = value
            conversion_factor = 1.0
            normalization_notes = "No normalization applied, unit preserved as-is"

        return normalized_value, canonical_unit, conversion_factor, normalization_notes

    def _build_drug_synonym_map(self, papers: list[dict]) -> dict[str, str]:
        """Build a map of drug names to their canonical generic names.

        Returns:
            Dict mapping lowercase drug names to their canonical generic names
        """
        canonical_for: dict[str, str] = {}

        # First, try to use drug annotations from papers if available
        for paper in papers:
            metadata = paper.get("metadata", {})
            drug_annotations = metadata.get("drug_annotations", [])
            if drug_annotations:
                for annotation in drug_annotations:
                    name = annotation.get("name", "").lower()
                    drug_type = annotation.get("type", "")
                    if name and drug_type in ("generic", "brand"):
                        # Prefer generic names as canonical
                        if drug_type == "generic":
                            canonical_for[name] = name
                        # For brand names, map to their generic equivalent if we can determine it
                        # This is a simplified approach - in a real implementation, we'd have a proper mapping
                        elif drug_type == "brand":
                            # Check if we have a known mapping
                            brand_to_generic = {
                                "coumadin": "warfarin",
                                "lipitor": "atorvastatin",
                                "plavix": "clopidogrel",
                                "nexium": "esomeprazole",  # Note: esomeprazole is the generic for nexium
                                "prilosec": "omeprazole",
                                "xarelto": "rivaroxaban",
                                "zyrtec": "cetirizine",
                            }
                            generic_name = brand_to_generic.get(name)
                            if generic_name:
                                canonical_for[name] = generic_name
                            # If no known mapping, use the brand name itself
                            else:
                                canonical_for[name] = name

        # If no annotations available, use the pharmaceutical processor to extract drug names
        if not canonical_for and self.pharma_processor:
            # Concatenate content from all papers to extract drug names
            all_content = ""
            for paper in papers:
                content = paper.get("page_content") or paper.get("content", "")
                all_content += content + " "

            if all_content:
                drug_entities = self.pharma_processor.extract_drug_names(all_content)
                for entity in drug_entities:
                    name = entity.get("name", "").lower()
                    drug_type = entity.get("type", "")
                    if name and drug_type in ("generic", "brand"):
                        # Prefer generic names as canonical
                        if drug_type == "generic":
                            canonical_for[name] = name
                        # For brand names, map to their generic equivalent if we can determine it
                        elif drug_type == "brand":
                            # Check if we have a known mapping
                            brand_to_generic = {
                                "coumadin": "warfarin",
                                "lipitor": "atorvastatin",
                                "plavix": "clopidogrel",
                                "nexium": "esomeprazole",  # Note: esomeprazole is the generic for nexium
                                "prilosec": "omeprazole",
                                "xarelto": "rivaroxaban",
                                "zyrtec": "cetirizine",
                            }
                            generic_name = brand_to_generic.get(name)
                            if generic_name:
                                canonical_for[name] = generic_name
                            # If no known mapping, use the brand name itself
                            else:
                                canonical_for[name] = name

        # Add default mappings for known brand/generic pairs
        default_mappings = {
            "coumadin": "warfarin",
            "lipitor": "atorvastatin",
            "plavix": "clopidogrel",
            "nexium": "esomeprazole",
            "prilosec": "omeprazole",
            "xarelto": "rivaroxaban",
            "zyrtec": "cetirizine",
        }

        for brand, generic in default_mappings.items():
            if brand not in canonical_for:
                canonical_for[brand] = generic
            if generic not in canonical_for:
                canonical_for[generic] = generic

        return canonical_for

    def analyze_drug_interactions(
        self, papers: list[dict], primary_drug: str, secondary_drugs: list[str] | None = None
    ) -> dict[str, Any]:
        """Analyze drug interactions focusing on a primary drug with potential interacting drugs.

        Args:
            papers: List of research papers with content and metadata
            primary_drug: Primary drug of interest
            secondary_drugs: Optional list of specific secondary drugs to analyze

        Returns:
            Comprehensive interaction analysis dictionary
        """
        prepared_papers: list[dict[str, Any]] = []

        try:
            prepared_papers = self._prepare_papers(papers)
            logger.info(
                "Analyzing drug interactions for %s across %s schema-validated papers",
                primary_drug,
                len(prepared_papers),
            )

            # Build synonym map for canonicalization
            canonical_for = self._build_drug_synonym_map(prepared_papers)

            # Normalize inputs to canonical forms
            primary_drug_canon = canonical_for.get(primary_drug.lower(), primary_drug.lower())
            secondary_drugs_canon = [canonical_for.get(d.lower(), d.lower()) for d in (secondary_drugs or [])]

            # Create display names mapping for reporting
            display_names = {}
            for name in [primary_drug] + (secondary_drugs or []):
                canon = canonical_for.get(name.lower(), name.lower())
                if canon not in display_names or len(name) < len(display_names[canon]):
                    display_names[canon] = name  # Prefer shorter names for display

            # Extract PK parameters for primary drug
            pk_parameters = self._extract_pk_parameters(prepared_papers, primary_drug_canon)

            # Analyze CYP interactions
            cyp_interactions = self._analyze_cyp_interactions(prepared_papers, primary_drug_canon)

            # Extract AUC/Cmax changes
            auc_cmax_changes = self._extract_auc_cmax_changes(
                prepared_papers, primary_drug_canon, secondary_drugs_canon
            )

            # Generate PK/PD summary
            pk_pd_summary = self._generate_pk_pd_summary(prepared_papers, primary_drug_canon)

            # Identify clinical recommendations
            clinical_recommendations = self._identify_clinical_recommendations(prepared_papers, "all")

            # Calculate interaction severity for each identified interaction
            interaction_severities = {}
            if secondary_drugs_canon:
                for secondary_drug_canon in secondary_drugs_canon:
                    severity = self._calculate_interaction_severity(
                        {
                            "primary_drug": primary_drug_canon,
                            "secondary_drug": secondary_drug_canon,
                            "papers": prepared_papers,
                        }
                    )
                    # Map back to original display name for reporting
                    display_name = display_names.get(secondary_drug_canon, secondary_drug_canon)
                    interaction_severities[display_name] = severity

            # Compute significance filtering statistics
            sig_count = sum(1 for c in auc_cmax_changes if c.get("is_significant"))
            non_sig_count = sum(1 for c in auc_cmax_changes if c.get("is_significant") is False)

            # Ensure all data is JSON-serializable before returning while preserving structure
            def _serialize_pk_entry(entry: Any) -> Any:
                if hasattr(entry, "__dataclass_fields__"):
                    return asdict(entry)
                if isinstance(entry, dict):
                    return {key: _serialize_pk_entry(value) for key, value in entry.items()}
                if isinstance(entry, list):
                    return [_serialize_pk_entry(value) for value in entry]
                return entry

            def _serialize_pk_parameter_map(pk_map: dict[str, Any]) -> dict[str, Any]:
                serialized: dict[str, Any] = {}
                for param_name, param_data in pk_map.items():
                    if isinstance(param_data, dict):
                        param_serialized = {}
                        for key, value in param_data.items():
                            if key == "parameters" and isinstance(value, list):
                                param_serialized[key] = [_serialize_pk_entry(item) for item in value]
                            else:
                                param_serialized[key] = _serialize_pk_entry(value)
                        serialized[param_name] = param_serialized
                    else:
                        serialized[param_name] = _serialize_pk_entry(param_data)
                return serialized

            pk_parameters = _serialize_pk_parameter_map(pk_parameters)
            auc_cmax_changes = [_serialize_pk_entry(x) for x in auc_cmax_changes]

            # Format comprehensive interaction report
            # Map secondary drugs back to display names
            secondary_drugs_display = [display_names.get(d, d) for d in secondary_drugs_canon]

            interaction_report = self._format_interaction_report(
                {
                    "primary_drug": display_names.get(primary_drug_canon, primary_drug_canon),
                    "secondary_drugs": secondary_drugs_display,
                    "pk_parameters": pk_parameters,
                    "pk_parameter_source": "text_parsing",
                    "cyp_interactions": cyp_interactions,
                    "auc_cmax_changes": auc_cmax_changes,
                    "pk_pd_summary": pk_pd_summary,
                    "clinical_recommendations": clinical_recommendations,
                    "interaction_severities": interaction_severities,
                }
            )

            analysis_result = {
                "primary_drug": display_names.get(primary_drug_canon, primary_drug_canon),
                "analyzed_papers": len(prepared_papers),
                "secondary_drugs_analyzed": secondary_drugs_display,
                "pk_parameters": pk_parameters,
                "pk_parameter_source": "text_parsing",
                "cyp_interactions": cyp_interactions,
                "auc_cmax_changes": auc_cmax_changes,
                "pk_pd_summary": pk_pd_summary,
                "clinical_recommendations": clinical_recommendations,
                "interaction_severities": interaction_severities,
                "formatted_report": interaction_report,
                "summary_statistics": {
                    "total_interactions_found": len(auc_cmax_changes),
                    "significant_pk_changes": sig_count,
                    "nonsignificant_pk_changes": non_sig_count,
                    "cyp_enzymes_involved": len(cyp_interactions.get("enzymes_identified", [])),
                    "high_severity_interactions": sum(
                        1 for s in interaction_severities.values() if s == "major" or s == "contraindicated"
                    ),
                    "pk_parameters_quantified": len([p for p in pk_parameters.values() if p]),
                },
            }

            # Validate the analysis result against the InteractionReport schema
            try:
                interaction_report_model = InteractionReport.model_validate(analysis_result)
                logger.info("Drug interaction analysis completed successfully with schema validation")
                # Return the validated model as a dictionary, pruning null fields for clarity
                return interaction_report_model.model_dump(exclude_none=True)
            except ValidationError as schema_error:
                # Log schema validation errors but continue with unvalidated result
                logger.error(f"DDI/PK report schema validation failed: {schema_error}")
                logger.info("Returning unvalidated analysis result due to schema errors")
                # Add schema validation status to the result
                analysis_result["schema_validation"] = {
                    "validated": False,
                    "errors": [
                        {"field": err["loc"], "message": err["msg"], "type": err["type"]}
                        for err in schema_error.errors()
                    ],
                    "note": "Analysis completed but failed schema validation - review field compatibility",
                }
                return analysis_result

        except Exception as e:
            logger.error(f"Error in drug interaction analysis: {e}")
            return {
                "error": str(e),
                "primary_drug": primary_drug,
                "analyzed_papers": len(prepared_papers) if prepared_papers else (len(papers) if papers else 0),
                "pk_parameters": {},
                "cyp_interactions": {},
                "auc_cmax_changes": [],
                "clinical_recommendations": [],
            }

    def _extract_pk_parameters(self, papers: list[dict], drug_name: str) -> dict[str, Any]:
        """Extract pharmacokinetic parameters with units and confidence intervals."""
        papers = self._ensure_prepared_papers(papers)
        pk_data = {"auc": [], "cmax": [], "clearance": [], "half_life": [], "volume_distribution": []}

        try:
            for paper in papers:
                content = paper.get("page_content") or paper.get("content", "")
                paper_id = paper.get("metadata", {}).get("pmid") or paper.get("id", "unknown")

                # Check if drug is mentioned in this paper
                if drug_name.lower() not in content.lower():
                    continue

                for param_name, param_info in self.pk_parameter_patterns.items():
                    for pattern in param_info["patterns"]:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            try:
                                value = float(match.group(1))
                                unit = match.group(2) if len(match.groups()) > 1 else ""
                                raw_unit = unit or param_info["units"][0]
                                original_text = match.group(0).strip()

                                (
                                    value_numeric,
                                    unit_canonical,
                                    conversion_factor,
                                    normalization_notes,
                                ) = self._normalize_pk_measurement(param_name, value, raw_unit)

                                # Extract confidence intervals if present
                                ci_pattern = r"95%.*?ci.*?(\d+(?:\.\d+)?)[^\d]*(\d+(?:\.\d+)?)"
                                ci_match = re.search(
                                    ci_pattern,
                                    content[max(0, match.start() - 100) : match.end() + 100],
                                    re.IGNORECASE,
                                )
                                confidence_interval = None
                                if ci_match:
                                    confidence_interval = (float(ci_match.group(1)), float(ci_match.group(2)))

                                # Extract p-value if present
                                p_pattern = r"p\s*[=<>]\s*(\d+(?:\.\d+)?)"
                                p_match = re.search(
                                    p_pattern,
                                    content[max(0, match.start() - 50) : match.end() + 50],
                                    re.IGNORECASE,
                                )
                                p_value = float(p_match.group(1)) if p_match else None

                                pk_parameter = PKParameter(
                                    parameter=param_name,
                                    value=value,
                                    unit=raw_unit,
                                    confidence_interval=confidence_interval,
                                    p_value=p_value,
                                    study_id=paper_id,
                                    value_numeric=value_numeric,
                                    unit_canonical=unit_canonical,
                                    original_text=original_text,
                                    # Provenance tracking
                                    original_value=value,
                                    original_unit=raw_unit,
                                    conversion_factor=conversion_factor,
                                    normalization_notes=normalization_notes,
                                )
                                pk_data[param_name].append(pk_parameter)

                            except (ValueError, IndexError) as exc:
                                logger.debug(f"Error parsing PK parameter: {exc}")
                                continue

            # Summarize extracted parameters
            summary = {}
            for param_name, parameters in pk_data.items():
                if not parameters:
                    continue

                raw_values = [p.value for p in parameters if p.value is not None]
                normalized_values = [p.value_numeric for p in parameters if p.value_numeric is not None]

                mean_normalized = sum(normalized_values) / len(normalized_values) if normalized_values else None
                mean_raw = sum(raw_values) / len(raw_values) if raw_values else None

                def _build_range(values: list[float]) -> list[float]:
                    if not values:
                        return []
                    if len(values) == 1:
                        return [values[0]]
                    return [min(values), max(values)]

                summary[param_name] = {
                    "count": len(parameters),
                    "mean_value": mean_normalized if mean_normalized is not None else mean_raw,
                    "mean_value_raw": mean_raw,
                    "normalized_range": _build_range(normalized_values),
                    "range_raw": _build_range(raw_values),
                    "units": sorted({p.unit for p in parameters if p.unit}),
                    "normalized_units": sorted({p.unit_canonical for p in parameters if p.unit_canonical}),
                    "studies": sorted({p.study_id for p in parameters}),
                    "parameters": [
                        {
                            "value": p.value,
                            "unit": p.unit,
                            "value_normalized": p.value_numeric,
                            "unit_canonical": p.unit_canonical,
                            "confidence_interval": p.confidence_interval,
                            "p_value": p.p_value,
                            "study_id": p.study_id,
                            "extraction_method": "text_parsing",
                            "original_text": p.original_text,
                            # Provenance information
                            "original_value": p.original_value,
                            "original_unit": p.original_unit,
                            "conversion_factor": p.conversion_factor,
                            "normalization_notes": p.normalization_notes,
                        }
                        for p in parameters
                    ],
                }

            return summary

        except Exception as e:
            logger.error(f"Error extracting PK parameters: {e}")
            return {}

    def _analyze_cyp_interactions(self, papers: list[dict], drug_name: str) -> dict[str, Any]:
        """Analyze CYP enzyme interactions (substrate/inhibitor/inducer classification)."""
        papers = self._ensure_prepared_papers(papers)
        cyp_analysis = {
            "enzymes_identified": [],
            "substrate_relationships": [],
            "inhibitor_relationships": [],
            "inducer_relationships": [],
            "strength_classifications": {},
        }

        try:
            for paper in papers:
                content = paper.get("page_content") or paper.get("content", "")
                paper_id = paper.get("metadata", {}).get("pmid") or paper.get("id", "unknown")

                # Check if drug is mentioned
                if drug_name.lower() not in content.lower():
                    continue

                # Extract CYP enzymes mentioned
                if self.pharma_processor:
                    cyp_enzymes = self.pharma_processor.extract_cyp_enzymes(content)
                else:
                    cyp_pattern = re.compile(r"cyp\s*-?\d[a-z0-9]*", re.IGNORECASE)
                    cyp_enzymes = [match.group().upper().replace(" ", "") for match in cyp_pattern.finditer(content)]

                cyp_analysis["enzymes_identified"].extend(cyp_enzymes)

                # Analyze relationship types
                for relationship_type, patterns in self.cyp_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            for enzyme in cyp_enzymes:
                                relationship_data = {
                                    "drug": drug_name,
                                    "enzyme": enzyme,
                                    "relationship": relationship_type,
                                    "study_id": paper_id,
                                }

                                # Determine strength (strong/moderate/weak)
                                strength = self._determine_interaction_strength(content, drug_name, enzyme)
                                relationship_data["strength"] = strength

                                cyp_analysis[f"{relationship_type}_relationships"].append(relationship_data)

            # Remove duplicates and summarize
            cyp_analysis["enzymes_identified"] = list(set(cyp_analysis["enzymes_identified"]))

            # Create strength classifications summary
            for relationship_type in ["substrate", "inhibitor", "inducer"]:
                relationships = cyp_analysis[f"{relationship_type}_relationships"]
                for rel in relationships:
                    enzyme = rel["enzyme"]
                    if enzyme not in cyp_analysis["strength_classifications"]:
                        cyp_analysis["strength_classifications"][enzyme] = {}
                    cyp_analysis["strength_classifications"][enzyme][relationship_type] = rel["strength"]

            return cyp_analysis

        except Exception as e:
            logger.error(f"Error analyzing CYP interactions: {e}")
            return {"error": str(e)}

    def _calculate_interaction_severity(self, interaction_data: dict[str, Any]) -> str:
        """Calculate interaction severity based on available data."""
        try:
            primary_drug = interaction_data["primary_drug"]
            secondary_drug = interaction_data["secondary_drug"]
            papers = self._ensure_prepared_papers(interaction_data["papers"])

            primary_lower = primary_drug.lower()
            secondary_lower = secondary_drug.lower()

            # Quantitative assessment using PK changes (AUC, Cmax, clearance)
            quantitative_changes = self._extract_auc_cmax_changes(papers, primary_drug, [secondary_drug])
            quantitative_changes = [
                change
                for change in quantitative_changes
                if change.get("interacting_drug") in (None, secondary_drug)
                and secondary_lower in change.get("context", "").lower()
                and change.get("direction") not in (None, "unknown")
            ]

            severity_rank = {"contraindicated": 4, "major": 3, "moderate": 2, "minor": 1}

            thresholds_increase = self.severity_thresholds.get("increase", {})
            thresholds_decrease = self.severity_thresholds.get("decrease", {})
            clearance_thresholds = self.severity_thresholds.get("clearance_decrease", {})

            # Use config defaults with fallback to original hardcoded values
            default_thresholds = {
                "increase": {"major": 5.0, "moderate": 2.0, "minor": 1.25},
                "decrease": {"major": 0.2, "moderate": 0.5, "minor": 0.8},
                "clearance_decrease": {"major": 0.2, "moderate": 0.5},
            }

            # Ensure all required thresholds have values
            for key, defaults in default_thresholds.items():
                if key == "increase":
                    current = thresholds_increase
                elif key == "decrease":
                    current = thresholds_decrease
                elif key == "clearance_decrease":
                    current = clearance_thresholds
                else:
                    continue

                for level, default_val in defaults.items():
                    if level not in current or current[level] is None:
                        current[level] = default_val

            def upgrade_severity(current: str | None, candidate: str | None) -> str | None:
                if not candidate:
                    return current
                if current is None:
                    return candidate
                return candidate if severity_rank.get(candidate, 0) > severity_rank.get(current, 0) else current

            severity_from_quant: str | None = None
            for change in quantitative_changes:
                direction = change.get("direction")
                parameter = (change.get("parameter") or "").lower()
                ratio = change.get("fold_change_ratio")
                unit = (change.get("unit") or "").lower()
                change_value = change.get("change_value")

                if ratio is None and change_value is not None:
                    if unit == "%":
                        delta = change_value / 100.0
                        if direction == "increase":
                            ratio = 1 + delta
                        elif direction == "decrease":
                            ratio = max(0.0, 1 - delta)
                    elif "fold" in unit:
                        if direction == "increase":
                            ratio = change_value
                        elif change_value and direction == "decrease":
                            ratio = (
                                change_value
                                if 0 < change_value < 1
                                else (1 / change_value if change_value > 0 else None)
                            )

                if not ratio or ratio <= 0:
                    continue

                candidate = None
                if direction == "increase":
                    if ratio >= thresholds_increase.get("major", 5.0):
                        candidate = "major"
                    elif ratio >= thresholds_increase.get("moderate", 2.0):
                        candidate = "moderate"
                    elif ratio >= thresholds_increase.get("minor", 1.25):
                        candidate = "minor"
                elif direction == "decrease":
                    if ratio <= thresholds_decrease.get("major", 0.2):
                        candidate = "major"
                    elif ratio <= thresholds_decrease.get("moderate", 0.5):
                        candidate = "moderate"
                    elif ratio <= thresholds_decrease.get("minor", 0.8):
                        candidate = "minor"

                if parameter == "clearance" and direction == "decrease":
                    clearance_major = clearance_thresholds.get("major", 0.2)
                    clearance_moderate = clearance_thresholds.get("moderate", 0.5)
                    if ratio <= clearance_major:
                        candidate = "major"
                    elif ratio <= clearance_moderate and candidate not in ("major", "moderate"):
                        candidate = "moderate"

                severity_from_quant = upgrade_severity(severity_from_quant, candidate)
                if severity_from_quant == "major":
                    break

            # Mechanistic assessment (strong inhibitor/inducer escalates severity)
            mechanistic_severity = None
            mechanistic_keywords_major = self.mechanistic_keywords.get("major", [])
            mechanistic_keywords_moderate = self.mechanistic_keywords.get("moderate", [])

            textual_severity: set[str] = set()

            for paper in papers:
                content = paper.get("page_content") or paper.get("content", "")
                content_lower = content.lower()

                if primary_lower in content_lower and secondary_lower in content_lower:
                    for severity, keywords in self.severity_keywords.items():
                        if any(keyword in content_lower for keyword in keywords):
                            textual_severity.add(severity)

                    if any(keyword in content_lower for keyword in mechanistic_keywords_major):
                        mechanistic_severity = "major"
                    elif any(keyword in content_lower for keyword in mechanistic_keywords_moderate):
                        mechanistic_severity = mechanistic_severity or "moderate"

            # Textual severity should respect hierarchy (contraindicated > major > moderate > minor)
            severity_hierarchy = ["contraindicated", "major", "moderate", "minor"]
            for label in severity_hierarchy:
                if label in textual_severity:
                    if label == "contraindicated":
                        return label
                    if label == "major":
                        return "major"
                    if label == "moderate":
                        upgraded = upgrade_severity(severity_from_quant, "moderate")
                        return upgraded or "moderate"
                    if label == "minor":
                        return severity_from_quant or "minor"

            if mechanistic_severity == "major":
                return "major"
            if mechanistic_severity == "moderate":
                upgraded = upgrade_severity(severity_from_quant, "moderate")
                return upgraded or "moderate"

            if severity_from_quant:
                return severity_from_quant

            if textual_severity:
                return min(textual_severity, key=lambda s: severity_hierarchy.index(s))

            return "unknown"

        except Exception as e:
            logger.error(f"Error calculating interaction severity: {e}")
            return "unknown"

    def _extract_auc_cmax_changes(
        self, papers: list[dict], primary_drug: str, drug_pairs: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Extract AUC/Cmax changes with percent changes and significance."""
        papers = self._ensure_prepared_papers(papers)
        changes = []

        # Normalize the filter list at function start
        drug_pairs_norm = {d.lower() for d in (drug_pairs or [])}

        try:
            for paper in papers:
                content = paper.get("page_content") or paper.get("content", "")
                paper_id = paper.get("metadata", {}).get("pmid") or paper.get("id", "unknown")

                # Enhanced AUC/Cmax change patterns to handle negative changes, ranges, and relative phrasing
                change_patterns = [
                    # Original patterns
                    r"auc.*?(?:increased?|decreased?|changed?).*?(\d+(?:\.\d+)?)\s*([%]|fold)",
                    r"cmax.*?(?:increased?|decreased?|changed?).*?(\d+(?:\.\d+)?)\s*([%]|fold)",
                    r"clearance.*?(?:increased?|decreased?|changed?).*?(\d+(?:\.\d+)?)\s*([%]|fold)",
                    r"(\d+(?:\.\d+)?)\s*([%]|fold).*?(?:increase|decrease|change).*?(?:auc|cmax|clearance)",
                    r"(?:increase|decrease|change).*?(\d+(?:\.\d+)?)\s*([%]|fold).*?(?:auc|cmax|clearance)",
                    # Enhanced patterns for negative changes and relative phrasing
                    r"(?:auc|cmax|clearance).*?(?:↓|reduced?|diminished?|lowered?).*?(\d+(?:\.\d+)?)\s*([%]|fold)",
                    r"(?:auc|cmax|clearance).*?(?:↑|elevated?|increased?|raised?).*?(\d+(?:\.\d+)?)\s*([%]|fold)",
                    r"(?:↓|↑|reduced?|elevated?|diminished?|raised?).*?(?:auc|cmax|clearance).*?(\d+(?:\.\d+)?)\s*([%]|fold)",
                    # Value ranges (e.g., "AUC increased by 20-30%")
                    r"(?:auc|cmax|clearance).*?(?:increased?|decreased?|changed?).*?(\d+(?:\.\d+)?)\s*[-–—to]\s*(\d+(?:\.\d+)?)\s*([%]|fold)",
                    r"(?:increased?|decreased?|changed?).*?(?:auc|cmax|clearance).*?(\d+(?:\.\d+)?)\s*[-–—to]\s*(\d+(?:\.\d+)?)\s*([%]|fold)",
                    # Relative percentage changes
                    r"(\d+(?:\.\d+)?)\s*([%])\s*(?:reduction|decrease|decline).*?(?:auc|cmax|clearance)",
                    r"(\d+(?:\.\d+)?)\s*([%])\s*(?:increase|elevation|rise).*?(?:auc|cmax|clearance)",
                ]

                for pattern in change_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        try:
                            # Extract the numeric change and unit, handling value ranges
                            groups = match.groups()
                            if len(groups) >= 3 and groups[1] is not None:
                                # Handle range patterns (e.g., "20-30%")
                                value1 = float(groups[0])
                                value2 = float(groups[1])
                                change_value = (value1 + value2) / 2  # Use average of range
                                unit = groups[2] if len(groups) > 2 else groups[-1]
                            else:
                                # Handle single value patterns
                                change_value = float(groups[0])
                                unit = groups[1] if len(groups) > 1 else ""

                            # Enhanced direction detection with negative symbols and relative phrasing
                            context = content[max(0, match.start() - 50) : match.end() + 50].lower()
                            increase_markers = [
                                "increase",
                                "increased",
                                "higher",
                                "elevated",
                                "raised",
                                "greater",
                                "↑",
                                "rise",
                                "elevation",
                            ]
                            decrease_markers = [
                                "decrease",
                                "decreased",
                                "lower",
                                "reduced",
                                "diminished",
                                "less",
                                "↓",
                                "reduction",
                                "decline",
                                "lowered",
                            ]

                            # Check for explicit direction indicators in the matched text
                            matched_text = match.group(0).lower()
                            has_increase = any(word in context for word in increase_markers) or any(
                                word in matched_text for word in increase_markers
                            )
                            has_decrease = any(word in context for word in decrease_markers) or any(
                                word in matched_text for word in decrease_markers
                            )

                            if has_increase and not has_decrease:
                                direction = "increase"
                            elif has_decrease and not has_increase:
                                direction = "decrease"
                            else:
                                direction = "unknown"

                            # Extract parameter type (AUC, Cmax, Clearance)
                            if "auc" in context:
                                parameter = "auc"
                            elif "cmax" in context:
                                parameter = "cmax"
                            elif "clearance" in context or "cl/f" in context:
                                parameter = "clearance"
                            else:
                                parameter = "unknown"

                            # Normalise to fold change representations
                            unit_normalized = unit.lower() if unit else ""
                            fold_change = None
                            fold_change_ratio = None

                            if unit_normalized == "%" and direction in ("increase", "decrease"):
                                delta = change_value / 100.0
                                if direction == "increase":
                                    fold_change_ratio = 1 + delta
                                elif direction == "decrease":
                                    fold_change_ratio = max(0.0, 1 - delta)
                                fold_change = fold_change_ratio
                            elif "fold" in unit_normalized and direction in ("increase", "decrease"):
                                if direction == "increase":
                                    fold_change = change_value
                                    fold_change_ratio = change_value
                                elif change_value > 0:
                                    fold_change = -abs(change_value)
                                    if 0 < change_value < 1:
                                        fold_change_ratio = change_value
                                    else:
                                        fold_change_ratio = 1 / change_value

                            # Default ratio when unable to normalise
                            if fold_change_ratio is None and fold_change is not None:
                                fold_change_ratio = abs(fold_change)

                            # Enhanced statistical significance and confidence interval extraction
                            # Looser p-value patterns
                            p_value_patterns = [
                                r"p\s*[=<>]\s*(\d+(?:\.\d+)?)",  # Original pattern
                                r"p\s*[=<>]\s*0?\.(\d+)",  # p=.05, p<.001
                                r"p[-\s]*value\s*[=<>]\s*(\d+(?:\.\d+)?)",  # p-value=0.05
                                r"p[-\s]*val\s*[=<>]\s*(\d+(?:\.\d+)?)",  # p-val=0.05
                                r"\bp\s*(\d+(?:\.\d+)?)",  # Just p followed by number
                                r"significance.*?(\d+(?:\.\d+)?)",  # significance level
                            ]

                            p_value = None
                            for p_pattern in p_value_patterns:
                                sig_match = re.search(p_pattern, context, re.IGNORECASE)
                                if sig_match:
                                    try:
                                        p_value = float(sig_match.group(1))
                                        # Handle cases like p=.05 where we captured "05"
                                        if p_value > 1:
                                            p_value = p_value / 100
                                        break
                                    except (ValueError, IndexError):
                                        continue

                            is_significant = p_value < 0.05 if p_value else None

                            # Enhanced confidence interval extraction
                            ci_patterns = [
                                r"95%?\s*(?:ci|confidence\s*interval)[\s:]*(\d+(?:\.\d+)?)\s*[-–—to]\s*(\d+(?:\.\d+)?)",
                                r"ci\s*95%?[\s:]*(\d+(?:\.\d+)?)\s*[-–—to]\s*(\d+(?:\.\d+)?)",
                                r"\[(\d+(?:\.\d+)?)\s*[-–—to,]\s*(\d+(?:\.\d+)?)\]",  # [1.2-2.5] or [1.2, 2.5]
                                r"\((\d+(?:\.\d+)?)\s*[-–—to,]\s*(\d+(?:\.\d+)?)\)",  # (1.2-2.5) or (1.2, 2.5)
                                r"confidence\s*interval[\s:]*(\d+(?:\.\d+)?)\s*[-–—to]\s*(\d+(?:\.\d+)?)",
                            ]

                            confidence_interval = None
                            for ci_pattern in ci_patterns:
                                ci_match = re.search(ci_pattern, context, re.IGNORECASE)
                                if ci_match:
                                    try:
                                        lower = float(ci_match.group(1))
                                        upper = float(ci_match.group(2))
                                        if lower > upper:
                                            lower, upper = upper, lower  # Swap if reversed
                                        confidence_interval = (lower, upper)
                                        break
                                    except (ValueError, IndexError):
                                        continue

                            # Try to identify the interacting drug
                            interacting_drug = None
                            if drug_pairs:
                                for drug in drug_pairs:
                                    if drug.lower() in context:
                                        interacting_drug = drug
                                        break

                            # Apply filter: when drug_pairs is provided, include only matching pairs
                            if drug_pairs and (not interacting_drug or interacting_drug.lower() not in drug_pairs_norm):
                                continue  # skip appending change_data

                            change_data = {
                                "primary_drug": primary_drug,
                                "interacting_drug": interacting_drug,
                                "parameter": parameter,
                                "change_value": change_value,
                                "unit": unit,
                                "direction": direction,
                                "fold_change": fold_change,
                                "fold_change_ratio": fold_change_ratio,
                                "p_value": p_value,
                                "is_significant": is_significant,
                                "confidence_interval": confidence_interval,
                                "study_id": paper_id,
                                "context": context,
                                "source": "text_parsing",
                                # Provenance for change calculations
                                "original_change_value": change_value,
                                "original_unit": unit,
                                "normalization_applied": "fold_change_ratio" if fold_change_ratio else "none",
                                "calculation_notes": f"Direction: {direction}, Unit: {unit}, Fold change: {fold_change_ratio}",
                            }
                            changes.append(change_data)

                        except (ValueError, IndexError) as e:
                            logger.debug(f"Error parsing AUC/Cmax change: {e}")
                            continue

            return changes

        except Exception as e:
            logger.error(f"Error extracting AUC/Cmax changes: {e}")
            return []

    def _generate_pk_pd_summary(self, papers: list[dict], drug_name: str) -> dict[str, Any]:
        """Generate pharmacokinetic/pharmacodynamic summary."""
        papers = self._ensure_prepared_papers(papers)
        summary = {
            "absorption": {},
            "distribution": {},
            "metabolism": {},
            "elimination": {},
            "pharmacodynamics": {"metrics": [], "therapeutic_windows": [], "dose_response_patterns": []},
        }

        pd_metric_seen: set[tuple[str, str, int]] = set()
        therapeutic_seen: set[tuple[str, str, int]] = set()
        dose_response_recorded: set[str] = set()

        try:
            for paper in papers:
                content = paper.get("page_content") or paper.get("content", "")
                content_lower = content.lower()
                study_id = paper.get("metadata", {}).get("pmid", "unknown")

                if drug_name.lower() not in content.lower():
                    continue

                # Absorption patterns
                absorption_patterns = [
                    r"bioavailability.*?(\d+(?:\.\d+)?)\s*[%]",
                    r"absorption.*?(\d+(?:\.\d+)?)\s*[%]",
                    r"tmax.*?(\d+(?:\.\d+)?)\s*(h|hr|hours?|min)",
                ]

                # Distribution patterns
                distribution_patterns = [
                    r"volume.*distribution.*?(\d+(?:\.\d+)?)\s*(l|l/kg)",
                    r"protein.*binding.*?(\d+(?:\.\d+)?)\s*[%]",
                    r"vd.*?(\d+(?:\.\d+)?)\s*(l|l/kg)",
                ]

                # Metabolism patterns
                metabolism_patterns = [
                    r"metabolized.*?(\d+(?:\.\d+)?)\s*[%]",
                    r"hepatic.*metabolism.*?(\d+(?:\.\d+)?)\s*[%]",
                ]

                # Elimination patterns
                elimination_patterns = [
                    r"renal.*elimination.*?(\d+(?:\.\d+)?)\s*[%]",
                    r"excreted.*unchanged.*?(\d+(?:\.\d+)?)\s*[%]",
                    r"elimination.*half.*life.*?(\d+(?:\.\d+)?)\s*(h|hr|hours?)",
                ]

                # Extract data for each category
                categories = {
                    "absorption": absorption_patterns,
                    "distribution": distribution_patterns,
                    "metabolism": metabolism_patterns,
                    "elimination": elimination_patterns,
                }

                for category, patterns in categories.items():
                    for pattern in patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            try:
                                value = float(match.group(1))
                                unit = match.group(2) if len(match.groups()) > 1 else ""

                                if category not in summary:
                                    summary[category] = []

                                summary[category].setdefault("values", []).append(
                                    {"value": value, "unit": unit, "pattern_matched": pattern, "study_id": study_id}
                                )
                            except (ValueError, IndexError):
                                continue

                pd_summary = summary["pharmacodynamics"]

                pd_patterns = [
                    ("EC50", r"\bEC50\b[^\d]*(\d+(?:\.\d+)?)\s*([a-zA-Z/%µμ]+)?"),
                    ("IC50", r"\bIC50\b[^\d]*(\d+(?:\.\d+)?)\s*([a-zA-Z/%µμ]+)?"),
                    ("Emax", r"\bEmax\b[^\d]*(\d+(?:\.\d+)?)\s*([a-zA-Z/%µμ]+)?"),
                    ("ED50", r"\bED50\b[^\d]*(\d+(?:\.\d+)?)\s*([a-zA-Z/%µμ]+)?"),
                ]

                for term, pattern in pd_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        try:
                            value = float(match.group(1))
                        except (ValueError, TypeError):
                            continue
                        unit = match.group(2).strip() if match.lastindex and match.group(2) else None
                        key = (study_id, term, match.start())
                        if key in pd_metric_seen:
                            continue

                        context_window = content[max(0, match.start() - 40) : match.end() + 40]
                        context_clean = re.sub(r"\s+", " ", context_window).strip()
                        pd_summary["metrics"].append(
                            {"term": term, "value": value, "unit": unit, "study_id": study_id, "context": context_clean}
                        )
                        pd_metric_seen.add(key)

                therapeutic_terms = ["therapeutic index", "therapeutic window"]
                for term in therapeutic_terms:
                    for match in re.finditer(term, content_lower):
                        key = (study_id, term, match.start())
                        if key in therapeutic_seen:
                            continue
                        snippet = content[max(0, match.start() - 50) : match.start() + 120]
                        snippet_clean = re.sub(r"\s+", " ", snippet).strip()
                        value_match = re.search(r"(\d+(?:\.\d+)?)", snippet_clean)
                        unit_match = re.search(r"(ng/ml|µg/ml|ug/ml|mg/l|%)", snippet_clean, re.IGNORECASE)
                        pd_summary["therapeutic_windows"].append(
                            {
                                "term": term,
                                "value": float(value_match.group(1)) if value_match else None,
                                "unit": unit_match.group(1) if unit_match else None,
                                "study_id": study_id,
                                "context": snippet_clean,
                            }
                        )
                        therapeutic_seen.add(key)

                dosage_entries: list[dict[str, Any]] = []
                if self.pharma_processor:
                    try:
                        dosage_entries = self.pharma_processor.extract_dosage_information(content)
                    except Exception:
                        dosage_entries = []

                if re.search(r"dose[-\s]?response", content_lower) and study_id not in dose_response_recorded:
                    amounts = [entry["amount"] for entry in dosage_entries if entry.get("amount") is not None]
                    units = sorted({entry["unit"] for entry in dosage_entries if entry.get("unit")})
                    pd_summary["dose_response_patterns"].append(
                        {
                            "study_id": study_id,
                            "dose_range": {
                                "min": min(amounts) if amounts else None,
                                "max": max(amounts) if amounts else None,
                            },
                            "units": units,
                            "notes": "Dose-response relationship described in study",
                        }
                    )
                    dose_response_recorded.add(study_id)

            return summary

        except Exception as e:
            logger.error(f"Error generating PK/PD summary: {e}")
            return {"error": str(e)}

    def _identify_clinical_recommendations(self, papers: list[dict], interaction_type: str) -> list[str]:
        """Identify clinical recommendations from papers."""
        papers = self._ensure_prepared_papers(papers)
        recommendations = []

        try:
            for paper in papers:
                content = paper.get("page_content") or paper.get("content", "")

                for pattern in self.recommendation_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Extract surrounding context for the recommendation
                        start = max(0, match.start() - 100)
                        end = min(len(content), match.end() + 200)
                        recommendation_text = content[start:end].strip()

                        # Clean up the recommendation
                        recommendation_text = re.sub(r"\s+", " ", recommendation_text)
                        if len(recommendation_text) > 300:
                            recommendation_text = recommendation_text[:300] + "..."

                        # Wrap with research-only language to prevent advice leakage
                        safe_recommendation_text = self._make_recommendation_research_safe(recommendation_text)

                        recommendations.append(
                            {
                                "text": safe_recommendation_text,
                                "action": match.group(1).lower(),
                                "study_id": paper.get("metadata", {}).get("pmid", "unknown"),
                                "recommendation_type": "research_recommendation",  # Flag indicating this is research data
                            }
                        )

            # Remove duplicates and group by action type
            unique_recommendations = []
            seen = set()
            for rec in recommendations:
                if rec["text"] not in seen:
                    unique_recommendations.append(rec)
                    seen.add(rec["text"])

            return unique_recommendations[:10]  # Limit to top 10 recommendations

        except Exception as e:
            logger.error(f"Error identifying clinical recommendations: {e}")
            return []

    def _format_interaction_report(self, analysis_results: dict[str, Any]) -> str:
        """Format comprehensive interaction report."""
        try:
            primary_drug = analysis_results["primary_drug"]
            secondary_drugs = analysis_results["secondary_drugs"]

            report_lines = [
                f"# Drug Interaction Analysis Report: {primary_drug.title()}",
                "",
                f"## Summary",
                f"Primary Drug: {primary_drug}",
                f"Secondary Drugs Analyzed: {', '.join(secondary_drugs) if secondary_drugs else 'All identified interactions'}",
                "",
            ]

            # PK Parameters Summary
            pk_params = analysis_results.get("pk_parameters", {})
            if pk_params:
                report_lines.extend(["## Pharmacokinetic Parameters", ""])
                pk_source = analysis_results.get("pk_parameter_source")
                if pk_source == "text_parsing":
                    report_lines.append(
                        "- Note: PK metrics derived from text parsing heuristics; validate against study tables when available."
                    )
                elif pk_source == "tabular_extraction":
                    report_lines.append("- Note: PK metrics normalized from structured tabular extractions.")
                elif pk_source:
                    report_lines.append(f"- Note: PK metrics derived from {pk_source}.")

                for param in sorted(pk_params.keys()):
                    data = pk_params[param]
                    if not data:
                        continue

                    mean_value = data.get("mean_value")
                    if mean_value is None:
                        continue

                    normalized_units = data.get("normalized_units") or []
                    raw_units = data.get("units") or []
                    canonical_unit = next((u for u in normalized_units if u), None)
                    primary_unit = canonical_unit or (raw_units[0] if raw_units else None)

                    unit_phrase = "unit not specified"
                    if canonical_unit:
                        unit_phrase = f"normalized to {canonical_unit}"
                    elif primary_unit:
                        unit_phrase = f"reported in {primary_unit}"

                    range_normalized = data.get("normalized_range") or []
                    range_raw = data.get("range_raw") or []

                    def _format_range(values: list[float]) -> str | None:
                        if not values:
                            return None
                        if len(values) == 1:
                            return f"single observation {values[0]:.2f}"
                        return f"range {values[0]:.2f}–{values[1]:.2f}"

                    range_text = _format_range(range_normalized) or _format_range(range_raw)

                    stats_parts: list[str] = []
                    stats_parts.append(f"mean {mean_value:.2f}")
                    if range_text:
                        stats_parts.append(range_text)
                    sample_count = data.get("count")
                    if sample_count:
                        stats_parts.append(f"n={sample_count}")
                    study_ids = data.get("studies") or []
                    if study_ids:
                        stats_parts.append(f"{len(study_ids)} studies")

                    line = f"- **{param.upper()}**: {', '.join(stats_parts)} ({unit_phrase})"
                    if raw_units and canonical_unit and len(raw_units) > 1:
                        line += f"; raw units observed: {', '.join(sorted(set(raw_units)))}"
                    report_lines.append(line)

            # CYP Interactions
            cyp_data = analysis_results.get("cyp_interactions", {})
            if cyp_data.get("enzymes_identified"):
                report_lines.extend(["", "## CYP Enzyme Interactions", ""])
                enzymes = cyp_data["enzymes_identified"]
                report_lines.append(f"- **Enzymes Involved**: {', '.join(enzymes)}")

                for relationship_type in ["substrate", "inhibitor", "inducer"]:
                    relationships = cyp_data.get(f"{relationship_type}_relationships", [])
                    if relationships:
                        report_lines.append(
                            f"- **{relationship_type.title()} Relationships**: {len(relationships)} identified"
                        )

            # AUC/Cmax Changes
            auc_cmax_changes = analysis_results.get("auc_cmax_changes", [])
            if auc_cmax_changes:
                report_lines.extend(["", "## Pharmacokinetic Changes", ""])
                change_sources = sorted({change.get("source") for change in auc_cmax_changes if change.get("source")})
                if change_sources:
                    human_sources = []
                    for src in change_sources:
                        if src == "text_parsing":
                            human_sources.append("text parsing")
                        elif src == "tabular_extraction":
                            human_sources.append("tabular extraction")
                        else:
                            human_sources.append(str(src))
                    report_lines.append(f"- Note: PK change metrics derived from {', '.join(human_sources)}.")

                for change in auc_cmax_changes[:5]:  # Top 5 changes
                    parameter = change.get("parameter", "Unknown").upper()
                    direction = change.get("direction", "unknown")
                    change_value = change.get("change_value")
                    unit = (change.get("unit") or "").strip()
                    fold_change_ratio = change.get("fold_change_ratio")

                    percent_value: float | None = None
                    if unit == "%" and change_value is not None:
                        percent_value = change_value if direction == "increase" else -abs(change_value)
                    elif fold_change_ratio is not None and fold_change_ratio > 0:
                        percent_value = (fold_change_ratio - 1) * 100
                        if direction == "decrease" and percent_value > 0:
                            percent_value = -percent_value

                    percent_text = f"{percent_value:+.1f}%" if percent_value is not None else None
                    fold_text = f"{fold_change_ratio:.2f}x" if fold_change_ratio not in (None, 0) else None
                    magnitude_display = None
                    if percent_text and fold_text:
                        magnitude_display = f"{percent_text} ({fold_text})"
                    elif percent_text:
                        magnitude_display = percent_text
                    elif fold_text:
                        magnitude_display = fold_text
                    elif change_value is not None:
                        magnitude_display = f"{change_value}{unit}"
                    else:
                        magnitude_display = "magnitude not reported"

                    interacting_drug = change.get("interacting_drug") or "unspecified partner"
                    drug_pair = f"{primary_drug} + {interacting_drug}"

                    detail_segments: list[str] = []
                    ci = change.get("confidence_interval")
                    if ci and len(ci) == 2:
                        detail_segments.append(f"95% CI {ci[0]:.2f}–{ci[1]:.2f}")
                    p_value = change.get("p_value")
                    if p_value is not None:
                        detail_segments.append(f"p={p_value:.3f}")
                    elif change.get("is_significant") is False:
                        detail_segments.append("not statistically significant")
                    study_id = change.get("study_id")
                    if study_id:
                        detail_segments.append(f"study {study_id}")

                    source_label = change.get("source")
                    if source_label == "text_parsing":
                        detail_segments.append("derived via text parsing")
                    elif source_label == "tabular_extraction":
                        detail_segments.append("derived from tabular extraction")

                    detail_text = f"; {'; '.join(detail_segments)}" if detail_segments else ""

                    report_lines.append(f"- {drug_pair}: {parameter} {direction} {magnitude_display}{detail_text}")

            # Clinical Recommendations
            recommendations = analysis_results.get("clinical_recommendations", [])
            if recommendations:
                report_lines.extend(["", "## Clinical Recommendations", ""])
                for rec in recommendations[:5]:  # Top 5 recommendations
                    report_lines.append(f"- {rec['text']}")

            # Interaction Severities
            severities = analysis_results.get("interaction_severities", {})
            if severities:
                report_lines.extend(["", "## Interaction Severity Assessment", ""])
                for drug, severity in severities.items():
                    report_lines.append(f"- **{primary_drug} + {drug}**: {severity.title()}")

            return "\n".join(report_lines)

        except Exception as e:
            logger.error(f"Error formatting interaction report: {e}")
            return f"Error generating report: {str(e)}"

    def _determine_interaction_strength(self, content: str, drug_name: str, enzyme: str) -> str:
        """Determine interaction strength (strong/moderate/weak)."""
        context = content.lower()

        # Strong indicators
        strong_keywords = ["strong", "potent", "significant", "major", "pronounced", "substantial"]
        moderate_keywords = ["moderate", "modest", "measurable", "noticeable"]
        weak_keywords = ["weak", "mild", "slight", "minor", "minimal"]

        if any(keyword in context for keyword in strong_keywords):
            return "strong"
        elif any(keyword in context for keyword in moderate_keywords):
            return "moderate"
        elif any(keyword in context for keyword in weak_keywords):
            return "weak"
        else:
            return "unknown"

    def _make_recommendation_research_safe(self, recommendation_text: str) -> str:
        """Wrap clinical recommendations with research-only language to prevent advice leakage."""
        try:
            # Remove or soften imperative language
            safe_text = recommendation_text

            # Replace second-person imperatives with research-only language
            imperative_replacements = {
                r"\byou\s+should\b": "literature reports suggest",
                r"\byou\s+must\b": "studies indicate the need to",
                r"\byou\s+need\s+to\b": "research suggests",
                r"\bpatients?\s+should\b": "literature recommends that patients",
                r"\bpatients?\s+must\b": "studies suggest patients may need to",
                r"\bavoid\b": "literature reports avoiding",
                r"\bmonitor\b": "studies recommend monitoring",
                r"\badjust\b": "research suggests adjusting",
                r"\breduce\b": "literature reports reducing",
                r"\bincrease\b": "studies suggest increasing",
                r"\bdo\s+not\b": "literature advises against",
                r"\bdon\'t\b": "research advises against",
            }

            for pattern, replacement in imperative_replacements.items():
                safe_text = re.sub(pattern, replacement, safe_text, flags=re.IGNORECASE)

            # Prefix with research context if not already present
            research_prefixes = [
                "literature reports",
                "studies suggest",
                "research indicates",
                "according to research",
                "clinical studies report",
                "published data suggest",
            ]

            has_research_context = any(prefix in safe_text.lower() for prefix in research_prefixes)

            if not has_research_context:
                safe_text = f"Literature reports suggest: {safe_text}"

            # Ensure it's clearly marked as research data, not clinical advice
            if not safe_text.lower().startswith(
                ("literature", "studies", "research", "according to", "clinical studies", "published data")
            ):
                safe_text = f"Research findings indicate: {safe_text}"

            return safe_text

        except Exception as e:
            logger.warning(f"Error making recommendation research-safe: {e}")
            # Fallback: wrap the entire text with clear research language
            return f"Literature reports suggest: {recommendation_text} (Note: Research data only, not clinical advice)"


__all__ = ["DDIPKProcessor", "PKParameter", "DrugInteraction", "InteractionReport"]
