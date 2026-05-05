"""Unit tests for DDI/PK processor module."""
import re
from unittest.mock import Mock, patch

import pytest

from src.ddi_pk_processor import DDIPKProcessor, DrugInteraction, PKParameter


class TestDDIPKProcessor:
    """Test cases for DDIPKProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a DDIPKProcessor instance for testing."""
        return DDIPKProcessor()

    @pytest.fixture
    def sample_papers(self):
        """Sample papers for testing drug interactions."""
        return [
            {
                "id": "paper_1",
                "page_content": "Warfarin and fluconazole interaction study. AUC increased by 85% (p<0.05). CYP2C9 inhibition was observed. Clinical recommendation: monitor INR closely.",
                "metadata": {
                    "pmid": "12345678",
                    "title": "Warfarin-Fluconazole DDI Study",
                    "journal": "Clinical Pharmacology",
                    "year": "2023",
                },
            },
            {
                "id": "paper_2",
                "page_content": "Cmax of warfarin increased 2.1-fold when co-administered with amiodarone. Half-life extended to 45 hours. Major interaction severity.",
                "metadata": {
                    "pmid": "87654321",
                    "title": "Warfarin-Amiodarone Interaction",
                    "journal": "Drug Interactions",
                    "year": "2022",
                },
            },
        ]

    def test_initialization(self, processor):
        """Test DDIPKProcessor initialization."""
        assert processor is not None
        assert hasattr(processor, "cyp_patterns")
        assert hasattr(processor, "severity_keywords")
        assert hasattr(processor, "pk_parameter_patterns")

    def test_analyze_drug_interactions_success(self, processor, sample_papers):
        """Test successful drug interaction analysis."""
        result = processor.analyze_drug_interactions(sample_papers, "warfarin", ["fluconazole", "amiodarone"])

        assert "error" not in result
        assert result["primary_drug"] == "warfarin"
        assert result["analyzed_papers"] == 2
        assert "secondary_drugs_analyzed" in result
        assert "pk_parameters" in result
        assert "cyp_interactions" in result
        assert "auc_cmax_changes" in result
        assert "clinical_recommendations" in result
        assert "formatted_report" in result

    def test_analyze_drug_interactions_no_secondary_drugs(self, processor, sample_papers):
        """Test drug interaction analysis without specific secondary drugs."""
        result = processor.analyze_drug_interactions(sample_papers, "warfarin")

        assert result["primary_drug"] == "warfarin"
        assert result["secondary_drugs_analyzed"] == []

    def test_extract_pk_parameters(self, processor, sample_papers):
        """Test PK parameter extraction."""
        pk_params = processor._extract_pk_parameters(sample_papers, "warfarin")

        assert isinstance(pk_params, dict)
        # Should extract AUC, Cmax, half-life from the sample papers
        if pk_params:
            for param_name, param_data in pk_params.items():
                if param_data:
                    assert "count" in param_data
                    assert "parameters" in param_data

    def test_analyze_cyp_interactions(self, processor, sample_papers):
        """Test CYP enzyme interaction analysis."""
        cyp_analysis = processor._analyze_cyp_interactions(sample_papers, "warfarin")

        assert "enzymes_identified" in cyp_analysis
        assert "substrate_relationships" in cyp_analysis
        assert "inhibitor_relationships" in cyp_analysis
        assert "inducer_relationships" in cyp_analysis
        assert "strength_classifications" in cyp_analysis

    def test_calculate_interaction_severity(self, processor, sample_papers):
        """Test interaction severity calculation."""
        interaction_data = {"primary_drug": "warfarin", "secondary_drug": "fluconazole", "papers": sample_papers}

        severity = processor._calculate_interaction_severity(interaction_data)
        assert severity in ["contraindicated", "major", "moderate", "minor", "unknown"]

    def test_extract_auc_cmax_changes(self, processor, sample_papers):
        """Test AUC/Cmax change extraction."""
        changes = processor._extract_auc_cmax_changes(sample_papers, "warfarin", ["fluconazole"])

        assert isinstance(changes, list)
        for change in changes:
            assert "primary_drug" in change
            assert "parameter" in change
            assert "change_value" in change
            assert "direction" in change
            assert "study_id" in change

    def test_generate_pk_pd_summary(self, processor, sample_papers):
        """Test PK/PD summary generation."""
        summary = processor._generate_pk_pd_summary(sample_papers, "warfarin")

        assert "absorption" in summary
        assert "distribution" in summary
        assert "metabolism" in summary
        assert "elimination" in summary

    def test_identify_clinical_recommendations(self, processor, sample_papers):
        """Test clinical recommendation identification."""
        recommendations = processor._identify_clinical_recommendations(sample_papers, "all")

        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert "text" in rec
            assert "action" in rec
            assert "study_id" in rec

    def test_format_interaction_report(self, processor):
        """Test interaction report formatting."""
        analysis_results = {
            "primary_drug": "warfarin",
            "secondary_drugs": ["fluconazole"],
            "pk_parameters": {"auc": {"mean_value": 1.85, "units": ["fold"], "count": 1}},
            "cyp_interactions": {"enzymes_identified": ["CYP2C9"]},
            "auc_cmax_changes": [{"parameter": "auc", "change_value": 85, "unit": "%", "direction": "increase"}],
            "clinical_recommendations": [{"text": "Monitor INR closely", "action": "monitor"}],
            "interaction_severities": {"fluconazole": "major"},
        }

        report = processor._format_interaction_report(analysis_results)

        assert isinstance(report, str)
        assert "warfarin" in report.lower()
        assert "fluconazole" in report.lower()

    def test_determine_interaction_strength(self, processor):
        """Test interaction strength determination."""
        # Test strong interaction
        strong_content = "This is a strong and significant interaction"
        strength = processor._determine_interaction_strength(strong_content, "drug1", "cyp2c9")
        assert strength == "strong"

        # Test moderate interaction
        moderate_content = "This shows moderate effects"
        strength = processor._determine_interaction_strength(moderate_content, "drug1", "cyp2c9")
        assert strength == "moderate"

        # Test weak interaction
        weak_content = "Only weak effects were observed"
        strength = processor._determine_interaction_strength(weak_content, "drug1", "cyp2c9")
        assert strength == "weak"

        # Test unknown
        unknown_content = "Some content without strength indicators"
        strength = processor._determine_interaction_strength(unknown_content, "drug1", "cyp2c9")
        assert strength == "unknown"

    @patch("src.ddi_pk_processor.PharmaceuticalProcessor")
    def test_with_pharmaceutical_processor(self, mock_processor_class):
        """Test DDIPKProcessor with PharmaceuticalProcessor."""
        mock_processor = Mock()
        mock_processor.extract_cyp_enzymes.return_value = ["CYP2C9", "CYP3A4"]
        mock_processor.extract_dosage_information.return_value = [{"amount": 5, "unit": "mg"}]

        processor = DDIPKProcessor(pharma_processor=mock_processor)

        papers = [{"page_content": "warfarin metabolism study", "metadata": {"pmid": "12345"}}]

        processor._analyze_cyp_interactions(papers, "warfarin")

        # Should have called the pharmaceutical processor
        mock_processor.extract_cyp_enzymes.assert_called()

    def test_severity_keyword_classification(self, processor):
        """Test severity keyword classification."""
        # Test all severity levels
        for severity, keywords in processor.severity_keywords.items():
            assert isinstance(keywords, list)
            assert len(keywords) > 0

    def test_normalize_pk_measurement_unit_variants(self, processor):
        """Handle unit variants such as ng·mL^-1 and geometric mean ratios."""
        value, canonical, *_ = processor._normalize_pk_measurement("cmax", 2.0, "ng·mL^-1")
        assert canonical == "ng/mL"
        assert value == 2.0

        value, canonical, *_ = processor._normalize_pk_measurement("cmax", 0.5, "mg/L")
        assert canonical == "ng/mL"
        assert value == 500.0

        value, canonical, *_ = processor._normalize_pk_measurement("auc", 1.8, "geometric mean ratio")
        assert canonical == "fold"
        assert value == 1.8

    def test_configurable_severity_thresholds(self):
        """Severity thresholds can be overridden via configuration."""
        papers = [
            {
                "page_content": "Cmax increased 3.5 fold when drugA was combined with drugB.",
                "metadata": {"pmid": "PMID-001"},
            }
        ]

        baseline = DDIPKProcessor()
        baseline_prepared = baseline._prepare_papers(papers)
        default_severity = baseline._calculate_interaction_severity(
            {
                "primary_drug": "drugA",
                "secondary_drug": "drugB",
                "papers": baseline_prepared,
            }
        )
        assert default_severity == "moderate"

        override = {"severity_thresholds": {"increase": {"major": 3.0}}}
        tuned = DDIPKProcessor(config=override)
        tuned_prepared = tuned._prepare_papers(papers)
        tuned_severity = tuned._calculate_interaction_severity(
            {
                "primary_drug": "drugA",
                "secondary_drug": "drugB",
                "papers": tuned_prepared,
            }
        )

        # With a lower major threshold the interaction should escalate to "major"
        assert tuned_severity == "major"

    def test_pk_parameter_patterns(self, processor):
        """Test PK parameter pattern definitions."""
        patterns = processor.pk_parameter_patterns

        required_params = ["auc", "cmax", "clearance", "half_life"]
        for param in required_params:
            assert param in patterns
            assert "patterns" in patterns[param]
            assert "units" in patterns[param]

    def test_error_handling(self, processor):
        """Test error handling in drug interaction analysis."""
        # Test with None input
        result = processor.analyze_drug_interactions(None, "warfarin")

        # Should handle gracefully
        assert "error" in result or result["analyzed_papers"] == 0

    def test_cyp_pattern_matching(self, processor):
        """Test CYP enzyme pattern matching."""
        test_content = "This drug is a substrate of CYP2C9 and inhibits CYP3A4"

        # Test substrate detection
        substrate_found = False
        for pattern in processor.cyp_patterns["substrate"]:
            if re.search(pattern, test_content, re.IGNORECASE):
                substrate_found = True
                break

        # Test inhibitor detection
        inhibitor_found = False
        for pattern in processor.cyp_patterns["inhibitor"]:
            if re.search(pattern, test_content, re.IGNORECASE):
                inhibitor_found = True
                break

        assert substrate_found
        assert inhibitor_found


class TestPKParameter:
    """Test cases for PKParameter dataclass."""

    def test_pk_parameter_creation(self):
        """Test PKParameter instance creation."""
        pk_param = PKParameter(
            parameter="auc",
            value=125.5,
            unit="ng⋅h/mL",
            confidence_interval=(100.0, 151.0),
            p_value=0.03,
            study_id="12345678",
        )

        assert pk_param.parameter == "auc"
        assert pk_param.value == 125.5
        assert pk_param.unit == "ng⋅h/mL"
        assert pk_param.confidence_interval == (100.0, 151.0)
        assert pk_param.p_value == 0.03
        assert pk_param.study_id == "12345678"

    def test_pk_parameter_with_none_values(self):
        """Test PKParameter with None values."""
        pk_param = PKParameter(
            parameter="cmax", value=None, unit="ng/mL", confidence_interval=None, p_value=None, study_id="87654321"
        )

        assert pk_param.parameter == "cmax"
        assert pk_param.value is None
        assert pk_param.confidence_interval is None
        assert pk_param.p_value is None


class TestDrugInteraction:
    """Test cases for DrugInteraction dataclass."""

    def test_drug_interaction_creation(self):
        """Test DrugInteraction instance creation."""
        interaction = DrugInteraction(
            primary_drug="warfarin",
            secondary_drug="fluconazole",
            interaction_type="pharmacokinetic",
            severity="major",
            mechanism="CYP2C9 inhibition",
            clinical_effect="Increased anticoagulation",
            evidence_level="Level 2",
            study_id="12345678",
        )

        assert interaction.primary_drug == "warfarin"
        assert interaction.secondary_drug == "fluconazole"
        assert interaction.interaction_type == "pharmacokinetic"
        assert interaction.severity == "major"
        assert interaction.mechanism == "CYP2C9 inhibition"
        assert interaction.clinical_effect == "Increased anticoagulation"
        assert interaction.evidence_level == "Level 2"
        assert interaction.study_id == "12345678"


if __name__ == "__main__":
    pytest.main([__file__])
