"""Unit tests for synthesis engine module."""
from unittest.mock import Mock, patch

import pytest

from src.synthesis_engine import KeyFinding, SynthesisEngine


class TestSynthesisEngine:
    """Test cases for SynthesisEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a SynthesisEngine instance for testing."""
        return SynthesisEngine()

    @pytest.fixture
    def sample_papers(self):
        """Sample papers for testing."""
        return [
            {
                "id": "paper_1",
                "page_content": "This randomized controlled trial studied warfarin interactions with fluconazole. Results showed a 2-fold increase in AUC.",
                "metadata": {
                    "pmid": "12345678",
                    "title": "Warfarin-Fluconazole Interaction Study",
                    "authors": ["Smith J", "Jones A"],
                    "journal": "Clinical Pharmacology",
                    "year": "2023",
                    "abstract": "A study of drug interactions between warfarin and fluconazole showing significant pharmacokinetic changes.",
                },
            },
            {
                "id": "paper_2",
                "page_content": "Meta-analysis of clinical trials examining cardiovascular effects. Safety profile shows minimal adverse events.",
                "metadata": {
                    "pmid": "87654321",
                    "title": "Meta-analysis of Cardiovascular Safety",
                    "authors": ["Brown K", "Davis M", "Wilson P"],
                    "journal": "Cardiology Research",
                    "year": "2022",
                },
            },
        ]

    def test_initialization(self, engine):
        """Test SynthesisEngine initialization."""
        assert engine is not None
        assert hasattr(engine, "study_type_patterns")
        assert hasattr(engine, "evidence_levels")

    def test_generate_meta_summary_success(self, engine, sample_papers):
        """Test successful meta-summary generation."""
        result = engine.generate_meta_summary(sample_papers, "drug interactions", max_bullet_points=5)

        assert "error" not in result
        assert result["query"] == "drug interactions"
        assert result["total_papers"] == 2
        assert "key_findings" in result
        assert "bullet_points" in result
        assert "comparative_analysis" in result
        assert "evidence_synthesis" in result
        assert "confidence_scores" in result
        assert "research_gaps" in result
        assert "citations" in result

    def test_generate_meta_summary_bullet_glyph(self, engine, sample_papers):
        """Bullet points optionally include glyphs."""
        result = engine.generate_meta_summary(sample_papers, "drug interactions", include_bullet_glyph=True)

        assert "bullet_points" in result
        if result["bullet_points"]:
            assert all(point.startswith("• ") for point in result["bullet_points"])

    def test_generate_meta_summary_empty_papers(self, engine):
        """Test meta-summary generation with empty papers list."""
        result = engine.generate_meta_summary([], "test query")

        assert result["total_papers"] == 0
        assert result["key_findings"] == []
        assert result["bullet_points"] == []

    def test_extract_key_findings(self, engine, sample_papers):
        """Test key findings extraction."""
        findings = engine._extract_key_findings(sample_papers)

        assert len(findings) <= len(sample_papers)
        for finding in findings:
            assert isinstance(finding, KeyFinding)
            assert finding.paper_id is not None
            assert finding.finding is not None
            assert finding.evidence_level is not None
            assert finding.confidence is not None
            assert isinstance(finding.drug_entities, list)

    def test_classify_study_type(self, engine):
        """Test study type classification."""
        # Test clinical trial classification
        rct_text = "This randomized controlled trial examined patients"
        assert engine._classify_study_type(rct_text) == "clinical_trial"

        # Test meta-analysis classification
        meta_text = "This meta-analysis of 15 studies showed"
        assert engine._classify_study_type(meta_text) == "meta_analysis"

        # Test default classification
        unknown_text = "This is some general text"
        assert engine._classify_study_type(unknown_text) == "review"

    def test_perform_comparative_analysis(self, engine, sample_papers):
        """Test comparative analysis functionality."""
        analysis = engine._perform_comparative_analysis(sample_papers, "drug interactions")

        assert "convergent_findings" in analysis
        assert "divergent_findings" in analysis
        assert "dose_response_patterns" in analysis
        assert "population_differences" in analysis
        assert "methodological_variations" in analysis

    def test_synthesize_evidence_levels(self, engine, sample_papers):
        """Test evidence level synthesis."""
        synthesis = engine._synthesize_evidence_levels(sample_papers)

        assert "evidence_distribution" in synthesis
        assert "total_papers" in synthesis
        assert "evidence_strength_score" in synthesis
        assert "primary_evidence_level" in synthesis
        assert "high_quality_papers" in synthesis

    def test_generate_bullet_points(self, engine, sample_papers):
        """Test bullet point generation."""
        # Create mock findings
        findings = [
            KeyFinding(
                paper_id="test_1",
                finding="Test finding 1",
                evidence_level="Level 2",
                confidence=0.9,
                drug_entities=["warfarin"],
                pk_parameters={},
                study_type="clinical_trial",
            ),
            KeyFinding(
                paper_id="test_2",
                finding="Test finding 2",
                evidence_level="Level 1",
                confidence=0.8,
                drug_entities=["fluconazole"],
                pk_parameters={},
                study_type="meta_analysis",
            ),
        ]

        comparative_analysis = {"convergent_findings": [], "dose_response_patterns": []}
        bullet_points = engine._generate_bullet_points(findings, comparative_analysis, [], max_points=5, min_points=1)

        assert isinstance(bullet_points, list)
        assert len(bullet_points) <= 5

    def test_identify_research_gaps(self, engine, sample_papers):
        """Test research gap identification."""
        gaps = engine._identify_research_gaps(sample_papers, "drug interactions")

        assert isinstance(gaps, list)
        assert len(gaps) <= 5

    def test_calculate_confidence_scores(self, engine, sample_papers):
        """Test confidence score calculation."""
        scores = engine._calculate_confidence_scores(sample_papers)

        assert "sample_size" in scores
        assert "study_quality" in scores
        assert "consistency" in scores
        assert "overall" in scores

        # Check score ranges
        for score in scores.values():
            if isinstance(score, (int, float)):
                assert 0 <= score <= 1

    def test_format_citations(self, engine, sample_papers):
        """Test citation formatting."""
        citations = engine._format_citations(sample_papers)

        assert isinstance(citations, list)
        assert len(citations) == len(sample_papers)

        for citation in citations:
            assert isinstance(citation, str)
            assert len(citation) > 0

    def test_extract_main_finding(self, engine):
        """Test main finding extraction."""
        content_with_conclusion = "Introduction text. Conclusion: This drug is effective. End."
        metadata = {"abstract": "Test abstract about drug efficacy."}

        finding = engine._extract_main_finding(content_with_conclusion, metadata)
        assert "effective" in finding.lower()

    def test_calculate_finding_confidence(self, engine):
        """Test finding confidence calculation."""
        content = "This is a comprehensive study with detailed methodology."
        metadata = {"pmid": "12345678", "doi": "10.1000/test"}

        confidence = engine._calculate_finding_confidence(content, "clinical_trial", metadata)

        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)

    def test_extract_outcomes(self, engine):
        """Test outcome extraction."""
        content = "Primary endpoint was efficacy. Secondary endpoints included safety and adverse events."
        outcomes = engine._extract_outcomes(content)

        assert isinstance(outcomes, list)
        # Should find at least efficacy and safety-related outcomes
        outcome_text = " ".join(outcomes).lower()
        assert any(term in outcome_text for term in ["efficacy", "safety", "adverse"])

    def test_are_findings_convergent(self, engine):
        """Test convergent findings detection."""
        convergent_findings = [
            "Drug X shows significant efficacy in clinical trials",
            "Clinical trials demonstrate Drug X efficacy and safety",
            "Studies show Drug X is effective and well-tolerated",
        ]

        divergent_findings = ["Drug A shows efficacy", "Drug B has safety concerns", "Different mechanism of action"]

        assert engine._are_findings_convergent(convergent_findings) == True
        assert engine._are_findings_convergent(divergent_findings) == False

    def test_synthesize_convergent_finding(self, engine):
        """Test convergent finding synthesis."""
        findings = [
            "Long finding about drug efficacy in clinical trials",
            "Short finding",
            "Medium length finding about safety",
        ]

        synthesized = engine._synthesize_convergent_finding(findings)
        assert synthesized == "Short finding"  # Should return shortest

    def test_extract_dose_response_data(self, engine, sample_papers):
        """Test dose-response data extraction."""
        with patch.object(engine, "pharma_processor") as mock_processor:
            mock_processor.extract_dosage_information.return_value = [
                {"amount": 10, "unit": "mg"},
                {"amount": 20, "unit": "mg"},
            ]

            dose_data = engine._extract_dose_response_data(sample_papers, "warfarin")

            assert isinstance(dose_data, list)

    @patch("src.synthesis_engine.PharmaceuticalProcessor")
    def test_with_pharmaceutical_processor(self, mock_processor_class):
        """Test SynthesisEngine with PharmaceuticalProcessor."""
        mock_processor = Mock()
        mock_processor.extract_drug_names.return_value = [{"name": "warfarin", "confidence": 0.9}]
        mock_processor.extract_pharmacokinetic_parameters.return_value = {"clearance": {"value": 10, "unit": "L/h"}}

        engine = SynthesisEngine(pharma_processor=mock_processor)

        papers = [{"id": "test", "page_content": "warfarin study results", "metadata": {"pmid": "12345"}}]

        engine._extract_key_findings(papers)

        # Should have called the pharmaceutical processor
        mock_processor.extract_drug_names.assert_called()
        mock_processor.extract_pharmacokinetic_parameters.assert_called()

    def test_error_handling(self, engine):
        """Test error handling in meta-summary generation."""
        # Test with invalid input
        result = engine.generate_meta_summary(None, "test query")

        # Should handle gracefully and return error info
        assert "error" in result or result["total_papers"] == 0

    def test_count_methods(self, engine, sample_papers):
        """Test various counting methods."""
        # Test study types counting
        study_counts = engine._count_study_types(sample_papers)
        assert isinstance(study_counts, dict)

        # Test with mock findings for drug and evidence counting
        mock_findings = [
            KeyFinding("1", "test", "Level 1", 0.8, ["drug1"], {}, "clinical_trial"),
            KeyFinding("2", "test", "Level 2", 0.9, ["drug1", "drug2"], {}, "meta_analysis"),
        ]

        drug_counts = engine._count_drug_mentions(mock_findings)
        assert isinstance(drug_counts, dict)
        assert "drug1" in drug_counts

        evidence_counts = engine._count_evidence_levels(mock_findings)
        assert isinstance(evidence_counts, dict)
        assert "Level 1" in evidence_counts or "Level 2" in evidence_counts


class TestKeyFinding:
    """Test cases for KeyFinding dataclass."""

    def test_key_finding_creation(self):
        """Test KeyFinding instance creation."""
        finding = KeyFinding(
            paper_id="test_id",
            finding="Test finding text",
            evidence_level="Level 1",
            confidence=0.85,
            drug_entities=["drug1", "drug2"],
            pk_parameters={"clearance": 10},
            study_type="clinical_trial",
        )

        assert finding.paper_id == "test_id"
        assert finding.finding == "Test finding text"
        assert finding.evidence_level == "Level 1"
        assert finding.confidence == 0.85
        assert finding.drug_entities == ["drug1", "drug2"]
        assert finding.pk_parameters == {"clearance": 10}
        assert finding.study_type == "clinical_trial"


if __name__ == "__main__":
    pytest.main([__file__])
