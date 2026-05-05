"""Unit tests for SynthesisEngine study phase regex classification."""
import os
import sys
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synthesis_engine import SynthesisEngine


class TestSynthesisEnginePhaseRegex(unittest.TestCase):
    """Test study phase classification in SynthesisEngine."""

    def setUp(self):
        """Set up test instance."""
        self.engine = SynthesisEngine()

    def test_phase_i_classification(self):
        """Test Phase I trial classification."""
        test_abstracts = [
            "This Phase I clinical trial evaluated the safety and tolerability of drug X.",
            "A phase i dose-escalation study in patients with advanced cancer.",
            "Phase I trial investigating maximum tolerated dose.",
            "This was a phase 1 study examining pharmacokinetics.",
        ]

        for abstract in test_abstracts:
            study_type = self.engine._classify_study_type(abstract)
            self.assertEqual(study_type, "clinical_trial", f"Failed to classify Phase I trial: {abstract}")

    def test_phase_ii_classification(self):
        """Test Phase II trial classification."""
        test_abstracts = [
            "A Phase II randomized controlled trial of drug Y.",
            "This phase ii efficacy study enrolled 100 patients.",
            "Phase 2 trial demonstrating significant improvement in outcomes.",
            "Multi-center phase II study of combination therapy.",
        ]

        for abstract in test_abstracts:
            study_type = self.engine._classify_study_type(abstract)
            self.assertEqual(study_type, "clinical_trial", f"Failed to classify Phase II trial: {abstract}")

    def test_phase_iii_classification(self):
        """Test Phase III trial classification."""
        test_abstracts = [
            "Large Phase III trial comparing drug Z to standard care.",
            "This phase iii randomized double-blind study.",
            "Phase 3 confirmatory trial with 1000 participants.",
            "International phase III clinical trial results.",
        ]

        for abstract in test_abstracts:
            study_type = self.engine._classify_study_type(abstract)
            self.assertEqual(study_type, "clinical_trial", f"Failed to classify Phase III trial: {abstract}")

    def test_phase_iv_classification(self):
        """Test Phase IV trial classification."""
        test_abstracts = [
            "Post-marketing Phase IV surveillance study.",
            "A phase iv safety study in real-world settings.",
            "Phase 4 observational study of long-term effects.",
            "Phase IV post-marketing clinical trial.",
        ]

        for abstract in test_abstracts:
            study_type = self.engine._classify_study_type(abstract)
            self.assertEqual(study_type, "clinical_trial", f"Failed to classify Phase IV trial: {abstract}")

    def test_roman_numeral_patterns(self):
        """Test various roman numeral patterns."""
        test_cases = [
            ("Phase I study", "clinical_trial"),
            ("Phase II trial", "clinical_trial"),
            ("Phase III investigation", "clinical_trial"),
            ("Phase IV surveillance", "clinical_trial"),
            ("phase i dose escalation", "clinical_trial"),
            ("phase ii efficacy", "clinical_trial"),
            ("phase iii confirmatory", "clinical_trial"),
            ("phase iv post-marketing", "clinical_trial"),
        ]

        for abstract, expected in test_cases:
            study_type = self.engine._classify_study_type(abstract)
            self.assertEqual(study_type, expected, f"Failed classification for: {abstract}")

    def test_non_clinical_trial_abstracts(self):
        """Test that non-clinical trial abstracts are not misclassified."""
        test_abstracts = [
            "This systematic review analyzed phase-related outcomes.",
            "Meta-analysis of multiple studies discussing phases of treatment.",
            "Review article about clinical trial phases in oncology.",
            "Observational study without phase designation.",
            "In vitro study of drug mechanisms.",
        ]

        for abstract in test_abstracts:
            study_type = self.engine._classify_study_type(abstract)
            self.assertNotEqual(study_type, "clinical_trial", f"Incorrectly classified as clinical trial: {abstract}")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        test_cases = [
            ("Phase V trial", "clinical_trial"),  # Should catch Phase V
            ("PHASE I STUDY", "clinical_trial"),  # All caps
            ("Mixed case Phase Ii trial", "clinical_trial"),  # Mixed case
            ("phase0 study", "review"),  # No space, should not match
            ("phasestudy", "review"),  # No space or number
            ("multi-phase study", "review"),  # Not specific phase
        ]

        for abstract, expected in test_cases:
            study_type = self.engine._classify_study_type(abstract)
            self.assertEqual(study_type, expected, f"Edge case failed for: {abstract}")


if __name__ == "__main__":
    unittest.main()
