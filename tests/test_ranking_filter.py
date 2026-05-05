import datetime

import pytest

from src.ranking_filter import StudyRankingFilter


def test_estimate_sample_size_prioritizes_explicit_n_equals():
    ranking_filter = StudyRankingFilter()
    text = "The cohort included n=48 participants and an additional 120 patients in registries."
    assert ranking_filter._estimate_sample_size_from_text(text) == 48


def test_estimate_sample_size_ignores_four_digit_years():
    ranking_filter = StudyRankingFilter()
    text = "A retrospective review followed 2020 participants from the 2020 outbreak."
    assert ranking_filter._estimate_sample_size_from_text(text) is None


def test_estimate_sample_size_with_enrolled_context():
    ranking_filter = StudyRankingFilter()
    text = "A total of 180 patients were enrolled across sites in 2020."
    assert ranking_filter._estimate_sample_size_from_text(text) == 180


def test_estimate_sample_size_with_randomized_prefix():
    ranking_filter = StudyRankingFilter()
    text = "Randomized 95 participants to receive either treatment or placebo."
    assert ranking_filter._estimate_sample_size_from_text(text) == 95


def test_recency_decay_years_parameter_adjusts_decay():
    current_year = datetime.datetime.now(datetime.timezone.utc).year
    default_filter = StudyRankingFilter()
    fast_decay_filter = StudyRankingFilter(recency_decay_years=5)

    year = current_year - 5
    default_score = default_filter._calculate_recency_score(year)
    fast_decay_score = fast_decay_filter._calculate_recency_score(year)

    assert fast_decay_score < default_score


def test_verbose_ranking_explanation_includes_terms():
    ranking_filter = StudyRankingFilter()
    paper = {
        "title": "Pharmacokinetics of drug interaction",
        "abstract": "CYP3A4 inhibitors such as ketoconazole reduce clearance.",
        "drug_names": ["ketoconazole"],
    }
    ranked = ranking_filter.rank_studies([paper], query="ketoconazole interaction")
    explanation = ranking_filter.get_ranking_explanation(ranked[0], verbose=True)
    assert "pharma relevance" in explanation
    assert "ketoconazole" in explanation


def test_rank_studies_populates_pharma_details():
    ranking_filter = StudyRankingFilter()
    paper = {
        "title": "CYP3A4 interaction study",
        "abstract": "Ketoconazole shows strong pharmacokinetics interaction with CYP3A4.",
        "drug_names": ["ketoconazole"],
        "cyp_enzymes": ["CYP3A4"],
        "mesh_terms": ["humans"],
    }
    ranked = ranking_filter.rank_studies([paper], query="ketoconazole interaction")
    assert ranked[0]["ranking_score"] > 0
    assert ranked[0]["ranking_breakdown"]["pharma_relevance"] > 0


def test_quality_score_handles_randomised_spelling():
    ranking_filter = StudyRankingFilter()
    score = ranking_filter._calculate_study_quality_score(["Randomised Controlled Trial"])
    assert score >= 0.88


def test_lower_quality_synonyms_reduce_score():
    ranking_filter = StudyRankingFilter()
    score = ranking_filter._calculate_study_quality_score(["Case Report"])
    assert score <= 0.45


def test_diversity_filter_short_circuits_exact_titles():
    ranking_filter = StudyRankingFilter()
    papers = [
        {"title": "Study One", "abstract": "Group A"},
        {"title": "Study One", "abstract": "Group A"},
    ]
    filtered = ranking_filter.apply_diversity_filter(papers)
    assert len(filtered) == 1


def test_observational_study_pattern_matching():
    """Test that observational study patterns match correctly with fixed regex."""
    ranking_filter = StudyRankingFilter()

    # Test direct observational match
    score1 = ranking_filter._match_tag_with_patterns("observational study")
    assert score1 == 0.7

    # Test observational with studies
    score2 = ranking_filter._match_tag_with_patterns("observational studies")
    assert score2 == 0.7

    # Test case insensitive
    score3 = ranking_filter._match_tag_with_patterns("OBSERVATIONAL STUDY")
    assert score3 == 0.7


def test_clinical_trial_phase_numeric_matching():
    """Test that clinical trial phases accept both roman and numeric values."""
    ranking_filter = StudyRankingFilter()

    # Test roman numerals (existing behavior)
    score1 = ranking_filter._match_tag_with_patterns("phase i clinical trial")
    assert score1 == 0.78

    score2 = ranking_filter._match_tag_with_patterns("phase iii clinical trial")
    assert score2 == 0.88

    # Test numeric values (new behavior)
    score3 = ranking_filter._match_tag_with_patterns("phase 1 clinical trial")
    assert score3 == 0.78

    score4 = ranking_filter._match_tag_with_patterns("phase 3 clinical trial")
    assert score4 == 0.88

    # Test case insensitive
    score5 = ranking_filter._match_tag_with_patterns("PHASE II CLINICAL TRIAL")
    assert score5 == 0.82

    score6 = ranking_filter._match_tag_with_patterns("PHASE 2 CLINICAL TRIAL")
    assert score6 == 0.82


def test_rank_studies_boosts_direct_dual_entity_match():
    ranking_filter = StudyRankingFilter()
    papers = [
        {
            "title": "Warfarin and fluconazole interaction review",
            "abstract": "Warfarin and fluconazole are evaluated together.",
        },
        {
            "title": "Warfarin interaction review",
            "abstract": "Warfarin is evaluated alone.",
        },
    ]

    ranked = ranking_filter.rank_studies(
        papers,
        query="warfarin fluconazole interaction",
        entities=["warfarin", "fluconazole"],
        intent="drug_interaction",
    )

    assert ranked[0]["title"] == "Warfarin and fluconazole interaction review"


def test_rank_studies_single_entity_match_scores_lower_than_dual():
    ranking_filter = StudyRankingFilter()
    papers = [
        {
            "title": "Warfarin and fluconazole interaction review",
            "abstract": "Warfarin and fluconazole are evaluated together.",
        },
        {
            "title": "Warfarin interaction review",
            "abstract": "Warfarin is evaluated alone.",
        },
    ]

    ranked = ranking_filter.rank_studies(
        papers,
        query="warfarin fluconazole interaction",
        entities=["warfarin", "fluconazole"],
        intent="drug_interaction",
    )
    scores = {paper["title"]: paper["ranking_score"] for paper in ranked}

    assert scores["Warfarin interaction review"] < scores["Warfarin and fluconazole interaction review"]


def test_rank_studies_single_entity_match_uses_single_boost_only():
    ranking_filter = StudyRankingFilter()
    papers = [
        {
            "title": "Warfarin cohort report",
            "abstract": "Single entity relevance signal.",
        },
        {
            "title": "Placebo cohort report",
            "abstract": "Single entity relevance signal.",
        },
    ]

    ranked = ranking_filter.rank_studies(papers, query="", entities=["warfarin"])
    relevance_by_title = {
        paper["title"]: paper["ranking_breakdown"]["pharma_relevance"]
        for paper in ranked
    }

    assert relevance_by_title["Warfarin cohort report"] == 0.1
    assert relevance_by_title["Placebo cohort report"] == 0.0


def test_rank_studies_duplicate_and_blank_entities_do_not_escalate_boost():
    ranking_filter = StudyRankingFilter()
    papers = [
        {
            "title": "Warfarin cohort report",
            "abstract": "Single entity relevance signal.",
        },
        {
            "title": "Placebo cohort report",
            "abstract": "Single entity relevance signal.",
        },
    ]

    ranked = ranking_filter.rank_studies(
        papers,
        query="",
        entities=[" warfarin ", "WARFARIN", "", "   "],
    )
    relevance_by_title = {
        paper["title"]: paper["ranking_breakdown"]["pharma_relevance"]
        for paper in ranked
    }

    assert relevance_by_title["Warfarin cohort report"] == 0.1
    assert relevance_by_title["Placebo cohort report"] == 0.0


def test_rank_studies_entity_boost_does_not_match_token_substrings():
    ranking_filter = StudyRankingFilter()
    papers = [
        {
            "title": "Warfarin cohort report",
            "abstract": "Single entity relevance signal.",
        },
    ]

    ranked = ranking_filter.rank_studies(papers, query="", entities=["far"])

    assert ranked[0]["ranking_breakdown"]["pharma_relevance"] == 0.0


def test_rank_studies_entity_boost_does_not_match_phrase_inside_unrelated_token():
    ranking_filter = StudyRankingFilter()
    papers = [
        {
            "title": "Alpha-beta receptor cohort",
            "abstract": "Single entity relevance signal.",
        },
    ]

    ranked = ranking_filter.rank_studies(papers, query="", entities=["beta receptor"])

    assert ranked[0]["ranking_breakdown"]["pharma_relevance"] == 0.0


def test_rank_studies_two_entity_match_in_multi_entity_query_uses_dual_boost():
    ranking_filter = StudyRankingFilter()
    papers = [
        {
            "title": "Warfarin and fluconazole cohort report",
            "abstract": "Two requested entities are present.",
        },
        {
            "title": "Warfarin fluconazole aspirin cohort report",
            "abstract": "All requested entities are present.",
        },
    ]

    ranked = ranking_filter.rank_studies(
        papers,
        query="",
        entities=["warfarin", "fluconazole", "aspirin"],
    )
    relevance_by_title = {
        paper["title"]: paper["ranking_breakdown"]["pharma_relevance"]
        for paper in ranked
    }

    assert relevance_by_title["Warfarin and fluconazole cohort report"] == 0.25
    assert relevance_by_title["Warfarin fluconazole aspirin cohort report"] == 0.25


def test_rank_studies_timing_intent_boosts_pk_terms():
    ranking_filter = StudyRankingFilter()
    papers = [
        {
            "title": "Dose timing exposure study",
            "abstract": "Pharmacokinetics and exposure were measured after dosing.",
        },
        {
            "title": "Dose timing survey",
            "abstract": "Patient preferences were measured after dosing.",
        },
    ]

    ranked = ranking_filter.rank_studies(papers, query="dose timing", intent="timing")

    assert ranked[0]["title"] == "Dose timing exposure study"


def test_rank_studies_timing_intent_adds_expected_pk_boost():
    ranking_filter = StudyRankingFilter()
    papers = [
        {
            "title": "Dose timing exposure study",
            "abstract": "Pharmacokinetics and exposure were measured after dosing.",
        },
    ]

    baseline = ranking_filter.rank_studies(papers, query="dose timing")[0]["ranking_breakdown"]["pharma_relevance"]
    boosted = ranking_filter.rank_studies(papers, query="dose timing", intent="timing")[0]["ranking_breakdown"][
        "pharma_relevance"
    ]

    assert boosted - baseline == pytest.approx(0.15)


def test_rank_studies_drug_interaction_intent_adds_expected_interaction_boost():
    ranking_filter = StudyRankingFilter()
    papers = [
        {
            "title": "Interaction cohort report",
            "abstract": "Interaction signal was evaluated.",
        },
    ]

    baseline = ranking_filter.rank_studies(papers, query="")[0]["ranking_breakdown"]["pharma_relevance"]
    boosted = ranking_filter.rank_studies(papers, query="", intent="drug_interaction")[0]["ranking_breakdown"][
        "pharma_relevance"
    ]

    assert boosted - baseline == pytest.approx(0.15)
