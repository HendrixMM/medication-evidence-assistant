from __future__ import annotations

from src.evidence.normalization_selection import score_candidate, select_best_candidate


def test_score_candidate_penalizes_combo_pack_exact_matches() -> None:
    score = score_candidate(
        raw_name="acetaminophen",
        candidate_name="{30 (ascorbic acid 120 MG / cholecalciferol 400 UNT) } Pack [Kit]",
        tty="BPCK",
        source="exact",
    )

    assert score < 0.3


def test_score_candidate_penalizes_unrelated_exact_matches() -> None:
    score = score_candidate(
        raw_name="ibuprofen",
        candidate_name="famotidine",
        tty="IN",
        source="exact",
    )

    assert score < 0.3


def test_select_best_candidate_prefers_clean_ingredient_match() -> None:
    winner = select_best_candidate(
        raw_name="melatonin",
        candidates=[
            {"name": "tryptophan", "tty": "IN", "source": "exact", "rxcui": "10898"},
            {"name": "melatonin", "tty": "IN", "source": "approximate", "rxcui": "6711"},
        ],
    )

    assert winner["name"] == "melatonin"


def test_select_best_candidate_rejects_unrelated_approximate_match_without_lexical_overlap() -> None:
    winner = select_best_candidate(
        raw_name="Tylenol",
        candidates=[
            {"name": "acetaminophen", "tty": "IN", "source": "approximate", "rxcui": "161"},
            {"name": "famotidine", "tty": "IN", "source": "approximate", "rxcui": "4278"},
        ],
    )

    assert winner is None
