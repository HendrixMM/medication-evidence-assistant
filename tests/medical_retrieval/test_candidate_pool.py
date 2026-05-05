from __future__ import annotations

import pytest

from src.medical_retrieval.candidate_pool import CandidatePool, DEFAULT_POOL_CANDIDATE_CAP
from src.medical_retrieval.pubmed_fetcher import CandidateProvenance, NormalizedArticle


def _article(pmid: str, *, query: str = "q", strategy: str = "literal_med", pass_index: int = 0, pmcid: str | None = None) -> NormalizedArticle:
    return NormalizedArticle(
        pmid=pmid,
        title=f"Title {pmid}",
        abstract=f"Abstract {pmid}",
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        pmcid=pmcid,
        provenance=[CandidateProvenance(
            query=query,
            strategy_family=strategy,
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            pass_index=pass_index,
        )],
    )


def test_candidate_pool_dedupes_by_pmid_and_merges_provenance_across_passes() -> None:
    pool = CandidatePool()
    pool.add_articles([_article("1", query="q1", strategy="literal_med", pass_index=0)])
    pool.add_articles([_article("1", query="q2", strategy="ingredient_or_class", pass_index=1)])

    assert pool.size == 1
    payload = pool.to_payloads()[0]
    assert payload["pmid"] == "1"
    strategies = {p["strategy_family"] for p in payload["provenance"]}
    assert strategies == {"literal_med", "ingredient_or_class"}
    pass_indices = {p["pass_index"] for p in payload["provenance"]}
    assert pass_indices == {0, 1}


def test_candidate_pool_dedupes_identical_provenance_entries() -> None:
    pool = CandidatePool()
    pool.add_articles([_article("9", query="q", strategy="s", pass_index=0)])
    pool.add_articles([_article("9", query="q", strategy="s", pass_index=0)])

    assert pool.size == 1
    assert len(pool.to_payloads()[0]["provenance"]) == 1


def test_candidate_pool_preserves_pmcid_from_first_seeing_article() -> None:
    pool = CandidatePool()
    pool.add_articles([_article("3", pmcid=None, pass_index=0)])
    pool.add_articles([_article("3", pmcid="PMC42", pass_index=1)])

    payload = pool.to_payloads()[0]
    assert payload["pmcid"] == "PMC42"


def test_candidate_pool_enforces_cap_and_counts_drops() -> None:
    pool = CandidatePool(max_candidates=2)
    pool.add_articles([_article("1"), _article("2"), _article("3")])

    assert pool.size == 2
    assert pool.dropped_at_cap == 1
    assert {p["pmid"] for p in pool.to_payloads()} == {"1", "2"}


def test_candidate_pool_rejects_invalid_max_candidates() -> None:
    with pytest.raises(ValueError, match="max_candidates"):
        CandidatePool(max_candidates=0)


def test_candidate_pool_default_cap_matches_documented_safety_limit() -> None:
    assert DEFAULT_POOL_CANDIDATE_CAP == 80
