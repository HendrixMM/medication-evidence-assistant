from __future__ import annotations

from src.clients.openai_llm_client import OpenAILLMAPIError
from src.evidence.llm_scorer import LLMScorer
from src.evidence.schemas import ArticleRecord


class StubLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate(self, *args, **kwargs) -> str:
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.response


class RaisingLLMClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def generate(self, *args, **kwargs) -> str:
        raise self.error


def _articles() -> list[ArticleRecord]:
    return [
        ArticleRecord(
            pmid="1",
            title="Review of acetaminophen liver injury",
            abstract="Clinical review",
            authors=["One"],
            journal="Journal",
            publication_date="2024-01-01",
            doi=None,
            url=None,
            publication_types=["Systematic Review"],
            study_type="systematic review",
            heuristic_score=0.8,
            llm_relevance_score=None,
            llm_reason=None,
        ),
        ArticleRecord(
            pmid="2",
            title="Case report of toxicity",
            abstract="Rare case",
            authors=["Two"],
            journal="Journal",
            publication_date="2010-01-01",
            doi=None,
            url=None,
            publication_types=["Case Reports"],
            study_type="case report",
            heuristic_score=0.2,
            llm_relevance_score=None,
            llm_reason=None,
        ),
    ]


def test_llm_scorer_applies_batched_scores() -> None:
    scorer = LLMScorer(
        llm_client=StubLLMClient(
            '{"scores":[{"pmid":"1","score":0.9,"reason":"Direct safety evidence"},'
            '{"pmid":"2","score":0.2,"reason":"Weak evidence"}]}'
        ),
        model="gpt-5.4-mini",
    )

    ranked, fallback_used = scorer.score_articles(
        question="Is Tylenol safe to take?",
        intent="safety",
        normalized_drug_names=["acetaminophen"],
        articles=_articles(),
    )

    assert fallback_used is False
    assert [article.pmid for article in ranked] == ["1", "2"]
    assert ranked[0].llm_relevance_score == 0.9


def test_llm_scorer_falls_back_to_heuristic_order_on_parse_failure() -> None:
    scorer = LLMScorer(llm_client=StubLLMClient("bad-json"), model="gpt-5.4-mini")

    ranked, fallback_used = scorer.score_articles(
        question="Is Tylenol safe to take?",
        intent="safety",
        normalized_drug_names=["acetaminophen"],
        articles=_articles(),
    )

    assert fallback_used is True
    assert [article.pmid for article in ranked] == ["1", "2"]
    assert all(article.llm_relevance_score is None for article in ranked)


def test_llm_scorer_prompt_targets_direct_answerability_and_low_score_cutoff() -> None:
    llm_client = StubLLMClient(
        '{"scores":[{"pmid":"1","score":0.9,"reason":"Direct answer"},'
        '{"pmid":"2","score":0.2,"reason":"Tangential"}]}'
    )
    scorer = LLMScorer(llm_client=llm_client, model="gpt-5.4-mini")

    scorer.score_articles(
        question="Is Tylenol safe to take every day for pain?",
        intent="safety",
        normalized_drug_names=["acetaminophen"],
        articles=_articles(),
    )

    prompt = llm_client.calls[0]["kwargs"]["messages"][0]["content"]
    assert "directly answer" in prompt
    assert "named drug" in prompt
    assert "below 0.45" in prompt
    assert '"normalized_drug_names": ["acetaminophen"]' in prompt


def test_llm_scorer_does_not_swallow_unexpected_client_contract_errors() -> None:
    scorer = LLMScorer(llm_client=RaisingLLMClient(TypeError("client contract bug")), model="gpt-5.4-mini")

    try:
        scorer.score_articles(
            question="Is Tylenol safe to take?",
            intent="safety",
            normalized_drug_names=["acetaminophen"],
            articles=_articles(),
        )
    except TypeError as exc:
        assert str(exc) == "client contract bug"
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("Expected unexpected client contract error to surface")


def test_llm_scorer_does_not_swallow_client_value_errors_from_generate() -> None:
    scorer = LLMScorer(llm_client=RaisingLLMClient(ValueError("client value bug")), model="gpt-5.4-mini")

    try:
        scorer.score_articles(
            question="Is Tylenol safe to take?",
            intent="safety",
            normalized_drug_names=["acetaminophen"],
            articles=_articles(),
        )
    except ValueError as exc:
        assert str(exc) == "client value bug"
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("Expected client ValueError to surface")


def test_llm_scorer_falls_back_on_invalid_numeric_scores() -> None:
    scorer = LLMScorer(
        llm_client=StubLLMClient(
            '{"scores":[{"pmid":"1","score":2,"reason":"Out of range"},'
            '{"pmid":"2","score":NaN,"reason":"Not finite"}]}'
        ),
        model="gpt-5.4-mini",
    )

    ranked, fallback_used = scorer.score_articles(
        question="Is Tylenol safe to take?",
        intent="safety",
        normalized_drug_names=["acetaminophen"],
        articles=_articles(),
    )

    assert fallback_used is True
    assert [article.pmid for article in ranked] == ["1", "2"]
    assert all(article.llm_relevance_score is None for article in ranked)


def test_llm_scorer_falls_back_when_reason_is_missing() -> None:
    scorer = LLMScorer(
        llm_client=StubLLMClient(
            '{"scores":[{"pmid":"1","score":0.9},'
            '{"pmid":"2","score":0.2,"reason":"Weak evidence"}]}'
        ),
        model="gpt-5.4-mini",
    )

    ranked, fallback_used = scorer.score_articles(
        question="Is Tylenol safe to take?",
        intent="safety",
        normalized_drug_names=["acetaminophen"],
        articles=_articles(),
    )

    assert fallback_used is True
    assert [article.pmid for article in ranked] == ["1", "2"]
    assert all(article.llm_relevance_score is None for article in ranked)


def test_llm_scorer_falls_back_on_operational_openai_errors() -> None:
    scorer = LLMScorer(llm_client=RaisingLLMClient(OpenAILLMAPIError("transient failure")), model="gpt-5.4-mini")

    ranked, fallback_used = scorer.score_articles(
        question="Is Tylenol safe to take?",
        intent="safety",
        normalized_drug_names=["acetaminophen"],
        articles=_articles(),
    )

    assert fallback_used is True
    assert [article.pmid for article in ranked] == ["1", "2"]
    assert all(article.llm_relevance_score is None for article in ranked)


def test_llm_scorer_falls_back_when_payload_omits_expected_pmids() -> None:
    scorer = LLMScorer(
        llm_client=StubLLMClient('{"scores":[{"pmid":"1","score":0.9,"reason":"Only one score"}]}'),
        model="gpt-5.4-mini",
    )

    ranked, fallback_used = scorer.score_articles(
        question="Is Tylenol safe to take?",
        intent="safety",
        normalized_drug_names=["acetaminophen"],
        articles=_articles(),
    )

    assert fallback_used is True
    assert [article.pmid for article in ranked] == ["1", "2"]
    assert all(article.llm_relevance_score is None for article in ranked)


def test_llm_scorer_falls_back_when_payload_includes_unexpected_pmids() -> None:
    scorer = LLMScorer(
        llm_client=StubLLMClient(
            '{"scores":[{"pmid":"1","score":0.9,"reason":"Expected"},'
            '{"pmid":"999","score":0.8,"reason":"Unexpected"}]}'
        ),
        model="gpt-5.4-mini",
    )

    ranked, fallback_used = scorer.score_articles(
        question="Is Tylenol safe to take?",
        intent="safety",
        normalized_drug_names=["acetaminophen"],
        articles=_articles(),
    )

    assert fallback_used is True
    assert [article.pmid for article in ranked] == ["1", "2"]
    assert all(article.llm_relevance_score is None for article in ranked)


def test_llm_scorer_falls_back_when_payload_duplicates_pmids() -> None:
    scorer = LLMScorer(
        llm_client=StubLLMClient(
            '{"scores":[{"pmid":"1","score":0.9,"reason":"First"},'
            '{"pmid":"1","score":0.7,"reason":"Duplicate"}]}'
        ),
        model="gpt-5.4-mini",
    )

    ranked, fallback_used = scorer.score_articles(
        question="Is Tylenol safe to take?",
        intent="safety",
        normalized_drug_names=["acetaminophen"],
        articles=_articles(),
    )

    assert fallback_used is True
    assert [article.pmid for article in ranked] == ["1", "2"]
    assert all(article.llm_relevance_score is None for article in ranked)
