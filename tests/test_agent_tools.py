"""Unit tests for agent tool adapters."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import requests

from guardrails.constants import MEDICAL_DISCLAIMER
from src.agent.config import AgentConfig
from src.agent.models import EvidenceLevel
from src.agent.models import EvidenceSummary
from src.agent.models import PlannedQuery
from src.agent.models import QueryIntent
from src.agent.models import RetrievedArticle
from src.agent.tools import MedlinePlusSearchTool
from src.agent.tools import PubMedSearchTool
from src.agent.tools import SummarizeEvidenceTool
from src.agent.tools import TranslateForConsumerTool
from src.medlineplus_client import MedlinePlusResult
from src.nvidia_llm_client import NVIDIALLMAPIError
from src.pubmed_eutils_client import PubMedEutilsClient


def _planned_query() -> PlannedQuery:
    return PlannedQuery(pubmed_query="warfarin interaction", sub_question_index=0, strategy_label="broad")


def _article(
    *,
    pmid: str = "12345",
    title: str | None = "Study title",
    abstract: str | None = "Study abstract",
    publication_date: str | None = None,
    url: str | None = None,
    ranking_score: float = 0.0,
    relevance_score: float = 0.0,
    publication_types: list[str] | None = None,
) -> RetrievedArticle:
    return RetrievedArticle(
        pmid=pmid,
        title=title,
        abstract=abstract,
        authors=["A. Author"],
        journal="Journal",
        publication_date=publication_date,
        url=url,
        publication_types=publication_types or ["Randomized Controlled Trial"],
        ranking_score=ranking_score,
        relevance_score=relevance_score,
        source_system="pubmed",
    )


def _evidence_summary(articles: list[RetrievedArticle] | None = None) -> EvidenceSummary:
    return EvidenceSummary(
        ranked_articles=articles or [_article(relevance_score=4.5)],
        evidence_level=EvidenceLevel.rct,
        key_findings=["Finding"],
        convergent_findings=["Convergent"],
        divergent_findings=["Divergent"],
        drug_interactions=[],
        pk_parameters={},
        research_gaps=["Gap"],
    )


def _pubmed_efetch_xml(
    *,
    pmid: str = "12345",
    title: str = "A study",
    abstract: str = "Abstract text",
    authors: list[tuple[str, str]] | None = None,
    journal: str = "Journal",
    year: str = "2023",
    month: str = "Jan",
    day: str = "15",
    publication_types: list[str] | str | None = None,
    doi: str | None = None,
) -> str:
    author_entries = authors or [("Author", "One"), ("Author", "Two")]
    publication_type_values = publication_types if isinstance(publication_types, list) else [publication_types or "Review"]
    author_xml = "".join(
        f"<Author><LastName>{last_name}</LastName><ForeName>{fore_name}</ForeName></Author>"
        for last_name, fore_name in author_entries
    )
    publication_type_xml = "".join(
        f"<PublicationType>{publication_type}</PublicationType>"
        for publication_type in publication_type_values
        if publication_type
    )
    article_id_xml = (
        f'<ArticleId IdType="doi">{doi}</ArticleId>'
        if doi
        else ""
    )
    return f"""
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>{pmid}</PMID>
      <Article>
        <ArticleTitle>{title}</ArticleTitle>
        <Abstract><AbstractText>{abstract}</AbstractText></Abstract>
        <AuthorList>{author_xml}</AuthorList>
        <Journal>
          <Title>{journal}</Title>
          <JournalIssue>
            <PubDate>
              <Year>{year}</Year>
              <Month>{month}</Month>
              <Day>{day}</Day>
            </PubDate>
          </JournalIssue>
        </Journal>
        <PublicationTypeList>{publication_type_xml}</PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>{article_id_xml}</ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
""".strip()


class TestPubMedSearchTool:
    def test_execute_does_not_mutate_injected_client_params(self) -> None:
        client = PubMedEutilsClient(base_url="https://example.test", email="agent@example.com")
        tool = PubMedSearchTool(client=client, max_results=5)

        with patch("src.pubmed_eutils_client.requests.get") as mock_get:
            esearch_response = Mock()
            esearch_response.json.return_value = {"esearchresult": {"idlist": ["12345"]}}
            esearch_response.raise_for_status.return_value = None
            efetch_response = Mock()
            efetch_response.text = _pubmed_efetch_xml()
            efetch_response.raise_for_status.return_value = None
            mock_get.side_effect = [esearch_response, efetch_response]

            result = tool.execute(_planned_query())

        assert result.error is None
        assert result.articles[0].pmid == "12345"
        assert "_params" not in client.__dict__

    def test_execute_returns_mapped_articles(self) -> None:
        client = PubMedEutilsClient(base_url="https://example.test", email="agent@example.com")
        tool = PubMedSearchTool(client=client, max_results=5)

        with patch("src.pubmed_eutils_client.requests.get") as mock_get:
            esearch_response = Mock()
            esearch_response.json.return_value = {"esearchresult": {"idlist": ["12345"]}}
            esearch_response.raise_for_status.return_value = None
            efetch_response = Mock()
            efetch_response.text = _pubmed_efetch_xml(doi="10.1000/test-doi")
            efetch_response.raise_for_status.return_value = None
            mock_get.side_effect = [esearch_response, efetch_response]

            result = tool.execute(_planned_query())

        assert len(result.articles) == 1
        assert result.articles[0].pmid == "12345"
        assert result.articles[0].source_system == "pubmed"
        assert result.articles[0].authors == ["Author, One", "Author, Two"]
        assert result.articles[0].publication_date == "2023-01-15"
        assert result.articles[0].url == "https://doi.org/10.1000/test-doi"
        assert mock_get.call_args_list[0].kwargs["params"]["tool"] == "pubmed_rag_agent"
        assert mock_get.call_args_list[1].kwargs["params"]["tool"] == "pubmed_rag_agent"
        assert mock_get.call_count == 2

    def test_execute_wraps_scalar_publication_types(self) -> None:
        client = PubMedEutilsClient(base_url="https://example.test", email="agent@example.com")
        tool = PubMedSearchTool(client=client, max_results=5)

        with patch("src.pubmed_eutils_client.requests.get") as mock_get:
            esearch_response = Mock()
            esearch_response.json.return_value = {"esearchresult": {"idlist": ["12345"]}}
            esearch_response.raise_for_status.return_value = None
            efetch_response = Mock()
            efetch_response.text = _pubmed_efetch_xml(publication_types="Review")
            efetch_response.raise_for_status.return_value = None
            mock_get.side_effect = [esearch_response, efetch_response]

            result = tool.execute(_planned_query())

        assert result.error is None
        assert result.articles[0].publication_types == ["Review"]

    def test_execute_honors_runtime_client_credential_updates(self) -> None:
        client = PubMedEutilsClient(base_url="https://example.test", email="initial@example.com", api_key="old-key")
        tool = PubMedSearchTool(client=client, max_results=5)
        client.email = "rotated@example.com"
        client.api_key = "new-key"

        with patch("src.pubmed_eutils_client.requests.get") as mock_get:
            esearch_response = Mock()
            esearch_response.json.return_value = {"esearchresult": {"idlist": []}}
            esearch_response.raise_for_status.return_value = None
            mock_get.return_value = esearch_response

            result = tool.execute(_planned_query())

        assert result.error is None
        assert mock_get.call_args.kwargs["params"]["email"] == "rotated@example.com"
        assert mock_get.call_args.kwargs["params"]["api_key"] == "new-key"

    def test_execute_parallel_calls_do_not_mutate_shared_client_state(self) -> None:
        client = PubMedEutilsClient(base_url="https://example.test", email="agent@example.com")
        tool = PubMedSearchTool(client=client, max_results=5)

        def _fake_get(url, params=None, timeout=30):
            assert params is not None
            assert params["tool"] == "pubmed_rag_agent"
            response = Mock()
            response.raise_for_status.return_value = None
            if url.endswith("esearch.fcgi"):
                response.json.return_value = {"esearchresult": {"idlist": ["12345"]}}
            else:
                response.text = _pubmed_efetch_xml(authors=[("Smith", "John"), ("Doe", "Jane")])
            return response

        with patch("src.pubmed_eutils_client.requests.get", side_effect=_fake_get):
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(tool.execute, [_planned_query() for _ in range(4)]))

        assert all(result.error is None for result in results)
        assert all(result.articles[0].authors == ["Smith, John", "Doe, Jane"] for result in results)
        params_after = client._params({"db": "pubmed"})
        assert params_after.get("tool") is None
        assert params_after["db"] == "pubmed"

    def test_client_with_tool_tag_reuses_cached_tagged_subclass(self) -> None:
        client = PubMedEutilsClient(base_url="https://example.test", email="agent@example.com")
        tool = PubMedSearchTool(client=client, max_results=5)

        first_tagged_client = tool._client_with_tool_tag()
        second_tagged_client = tool._client_with_tool_tag()

        assert type(first_tagged_client) is type(second_tagged_client)
        assert type(first_tagged_client).__name__ == "TaggedPubMedEutilsClient"
        assert type(first_tagged_client) is not type(client)

    def test_execute_returns_empty_on_no_pmids(self) -> None:
        client = PubMedEutilsClient(base_url="https://example.test")
        tool = PubMedSearchTool(client=client, max_results=5)

        with patch("src.pubmed_eutils_client.requests.get") as mock_get:
            esearch_response = Mock()
            esearch_response.json.return_value = {"esearchresult": {"idlist": []}}
            esearch_response.raise_for_status.return_value = None
            mock_get.return_value = esearch_response

            result = tool.execute(_planned_query())

        assert result.articles == []
        assert result.error is None


class TestMedlinePlusSearchTool:
    def test_execute_maps_medlineplus_results_to_retrieved_articles(self) -> None:
        client = Mock()
        client.search.return_value = [
            MedlinePlusResult(
                identifier="medlineplus:painrelievers",
                title="Pain Relievers",
                summary="Acetaminophen and ibuprofen can help with pain or fever.",
                url="https://medlineplus.gov/painrelievers.html",
                sections={
                    "groupName": ["Drug Therapy"],
                    "snippet": ["Pain relievers include acetaminophen and ibuprofen."],
                },
            )
        ]
        tool = MedlinePlusSearchTool(client=client, max_results=2)

        result = tool.execute(PlannedQuery(pubmed_query='("Tylenol"[tiab]) AND Humans[Mesh]', sub_question_index=0, strategy_label="broad"))

        assert result.error is None
        assert result.tool_name == "medlineplus_search"
        assert result.query_used == "acetaminophen"
        assert result.articles[0].source_system == "medlineplus"
        assert result.articles[0].pmid == "medlineplus:painrelievers"
        assert result.articles[0].title == "Pain Relievers"
        assert result.articles[0].journal == "MedlinePlus"
        assert result.articles[0].url == "https://medlineplus.gov/painrelievers.html"
        assert "Pain relievers include acetaminophen" in result.articles[0].abstract
        client.search.assert_called_once_with("acetaminophen", max_results=2)

    def test_execute_captures_network_error(self) -> None:
        client = PubMedEutilsClient(base_url="https://example.test")
        tool = PubMedSearchTool(client=client, max_results=5)

        with patch(
            "src.pubmed_eutils_client.requests.get",
            side_effect=requests.RequestException("network"),
        ):
            result = tool.execute(_planned_query())

        assert result.error is not None
        assert result.articles == []

    def test_execute_records_execution_time(self) -> None:
        client = PubMedEutilsClient(base_url="https://example.test")
        tool = PubMedSearchTool(client=client, max_results=5)

        with patch("src.pubmed_eutils_client.requests.get") as mock_get:
            esearch_response = Mock()
            esearch_response.json.return_value = {"esearchresult": {"idlist": []}}
            esearch_response.raise_for_status.return_value = None
            mock_get.return_value = esearch_response

            result = tool.execute(_planned_query())

        assert result.execution_time_ms >= 0

    def test_execute_uses_client_subclass_public_methods_and_param_overrides(self) -> None:
        esearch_types: list[str] = []
        efetch_types: list[str] = []

        class CustomClient(PubMedEutilsClient):
            def esearch(self, query: str, retmax: int = 30) -> list[str]:
                esearch_types.append(type(self).__name__)
                return super().esearch(query, retmax=retmax)

            def efetch_xml(self, pmids: list[str]) -> str:
                efetch_types.append(type(self).__name__)
                return super().efetch_xml(pmids)

            def _params(self, extra=None):
                params = super()._params(extra)
                params["custom"] = "value"
                return params

        client = CustomClient(base_url="https://example.test", email="agent@example.com")
        tool = PubMedSearchTool(client=client, max_results=5)

        with patch("src.pubmed_eutils_client.requests.get") as mock_get:
            esearch_response = Mock()
            esearch_response.json.return_value = {"esearchresult": {"idlist": ["12345"]}}
            esearch_response.raise_for_status.return_value = None
            efetch_response = Mock()
            efetch_response.text = _pubmed_efetch_xml()
            efetch_response.raise_for_status.return_value = None
            mock_get.side_effect = [esearch_response, efetch_response]

            result = tool.execute(_planned_query())

        assert result.error is None
        assert esearch_types == ["TaggedCustomClient"]
        assert efetch_types == ["TaggedCustomClient"]
        assert mock_get.call_args_list[0].kwargs["params"]["tool"] == "pubmed_rag_agent"
        assert mock_get.call_args_list[0].kwargs["params"]["custom"] == "value"
        assert mock_get.call_args_list[1].kwargs["params"]["tool"] == "pubmed_rag_agent"
        assert mock_get.call_args_list[1].kwargs["params"]["custom"] == "value"

    def test_execute_returns_error_on_malformed_xml(self) -> None:
        client = PubMedEutilsClient(base_url="https://example.test")
        tool = PubMedSearchTool(client=client, max_results=5)

        with patch("src.pubmed_eutils_client.requests.get") as mock_get:
            esearch_response = Mock()
            esearch_response.json.return_value = {"esearchresult": {"idlist": ["12345"]}}
            esearch_response.raise_for_status.return_value = None
            efetch_response = Mock()
            efetch_response.text = "<<not valid xml>>"
            efetch_response.raise_for_status.return_value = None
            mock_get.side_effect = [esearch_response, efetch_response]

            result = tool.execute(_planned_query())

        assert result.error is not None
        assert result.articles == []


class TestSummarizeEvidenceTool:
    def test_execute_calls_ranking_filter(self) -> None:
        ranking_filter = Mock()
        ranking_filter.rank_studies.return_value = [{"pmid": "12345", "ranking_score": 1.0}]
        synthesis_engine = Mock()
        synthesis_engine.generate_meta_summary.return_value = {
            "key_findings": [],
            "comparative_analysis": {},
            "research_gaps": [],
            "evidence_synthesis": {},
        }
        ddi_pk_processor = Mock()
        ddi_pk_processor.analyze_drug_interactions.return_value = {}
        relevance_scorer = Mock()
        relevance_scorer.score_sources.return_value = []
        tool = SummarizeEvidenceTool(
            ranking_filter=ranking_filter,
            synthesis_engine=synthesis_engine,
            ddi_pk_processor=ddi_pk_processor,
            relevance_scorer=relevance_scorer,
            config=AgentConfig(),
        )

        tool.execute([_article()], "query", QueryIntent.side_effects, [])

        ranking_filter.rank_studies.assert_called_once()
        ranker_input = ranking_filter.rank_studies.call_args.args[0]
        assert ranker_input[0]["study_types"] == ["Randomized Controlled Trial"]
        call_kwargs = ranking_filter.rank_studies.call_args.kwargs
        assert call_kwargs.get("entities") == []
        assert call_kwargs.get("intent") == "side_effects"

    def test_execute_forwards_entities_and_intent_to_rank_studies(self) -> None:
        ranking_filter = Mock()
        ranking_filter.rank_studies.return_value = [{"pmid": "12345", "ranking_score": 1.0}]
        synthesis_engine = Mock()
        synthesis_engine.generate_meta_summary.return_value = {
            "key_findings": [],
            "comparative_analysis": {},
            "research_gaps": [],
            "evidence_synthesis": {},
        }
        ddi_pk_processor = Mock()
        ddi_pk_processor.analyze_drug_interactions.return_value = {}
        relevance_scorer = Mock()
        relevance_scorer.score_sources.return_value = []
        tool = SummarizeEvidenceTool(
            ranking_filter=ranking_filter,
            synthesis_engine=synthesis_engine,
            ddi_pk_processor=ddi_pk_processor,
            relevance_scorer=relevance_scorer,
            config=AgentConfig(),
        )

        tool.execute(
            [_article()],
            "query",
            QueryIntent.drug_interaction,
            ["warfarin", "fluconazole"],
        )

        call_kwargs = ranking_filter.rank_studies.call_args.kwargs
        assert call_kwargs["entities"] == ["warfarin", "fluconazole"]
        assert call_kwargs["intent"] == "drug_interaction"

    def test_execute_skips_ddi_for_non_interaction_intent(self) -> None:
        ranking_filter = Mock()
        ranking_filter.rank_studies.return_value = [{"pmid": "12345", "ranking_score": 1.0}]
        synthesis_engine = Mock()
        synthesis_engine.generate_meta_summary.return_value = {
            "key_findings": [],
            "comparative_analysis": {},
            "research_gaps": [],
            "evidence_synthesis": {},
        }
        ddi_pk_processor = Mock()
        tool = SummarizeEvidenceTool(
            ranking_filter=ranking_filter,
            synthesis_engine=synthesis_engine,
            ddi_pk_processor=ddi_pk_processor,
            relevance_scorer=None,
            config=AgentConfig(),
        )

        tool.execute([_article()], "query", QueryIntent.side_effects, ["warfarin", "fluconazole"])

        ddi_pk_processor.analyze_drug_interactions.assert_not_called()

    def test_execute_calls_ddi_for_drug_interaction_intent(self) -> None:
        ranking_filter = Mock()
        ranking_filter.rank_studies.return_value = [{"pmid": "12345", "ranking_score": 1.0}]
        synthesis_engine = Mock()
        synthesis_engine.generate_meta_summary.return_value = {
            "key_findings": [],
            "comparative_analysis": {},
            "research_gaps": [],
            "evidence_synthesis": {},
        }
        ddi_pk_processor = Mock()
        ddi_pk_processor.analyze_drug_interactions.return_value = {"pk_parameters": {"auc": "2x"}}
        tool = SummarizeEvidenceTool(
            ranking_filter=ranking_filter,
            synthesis_engine=synthesis_engine,
            ddi_pk_processor=ddi_pk_processor,
            relevance_scorer=None,
            config=AgentConfig(),
        )

        tool.execute([_article()], "query", QueryIntent.drug_interaction, ["warfarin", "fluconazole"])

        ddi_pk_processor.analyze_drug_interactions.assert_called_once()
        args = ddi_pk_processor.analyze_drug_interactions.call_args.args
        assert args[1] == "warfarin"
        assert args[2] == ["fluconazole"]

    def test_execute_skips_relevance_scoring_when_disabled(self) -> None:
        ranking_filter = Mock()
        ranking_filter.rank_studies.return_value = [{"pmid": "12345", "ranking_score": 1.0}]
        synthesis_engine = Mock()
        synthesis_engine.generate_meta_summary.return_value = {
            "key_findings": [],
            "comparative_analysis": {},
            "research_gaps": [],
            "evidence_synthesis": {},
        }
        relevance_scorer = Mock()
        tool = SummarizeEvidenceTool(
            ranking_filter=ranking_filter,
            synthesis_engine=synthesis_engine,
            ddi_pk_processor=Mock(),
            relevance_scorer=relevance_scorer,
            config=AgentConfig(enable_source_relevance_scoring=False),
        )

        tool.execute([_article()], "query", QueryIntent.side_effects, [])

        relevance_scorer.score_sources.assert_not_called()

    def test_execute_returns_minimal_summary_on_exception(self) -> None:
        ranking_filter = Mock()
        ranking_filter.rank_studies.side_effect = RuntimeError("boom")
        tool = SummarizeEvidenceTool(
            ranking_filter=ranking_filter,
            synthesis_engine=Mock(),
            ddi_pk_processor=Mock(),
            relevance_scorer=None,
            config=AgentConfig(),
        )

        result = tool.execute([_article()], "query", QueryIntent.side_effects, [])

        assert result.evidence_level == EvidenceLevel.insufficient

    def test_execute_normalizes_dict_findings_without_fallback(self) -> None:
        ranking_filter = Mock()
        ranking_filter.rank_studies.return_value = [
            {
                "pmid": "12345",
                "ranking_score": 1.0,
                "study_types": ["Randomized Controlled Trial"],
            }
        ]
        synthesis_engine = Mock()
        synthesis_engine.generate_meta_summary.return_value = {
            "key_findings": ["Finding"],
            "comparative_analysis": {
                "convergent_findings": [{"finding": "Shared efficacy finding"}],
                "divergent_findings": [{"summary": "Conflicting safety signal"}],
            },
            "research_gaps": [],
            "evidence_synthesis": {"highest_level": "Level 2"},
        }
        tool = SummarizeEvidenceTool(
            ranking_filter=ranking_filter,
            synthesis_engine=synthesis_engine,
            ddi_pk_processor=Mock(),
            relevance_scorer=None,
            config=AgentConfig(),
        )

        result = tool.execute([_article()], "query", QueryIntent.side_effects, [])

        assert result.evidence_level == EvidenceLevel.rct
        assert result.key_findings == ["Finding"]
        assert result.convergent_findings == ["Shared efficacy finding"]
        assert result.divergent_findings == ["Conflicting safety signal"]

    def test_execute_extracts_text_fields_from_structured_key_findings(self) -> None:
        ranking_filter = Mock()
        ranking_filter.rank_studies.return_value = [
            {
                "pmid": "12345",
                "ranking_score": 1.0,
                "study_types": ["Randomized Controlled Trial"],
            }
        ]
        synthesis_engine = Mock()
        synthesis_engine.generate_meta_summary.return_value = {
            "key_findings": [
                {"finding": "Primary efficacy improved"},
                {"summary": "Safety was acceptable"},
                {"text": "No major interaction observed"},
            ],
            "comparative_analysis": {},
            "research_gaps": [],
            "evidence_synthesis": {"highest_level": "Level 2"},
        }
        tool = SummarizeEvidenceTool(
            ranking_filter=ranking_filter,
            synthesis_engine=synthesis_engine,
            ddi_pk_processor=Mock(),
            relevance_scorer=None,
            config=AgentConfig(),
        )

        result = tool.execute([_article()], "query", QueryIntent.side_effects, [])

        assert result.key_findings == [
            "Primary efficacy improved",
            "Safety was acceptable",
            "No major interaction observed",
        ]

    def test_execute_preserves_original_articles_when_ranker_returns_unmatched_pmids(self) -> None:
        ranking_filter = Mock()
        ranking_filter.rank_studies.return_value = [{"pmid": "missing-pmid", "ranking_score": 9.0}]
        synthesis_engine = Mock()
        synthesis_engine.generate_meta_summary.return_value = {
            "key_findings": [],
            "comparative_analysis": {},
            "research_gaps": [],
            "evidence_synthesis": {},
        }
        tool = SummarizeEvidenceTool(
            ranking_filter=ranking_filter,
            synthesis_engine=synthesis_engine,
            ddi_pk_processor=Mock(),
            relevance_scorer=None,
            config=AgentConfig(),
        )

        articles = [_article(pmid="12345"), _article(pmid="67890")]
        result = tool.execute(articles, "query", QueryIntent.side_effects, [])

        assert [article.pmid for article in result.ranked_articles] == ["12345", "67890"]
        assert all(article.ranking_score == 0.0 for article in result.ranked_articles)

    def test_execute_preserves_unmatched_articles_after_ranked_matches(self) -> None:
        ranking_filter = Mock()
        ranking_filter.rank_studies.return_value = [{"pmid": "67890", "ranking_score": 9.0}]
        synthesis_engine = Mock()
        synthesis_engine.generate_meta_summary.return_value = {
            "key_findings": [],
            "comparative_analysis": {},
            "research_gaps": [],
            "evidence_synthesis": {},
        }
        tool = SummarizeEvidenceTool(
            ranking_filter=ranking_filter,
            synthesis_engine=synthesis_engine,
            ddi_pk_processor=Mock(),
            relevance_scorer=None,
            config=AgentConfig(),
        )

        articles = [_article(pmid="12345"), _article(pmid="67890"), _article(pmid="13579")]
        result = tool.execute(articles, "query", QueryIntent.side_effects, [])

        assert [article.pmid for article in result.ranked_articles] == ["67890", "12345", "13579"]
        assert result.ranked_articles[0].ranking_score == 9.0
        assert result.ranked_articles[1].ranking_score == 0.0
        assert result.ranked_articles[2].ranking_score == 0.0

    def test_execute_normalizes_real_comparative_analysis_schema(self) -> None:
        ranking_filter = Mock()
        ranking_filter.rank_studies.return_value = [
            {
                "pmid": "12345",
                "ranking_score": 1.0,
                "study_types": ["Randomized Controlled Trial"],
            }
        ]
        synthesis_engine = Mock()
        synthesis_engine.generate_meta_summary.return_value = {
            "key_findings": [],
            "comparative_analysis": {
                "convergent_findings": [
                    {
                        "drug": "warfarin",
                        "convergent_finding": "Consistent increase in exposure",
                        "papers_count": 3,
                    }
                ],
                "divergent_findings": [
                    {
                        "entity": "INR response",
                        "directions": ["increase", "decrease"],
                        "papers": ["1", "2"],
                        "sample_findings": ["Increase reported in older adults", "Decrease in one cohort"],
                    }
                ],
            },
            "research_gaps": [],
            "evidence_synthesis": {"highest_level": "Level 2"},
        }
        tool = SummarizeEvidenceTool(
            ranking_filter=ranking_filter,
            synthesis_engine=synthesis_engine,
            ddi_pk_processor=Mock(),
            relevance_scorer=None,
            config=AgentConfig(),
        )

        result = tool.execute([_article()], "query", QueryIntent.side_effects, [])

        assert result.convergent_findings == ["For warfarin: Consistent increase in exposure (3 studies)"]
        assert result.divergent_findings == [
            "Divergent evidence for INR response: increase, decrease across 2 studies "
            "(e.g., Increase reported in older adults; Decrease in one cohort)"
        ]

    def test_execute_propagates_publication_year_to_ranking_input(self) -> None:
        ranking_filter = Mock()
        ranking_filter.rank_studies.return_value = [{"pmid": "12345", "ranking_score": 1.0}]
        synthesis_engine = Mock()
        synthesis_engine.generate_meta_summary.return_value = {
            "key_findings": [],
            "comparative_analysis": {},
            "research_gaps": [],
            "evidence_synthesis": {},
        }
        tool = SummarizeEvidenceTool(
            ranking_filter=ranking_filter,
            synthesis_engine=synthesis_engine,
            ddi_pk_processor=Mock(),
            relevance_scorer=None,
            config=AgentConfig(),
        )

        tool.execute([_article(publication_date="2021-03-04")], "query", QueryIntent.side_effects, [])

        ranked_input = ranking_filter.rank_studies.call_args.args[0][0]
        assert ranked_input["publication_year"] == 2021
        assert ranked_input["year"] == 2021
        assert ranked_input["metadata"]["publication_year"] == 2021
        assert ranked_input["metadata"]["year"] == 2021

    def test_execute_normalizes_scalar_research_gaps(self) -> None:
        ranking_filter = Mock()
        ranking_filter.rank_studies.return_value = [{"pmid": "12345", "ranking_score": 1.0}]
        synthesis_engine = Mock()
        synthesis_engine.generate_meta_summary.return_value = {
            "key_findings": [],
            "comparative_analysis": {},
            "research_gaps": "Need larger prospective studies",
            "evidence_synthesis": {},
        }
        tool = SummarizeEvidenceTool(
            ranking_filter=ranking_filter,
            synthesis_engine=synthesis_engine,
            ddi_pk_processor=Mock(),
            relevance_scorer=None,
            config=AgentConfig(),
        )

        result = tool.execute([_article()], "query", QueryIntent.side_effects, [])

        assert result.research_gaps == ["Need larger prospective studies"]

    def test_execute_normalizes_scalar_findings_without_splitting_characters(self) -> None:
        ranking_filter = Mock()
        ranking_filter.rank_studies.return_value = [{"pmid": "12345", "ranking_score": 1.0}]
        synthesis_engine = Mock()
        synthesis_engine.generate_meta_summary.return_value = {
            "key_findings": "Single key finding",
            "comparative_analysis": {
                "convergent_findings": "Shared signal",
                "divergent_findings": "Conflicting signal",
            },
            "research_gaps": [],
            "evidence_synthesis": {},
        }
        tool = SummarizeEvidenceTool(
            ranking_filter=ranking_filter,
            synthesis_engine=synthesis_engine,
            ddi_pk_processor=Mock(),
            relevance_scorer=None,
            config=AgentConfig(),
        )

        result = tool.execute([_article()], "query", QueryIntent.side_effects, [])

        assert result.key_findings == ["Single key finding"]
        assert result.convergent_findings == ["Shared signal"]
        assert result.divergent_findings == ["Conflicting signal"]


class TestTranslateForConsumerTool:
    def test_init_does_not_load_config_from_env_implicitly(self) -> None:
        llm_client = Mock()

        with patch("src.agent.tools.translate_for_consumer.AgentConfig.from_env", side_effect=AssertionError):
            tool = TranslateForConsumerTool(llm_client=llm_client)

        assert isinstance(tool.config, AgentConfig)

    def test_execute_returns_consumer_answer_with_disclaimer(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = (
            '{"answer": "Summary", "uncertainties": ["One"], "safety_warnings": ["Two"], "verdict": "ok"}'
        )
        tool = TranslateForConsumerTool(llm_client=llm_client)

        result = tool.execute("What are the side effects?", QueryIntent.side_effects, _evidence_summary())

        assert result.disclaimer == MEDICAL_DISCLAIMER

    def test_execute_falls_back_on_llm_error(self) -> None:
        llm_client = Mock()
        llm_client.generate.side_effect = NVIDIALLMAPIError("failure")
        tool = TranslateForConsumerTool(llm_client=llm_client)

        result = tool.execute("What are the side effects?", QueryIntent.side_effects, _evidence_summary())

        assert result.answer.startswith("I found relevant MedlinePlus pages")
        assert result.sources

    def test_execute_falls_back_to_empty_sources_when_source_mapping_fails(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = '{"answer": "Summary", "uncertainties": [], "safety_warnings": []}'
        tool = TranslateForConsumerTool(llm_client=llm_client)

        with patch.object(tool, "_build_sources", side_effect=RuntimeError("boom")):
            result = tool.execute("What are the side effects?", QueryIntent.side_effects, _evidence_summary())

        assert result.answer == "Summary"
        assert result.sources == []

    def test_execute_uses_intent_prompt_template(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = '{"answer": "Summary", "uncertainties": [], "safety_warnings": []}'
        tool = TranslateForConsumerTool(llm_client=llm_client)

        tool.execute("Will this interact?", QueryIntent.drug_interaction, _evidence_summary())

        messages = llm_client.generate.call_args.kwargs["messages"]
        assert "Will this interact?" in messages[1]["content"]

    def test_execute_handles_non_json_llm_response(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = "plain text answer"
        tool = TranslateForConsumerTool(llm_client=llm_client)

        result = tool.execute("What are the side effects?", QueryIntent.side_effects, _evidence_summary())

        assert result.answer == "plain text answer"
        assert result.uncertainties == []

    def test_execute_handles_valid_non_dict_json_response_without_fallback(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = '["array-wrapped answer"]'
        tool = TranslateForConsumerTool(llm_client=llm_client)

        result = tool.execute("What are the side effects?", QueryIntent.side_effects, _evidence_summary())

        assert result.answer == "array-wrapped answer"
        assert result.sources
        assert result.disclaimer == MEDICAL_DISCLAIMER

    def test_execute_uses_non_empty_fallback_when_json_omits_answer(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = (
            '{"summary": "Structured summary without answer", "uncertainties": [], "safety_warnings": []}'
        )
        tool = TranslateForConsumerTool(llm_client=llm_client)

        result = tool.execute("What are the side effects?", QueryIntent.side_effects, _evidence_summary())

        assert result.answer == "Structured summary without answer"
        assert result.answer

    def test_execute_wraps_scalar_warning_fields(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = (
            '{"answer": "Summary", "uncertainties": "Needs replication", '
            '"safety_warnings": "Consult your clinician", "verdict": "ok"}'
        )
        tool = TranslateForConsumerTool(llm_client=llm_client)

        result = tool.execute("What are the side effects?", QueryIntent.side_effects, _evidence_summary())

        assert result.uncertainties == ["Needs replication"]
        assert result.safety_warnings == ["Consult your clinician"]

    def test_execute_selects_unknown_prompt_for_mechanism_intent(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = '{"answer": "Summary", "uncertainties": [], "safety_warnings": []}'
        tool = TranslateForConsumerTool(llm_client=llm_client)

        result = tool.execute("How does it work?", QueryIntent.mechanism, _evidence_summary())

        assert result.answer == "Summary"

    def test_execute_honors_injected_source_threshold(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = '{"answer": "Summary", "uncertainties": [], "safety_warnings": []}'
        tool = TranslateForConsumerTool(
            llm_client=llm_client,
            config=AgentConfig(source_relevance_threshold=4.5),
        )
        evidence = _evidence_summary(
            [
                _article(pmid="1", relevance_score=4.6),
                _article(pmid="2", relevance_score=4.4),
            ]
        )

        result = tool.execute("What are the side effects?", QueryIntent.side_effects, evidence)

        assert [source.pmid for source in result.sources] == ["1"]

    def test_build_sources_falls_back_in_relevance_order(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = '{"answer": "Summary", "uncertainties": [], "safety_warnings": []}'
        tool = TranslateForConsumerTool(
            llm_client=llm_client,
            config=AgentConfig(source_relevance_threshold=10.0),
        )
        evidence = _evidence_summary(
            [
                _article(pmid="1", relevance_score=1.0),
                _article(pmid="2", relevance_score=4.0),
                _article(pmid="3", relevance_score=2.5),
            ]
        )

        result = tool.execute("What are the side effects?", QueryIntent.side_effects, evidence)

        assert [source.pmid for source in result.sources] == ["2", "3", "1"]

    def test_build_sources_propagates_pubmed_year_and_url(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = '{"answer": "Summary", "uncertainties": [], "safety_warnings": []}'
        tool = TranslateForConsumerTool(llm_client=llm_client)
        evidence = _evidence_summary(
            [
                _article(
                    pmid="1",
                    relevance_score=4.9,
                    publication_date="2023-01-15",
                    url="https://example.test/articles/1",
                )
            ]
        )

        result = tool.execute("What are the side effects?", QueryIntent.side_effects, evidence)

        assert result.sources[0].year == 2023
        assert result.sources[0].url == "https://example.test/articles/1"

    def test_execute_generates_research_informed_answer_when_no_articles(self) -> None:
        llm_client = Mock()
        llm_client.generate.return_value = (
            '{"answer": "Tylenol and Advil are common options for pain or fever, but the better choice depends on '
            'stomach, kidney, liver, bleeding, and medication risks.", "uncertainties": ["No PubMed sources attached"], '
            '"safety_warnings": ["Ask a clinician or pharmacist for personal dosing advice."]}'
        )
        tool = TranslateForConsumerTool(llm_client=llm_client)
        empty_evidence = EvidenceSummary(
            ranked_articles=[],
            evidence_level=EvidenceLevel.rct,
            key_findings=[],
            convergent_findings=[],
            divergent_findings=[],
            drug_interactions=[],
            pk_parameters={},
            research_gaps=[],
        )

        result = tool.execute("What are the side effects?", QueryIntent.side_effects, empty_evidence)

        llm_client.generate.assert_called_once()
        assert "common options" in result.answer
        assert result.disclaimer == MEDICAL_DISCLAIMER
        assert result.safety_warnings
        assert result.sources == []
