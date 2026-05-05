"""PubMed retrieval adapter for agent workflows."""
from __future__ import annotations

import copy
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import ClassVar

from src.pubmed_eutils_client import PubMedEutilsClient

from ..config import AgentConfig
from ..models import PlannedQuery, RetrievalResult, RetrievedArticle


class PubMedSearchTool:
    """Execute PubMed search and map results into agent models."""

    TOOL_IDENTIFIER = "pubmed_rag_agent"
    _TAGGED_CLIENT_CLASSES: ClassVar[dict[type[PubMedEutilsClient], type[PubMedEutilsClient]]] = {}

    def __init__(
        self,
        client: PubMedEutilsClient,
        max_results: int = AgentConfig.pubmed_max_results_per_query,
    ) -> None:
        self.client = client
        self.max_results = max_results
        self._tagged_client_class = self._get_tagged_client_class(type(client))

    def execute(self, query: PlannedQuery) -> RetrievalResult:
        start_time = time.perf_counter()
        tagged_client = self._client_with_tool_tag()

        try:
            pmids = tagged_client.esearch(query.pubmed_query, retmax=max(1, int(self.max_results)))
            if not pmids:
                return RetrievalResult(
                    tool_name="pubmed_search",
                    query_used=query.pubmed_query,
                    articles=[],
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    error=None,
                )

            xml_text = tagged_client.efetch_xml(pmids)
            articles = self._parse_efetch_xml(xml_text)
            return RetrievalResult(
                tool_name="pubmed_search",
                query_used=query.pubmed_query,
                articles=articles,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                error=None,
            )
        except Exception as exc:
            return RetrievalResult(
                tool_name="pubmed_search",
                query_used=query.pubmed_query,
                articles=[],
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                error=str(exc),
            )

    def _client_with_tool_tag(self) -> PubMedEutilsClient:
        tagged_client = copy.copy(self.client)
        tagged_client.__class__ = self._tagged_client_class
        return tagged_client

    def _parse_efetch_xml(self, xml_text: str) -> list[RetrievedArticle]:
        if not xml_text:
            return []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise ValueError(f"EFetch response is not valid XML: {exc}") from exc

        articles: list[RetrievedArticle] = []
        for article in root.findall("PubmedArticle"):
            medline = article.find("MedlineCitation")
            if medline is None:
                continue

            article_node = medline.find("Article")
            if article_node is None:
                continue

            articles.append(
                RetrievedArticle(
                    pmid=self._element_text(medline.find("PMID")),
                    title=self._element_text(article_node.find("ArticleTitle")) or None,
                    abstract=self._abstract_text(article_node.find("Abstract")) or None,
                    authors=self._authors_from_xml(article_node.find("AuthorList")),
                    journal=self._element_text(article_node.find("Journal/Title")) or None,
                    publication_date=self._publication_date_from_xml(
                        article_node.find("Journal/JournalIssue/PubDate")
                    ),
                    url=self._doi_url(article.find("PubmedData/ArticleIdList")),
                    publication_types=self._publication_types_from_xml(article_node.find("PublicationTypeList")),
                    ranking_score=0.0,
                    relevance_score=0.0,
                    source_system="pubmed",
                )
            )
        return articles

    @classmethod
    def _get_tagged_client_class(cls, base_class: type[PubMedEutilsClient]) -> type[PubMedEutilsClient]:
        tagged_class = cls._TAGGED_CLIENT_CLASSES.get(base_class)
        if tagged_class is not None:
            return tagged_class

        tagged_class = type(f"Tagged{base_class.__name__}", (base_class,), {})

        def _params_with_tool(
            self: PubMedEutilsClient,
            extra: dict[str, object] | None = None,
        ) -> dict[str, object]:
            params = super(tagged_class, self)._params(extra)
            params["tool"] = PubMedSearchTool.TOOL_IDENTIFIER
            return params

        tagged_class._params = _params_with_tool
        cls._TAGGED_CLIENT_CLASSES[base_class] = tagged_class
        return tagged_class

    @staticmethod
    def _element_text(node: ET.Element | None) -> str:
        if node is None:
            return ""
        return "".join(node.itertext()).strip()

    @staticmethod
    def _abstract_text(abstract_elem: ET.Element | None) -> str:
        if abstract_elem is None:
            return ""

        parts: list[str] = []
        for abstract_text in abstract_elem.findall("AbstractText"):
            label = (abstract_text.attrib.get("Label") or "").strip()
            text = "".join(abstract_text.itertext()).strip()
            if label and text:
                parts.append(f"{label}: {text}")
            elif label:
                parts.append(label)
            elif text:
                parts.append(text)
        return " ".join(parts)

    @staticmethod
    def _authors_from_xml(author_list_elem: ET.Element | None) -> list[str]:
        if author_list_elem is None:
            return []

        authors: list[str] = []
        for author in author_list_elem.findall("Author"):
            collective_name = PubMedSearchTool._element_text(author.find("CollectiveName"))
            if collective_name:
                authors.append(collective_name)
                continue

            last_name = PubMedSearchTool._element_text(author.find("LastName"))
            fore_name = (
                PubMedSearchTool._element_text(author.find("ForeName"))
                or PubMedSearchTool._element_text(author.find("FirstName"))
            )
            if last_name and fore_name:
                authors.append(f"{last_name}, {fore_name}")
            elif last_name:
                authors.append(last_name)
        return authors

    @staticmethod
    def _publication_date_from_xml(pub_date_elem: ET.Element | None) -> str | None:
        if pub_date_elem is None:
            return None

        year = PubMedSearchTool._element_text(pub_date_elem.find("Year"))
        month = PubMedSearchTool._element_text(pub_date_elem.find("Month"))
        day = PubMedSearchTool._element_text(pub_date_elem.find("Day"))
        if year and month and day:
            try:
                return datetime.strptime(f"{year} {month} {day}", "%Y %b %d").date().isoformat()
            except ValueError:
                try:
                    return datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").date().isoformat()
                except ValueError:
                    pass
        if year and month:
            try:
                return datetime.strptime(f"{year} {month}", "%Y %b").date().isoformat()
            except ValueError:
                pass
        return year or None

    @staticmethod
    def _publication_types_from_xml(publication_type_list: ET.Element | None) -> list[str]:
        if publication_type_list is None:
            return []
        publication_types = []
        for publication_type in publication_type_list.findall("PublicationType"):
            text = PubMedSearchTool._element_text(publication_type)
            if text:
                publication_types.append(text)
        return publication_types

    @staticmethod
    def _doi_url(article_id_list: ET.Element | None) -> str | None:
        if article_id_list is None:
            return None
        for article_id in article_id_list.findall("ArticleId"):
            if article_id.attrib.get("IdType") != "doi":
                continue
            doi = "".join(article_id.itertext()).strip()
            if doi:
                return f"https://doi.org/{doi}"
        return None
