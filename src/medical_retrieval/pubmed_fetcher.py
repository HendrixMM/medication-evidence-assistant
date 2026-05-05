from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.medical_retrieval.pmc_mapper import PMCIDMapper
from src.medical_retrieval.source_normalizer import normalize_source_url

DEFAULT_FETCHER_CANDIDATE_CAP = 80
PUBMED_URL_TEMPLATE = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


class PubmedFetcherError(RuntimeError):
    """Expected operational failure from PubMed normalize-and-fetch."""


class _PubmedClientProtocol(Protocol):
    def efetch_xml(self, pmids: list[str]) -> str:
        ...

    def parse_efetch(self, xml_text: str) -> list[dict[str, Any]]:
        ...


@dataclass
class CandidateProvenance:
    query: str
    strategy_family: str
    snippet: str | None = None
    source_url: str | None = None
    pass_index: int | None = None


@dataclass
class NormalizedArticle:
    pmid: str
    title: str | None = None
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    journal: str | None = None
    publication_date: str | None = None
    publication_types: list[str] = field(default_factory=list)
    doi: str | None = None
    pubmed_url: str = ""
    pmcid: str | None = None
    provenance: list[CandidateProvenance] = field(default_factory=list)

    def to_tool_payload(self) -> dict[str, Any]:
        return {
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "url": self.pubmed_url,
            "title": self.title,
            "abstract": self.abstract,
            "authors": list(self.authors),
            "journal": self.journal,
            "publication_date": self.publication_date,
            "publication_types": list(self.publication_types),
            "doi": self.doi,
            "provenance": [_provenance_to_dict(p) for p in self.provenance],
        }


def _provenance_to_dict(prov: CandidateProvenance) -> dict[str, Any]:
    return {
        "query": prov.query,
        "strategy_family": prov.strategy_family,
        "snippet": prov.snippet,
        "source_url": prov.source_url,
        "pass_index": prov.pass_index,
    }


class PubmedFetcher:
    def __init__(
        self,
        pubmed_client: _PubmedClientProtocol,
        pmc_mapper: PMCIDMapper | None = None,
        max_candidates: int = DEFAULT_FETCHER_CANDIDATE_CAP,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        self.pubmed_client = pubmed_client
        self.pmc_mapper = pmc_mapper
        self.max_candidates = max_candidates
        self.fetch_count = 0

    def normalize_and_fetch(
        self,
        arguments: dict[str, Any],
        *,
        pass_index: int | None = None,
    ) -> dict[str, Any]:
        candidates = arguments.get("candidates")
        if not isinstance(candidates, list):
            raise PubmedFetcherError("candidates must be a list")

        provenance_by_pmid: dict[str, list[CandidateProvenance]] = {}
        pending_pmcids: dict[str, list[CandidateProvenance]] = {}
        unmapped: list[dict[str, Any]] = []

        for raw in candidates:
            self._classify_candidate(
                raw,
                pass_index=pass_index,
                provenance_by_pmid=provenance_by_pmid,
                pending_pmcids=pending_pmcids,
                unmapped=unmapped,
            )

        pmcid_for_pmid = self._resolve_pending_pmcids(
            pending_pmcids, provenance_by_pmid, unmapped
        )

        articles = self._fetch_articles(list(provenance_by_pmid.keys()))
        normalized_articles = self._attach_provenance(
            articles, provenance_by_pmid, pmcid_for_pmid, unmapped
        )

        return {
            "normalized": [a.to_tool_payload() for a in normalized_articles],
            "unmapped": unmapped,
            "fetch_count": self.fetch_count,
        }

    def _classify_candidate(
        self,
        raw: Any,
        *,
        pass_index: int | None,
        provenance_by_pmid: dict[str, list[CandidateProvenance]],
        pending_pmcids: dict[str, list[CandidateProvenance]],
        unmapped: list[dict[str, Any]],
    ) -> None:
        if not isinstance(raw, dict):
            unmapped.append({"url": None, "reason": "candidate_not_object"})
            return
        url = raw.get("url")
        if not isinstance(url, str) or not url:
            unmapped.append({"url": url, "reason": "missing_url"})
            return

        normalized = normalize_source_url(url)
        prov = CandidateProvenance(
            query=str(raw.get("query") or ""),
            strategy_family=str(raw.get("strategy_family") or ""),
            snippet=raw.get("snippet") if isinstance(raw.get("snippet"), str) else None,
            source_url=url,
            pass_index=pass_index,
        )

        if normalized.pmid:
            self._add_pmid_provenance(provenance_by_pmid, normalized.pmid, prov, url, unmapped)
        elif normalized.pmcid:
            pending_pmcids.setdefault(normalized.pmcid, []).append(prov)
        else:
            unmapped.append({"url": url, "reason": "unsupported_url"})

    def _add_pmid_provenance(
        self,
        provenance_by_pmid: dict[str, list[CandidateProvenance]],
        pmid: str,
        prov: CandidateProvenance,
        source_url: str,
        unmapped: list[dict[str, Any]],
    ) -> None:
        if pmid not in provenance_by_pmid:
            if len(provenance_by_pmid) >= self.max_candidates:
                unmapped.append({"url": source_url, "reason": "candidate_cap_reached"})
                return
            provenance_by_pmid[pmid] = []
        provenance_by_pmid[pmid].append(prov)

    def _resolve_pending_pmcids(
        self,
        pending_pmcids: dict[str, list[CandidateProvenance]],
        provenance_by_pmid: dict[str, list[CandidateProvenance]],
        unmapped: list[dict[str, Any]],
    ) -> dict[str, str]:
        if not pending_pmcids:
            return {}
        mappings = self._map_pmcids(list(pending_pmcids.keys()))
        pmcid_for_pmid: dict[str, str] = {}
        for pmcid, provs in pending_pmcids.items():
            pmid = mappings.get(pmcid)
            if not pmid:
                for prov in provs:
                    unmapped.append(
                        {"url": prov.source_url, "reason": "pmcid_unmapped", "pmcid": pmcid}
                    )
                continue
            pmcid_for_pmid[pmid] = pmcid
            for prov in provs:
                self._add_pmid_provenance(
                    provenance_by_pmid, pmid, prov, prov.source_url or "", unmapped
                )
        return pmcid_for_pmid

    def _attach_provenance(
        self,
        articles: dict[str, NormalizedArticle],
        provenance_by_pmid: dict[str, list[CandidateProvenance]],
        pmcid_for_pmid: dict[str, str],
        unmapped: list[dict[str, Any]],
    ) -> list[NormalizedArticle]:
        normalized_articles: list[NormalizedArticle] = []
        for pmid, provs in provenance_by_pmid.items():
            article = articles.get(pmid)
            if article is None:
                for prov in provs:
                    unmapped.append(
                        {"url": prov.source_url, "reason": "pubmed_fetch_missing", "pmid": pmid}
                    )
                continue
            article.provenance = provs
            article.pmcid = pmcid_for_pmid.get(pmid)
            normalized_articles.append(article)
        return normalized_articles

    def _map_pmcids(self, pmcids: list[str]) -> dict[str, str]:
        if not pmcids or self.pmc_mapper is None:
            return {}
        return self.pmc_mapper.pmcids_to_pmids(pmcids)

    def _fetch_articles(self, pmids: list[str]) -> dict[str, NormalizedArticle]:
        if not pmids:
            return {}
        try:
            xml_text = self.pubmed_client.efetch_xml(pmids)
            parsed = self.pubmed_client.parse_efetch(xml_text)
        except Exception as exc:
            raise PubmedFetcherError("PubMed canonical fetch failed") from exc
        self.fetch_count += len(pmids)
        result: dict[str, NormalizedArticle] = {}
        for record in parsed:
            article = _record_to_article(record)
            if article is not None:
                result[article.pmid] = article
        return result


def _record_to_article(record: dict[str, Any]) -> NormalizedArticle | None:
    pmid = str(record.get("pmid") or "").strip()
    if not pmid:
        return None
    authors_raw = record.get("authors") or ""
    if isinstance(authors_raw, list):
        authors_list = [str(a) for a in authors_raw if a]
    else:
        authors_list = [a.strip() for a in str(authors_raw).split(",") if a.strip()]
    return NormalizedArticle(
        pmid=pmid,
        title=record.get("title") or None,
        abstract=record.get("abstract") or None,
        authors=authors_list,
        journal=record.get("journal") or None,
        publication_date=record.get("publication_date") or None,
        publication_types=list(record.get("publication_types") or []),
        doi=record.get("doi") or None,
        pubmed_url=PUBMED_URL_TEMPLATE.format(pmid=pmid),
    )
