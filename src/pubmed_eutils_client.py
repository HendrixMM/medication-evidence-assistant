"""
PubMed E-utilities Client (ESearch, EFetch, ELink)

Primary, free alternative to Apify for PubMed metadata and abstracts.
Maps responses to the project's unified article schema.
"""
from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        return default
    return str(v).strip()


class PubMedEutilsClient:
    """Minimal client for PubMed E-utilities with unified schema mapping."""

    def __init__(self, base_url: str | None = None, email: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (
            base_url or _env("PUBMED_EUTILS_BASE_URL", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
        ).rstrip("/")
        self.email = email or _env("PUBMED_EMAIL")
        self.api_key = api_key or _env("PUBMED_EUTILS_API_KEY")

    def _params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        if extra:
            params.update(extra)
        return params

    def esearch(self, query: str, retmax: int = 30) -> list[str]:
        url = f"{self.base_url}/esearch.fcgi"
        params = self._params(
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max(1, int(retmax)),
            }
        )
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return list(data.get("esearchresult", {}).get("idlist", []) or [])

    def efetch_xml(self, pmids: list[str]) -> str:
        if not pmids:
            return ""
        url = f"{self.base_url}/efetch.fcgi"
        params = self._params(
            {
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "abstract",
                "retmode": "xml",
            }
        )
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.text

    def elink_pmc(self, pmid: str) -> str | None:
        url = f"{self.base_url}/elink.fcgi"
        params = self._params(
            {
                "dbfrom": "pubmed",
                "db": "pmc",
                "id": pmid,
                "retmode": "json",
            }
        )
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            linksets = data.get("linksets") or data.get("linkset")  # handle possible variants
            if not linksets:
                return None
            ls = linksets[0]
            dbs = ls.get("linksetdbs") or []
            for db in dbs:
                if db.get("dbto") == "pmc" and db.get("links"):
                    pmcid = db["links"][0].get("id")
                    if pmcid:
                        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
        except Exception as exc:
            logger.debug("ELink PMC lookup failed for %s: %s", pmid, exc)
        return None

    def _text(self, node: ET.Element | None) -> str:
        return (node.text or "").strip() if node is not None else ""

    def _parse_pub_date(self, pub_date_elem: ET.Element | None) -> str:
        if pub_date_elem is None:
            return ""
        year = self._text(pub_date_elem.find("Year"))
        month = self._text(pub_date_elem.find("Month"))
        day = self._text(pub_date_elem.find("Day"))
        if year and month and day:
            # Month can be names; attempt to parse
            try:
                dt = datetime.strptime(f"{year} {month} {day}", "%Y %b %d")
                return dt.date().isoformat()
            except Exception:
                try:
                    dt = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
                    return dt.date().isoformat()
                except Exception:
                    pass
        if year and month:
            try:
                dt = datetime.strptime(f"{year} {month}", "%Y %b")
                return dt.date().isoformat()
            except Exception:
                pass
        return year or ""

    def _join_abstract(self, abstract_elem: ET.Element | None) -> str:
        if abstract_elem is None:
            return ""
        parts: list[str] = []
        for at in abstract_elem.findall("AbstractText"):
            label = at.attrib.get("Label")
            text = (at.text or "").strip()
            if label:
                parts.append(f"{label}: {text}" if text else label)
            else:
                parts.append(text)
        return " ".join([p for p in parts if p])

    def _authors_to_string(self, author_list_elem: ET.Element | None) -> str:
        if author_list_elem is None:
            return ""
        names: list[str] = []
        for author in author_list_elem.findall("Author"):
            last = self._text(author.find("LastName"))
            fore = self._text(author.find("ForeName")) or self._text(author.find("FirstName"))
            collab = self._text(author.find("CollectiveName"))
            if collab:
                names.append(collab)
                continue
            if last and fore:
                names.append(f"{last}, {fore}")
            elif last:
                names.append(last)
        return ", ".join(names)

    def _extract_doi(self, article_id_list: ET.Element | None) -> str:
        if article_id_list is None:
            return ""
        for aid in article_id_list.findall("ArticleId"):
            if aid.attrib.get("IdType") == "doi":
                return (aid.text or "").strip()
        return ""

    def parse_efetch(self, xml_text: str) -> list[dict[str, Any]]:
        if not xml_text:
            return []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("Failed to parse EFetch XML: %s", exc)
            return []
        results: list[dict[str, Any]] = []
        for article in root.findall("PubmedArticle"):
            try:
                medline = article.find("MedlineCitation")
                if medline is None:
                    continue
                pmid = self._text(medline.find("PMID"))
                art = medline.find("Article")
                title = self._text(art.find("ArticleTitle")) if art is not None else ""
                abstract = self._join_abstract(art.find("Abstract") if art is not None else None)
                author_str = self._authors_to_string(art.find("AuthorList") if art is not None else None)
                journal_title = self._text(art.find("Journal/Title")) if art is not None else ""
                pub_date = self._parse_pub_date(art.find("Journal/JournalIssue/PubDate") if art is not None else None)
                doi = self._extract_doi(article.find("PubmedData/ArticleIdList"))

                # Extract publication types
                publication_types = []
                if art is not None:
                    pub_type_list = art.find("PublicationTypeList")
                    if pub_type_list is not None:
                        for pub_type in pub_type_list.findall("PublicationType"):
                            pub_type_text = self._text(pub_type)
                            if pub_type_text:
                                publication_types.append(pub_type_text)

                # full text URL via ELink (best-effort)
                full_url = self.elink_pmc(pmid) or ""
                results.append(
                    {
                        "pmid": pmid,
                        "doi": doi,
                        "title": title,
                        "abstract": abstract,
                        "authors": author_str,
                        "publication_date": pub_date,
                        "journal": journal_title,
                        "publication_types": publication_types,
                        "full_text_url": full_url,
                        "provider": "pubmed_eutils",
                        "provider_family": "ncbi",
                        "ingestion": "eutils",
                    }
                )
            except Exception as exc:
                logger.debug("Skipping malformed article: %s", exc)
        return results

    def search_and_fetch(self, query: str, max_items: int = 30) -> list[dict[str, Any]]:
        pmids = self.esearch(query, retmax=max_items)
        if not pmids:
            return []
        xml_text = self.efetch_xml(pmids)
        return self.parse_efetch(xml_text)
