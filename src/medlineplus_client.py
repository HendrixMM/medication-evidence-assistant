"""MedlinePlus Web Service client for patient-facing health topic retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
import html
import re
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests


MEDLINEPLUS_WEB_SERVICE_URL = "https://wsearch.nlm.nih.gov/ws/query"
_CONSUMER_QUERY_SYNONYMS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\btylenol\b", re.IGNORECASE), "acetaminophen"),
    (re.compile(r"\b(?:advil|motrin)\b", re.IGNORECASE), "ibuprofen"),
    (re.compile(r"\baleve\b", re.IGNORECASE), "naproxen"),
    (re.compile(r"\bbenadryl\b", re.IGNORECASE), "diphenhydramine"),
    (re.compile(r"\bclaritin\b", re.IGNORECASE), "loratadine"),
    (re.compile(r"\bzyrtec\b", re.IGNORECASE), "cetirizine"),
)


@dataclass(frozen=True)
class MedlinePlusResult:
    """Normalized MedlinePlus health topic or drug page search result."""

    identifier: str
    title: str
    summary: str
    url: str
    sections: dict[str, list[str]] = field(default_factory=dict)


class MedlinePlusClient:
    """Small wrapper around the NLM MedlinePlus Web Service.

    The service returns XML search results for MedlinePlus health topic records.
    It is intentionally separate from MedlinePlus Connect, which is an EHR code
    lookup service and not the natural-language search source needed here.
    """

    DEFAULT_TOOL = "rag_template_medlineplus"

    def __init__(
        self,
        *,
        base_url: str = MEDLINEPLUS_WEB_SERVICE_URL,
        tool: str = DEFAULT_TOOL,
        email: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url
        self.tool = tool
        self.email = email
        self.timeout = timeout

    def search(self, query: str, *, max_results: int = 5) -> list[MedlinePlusResult]:
        """Search MedlinePlus health topics by natural-language query."""

        normalized_query = query.strip()
        if not normalized_query:
            return []

        params: dict[str, object] = {
            "db": "healthTopics",
            "term": normalized_query,
            "retmax": max(1, int(max_results)),
            "rettype": "all",
            "tool": self.tool,
        }
        if self.email:
            params["email"] = self.email

        response = requests.get(self.base_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return self.parse_response(response.text)

    @staticmethod
    def parse_response(xml_text: str) -> list[MedlinePlusResult]:
        if not xml_text.strip():
            return []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise ValueError(f"MedlinePlus response is not valid XML: {exc}") from exc

        results: list[MedlinePlusResult] = []
        for document in root.findall(".//document"):
            url = str(document.attrib.get("url") or "").strip()
            content_by_name = _content_by_name(document)
            title = _first_content(content_by_name, "title") or url
            summary = (
                _first_content(content_by_name, "FullSummary")
                or _first_content(content_by_name, "fullSummary")
                or _first_content(content_by_name, "snippet")
            )
            sections = {
                name: values
                for name, values in content_by_name.items()
                if name not in {"title", "FullSummary", "fullSummary", "organizationName", "mesh", "altTitle"}
            }
            if not title and not summary and not url:
                continue
            results.append(
                MedlinePlusResult(
                    identifier=_stable_identifier(url=url, title=title, rank=document.attrib.get("rank")),
                    title=title,
                    summary=summary,
                    url=url,
                    sections=sections,
                )
            )
        return results


def _content_by_name(document: ET.Element) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for content in document.findall("content"):
        name = str(content.attrib.get("name") or "").strip()
        if not name:
            continue
        text = _clean_text("".join(content.itertext()))
        if text:
            grouped.setdefault(name, []).append(text)
    return grouped


def _first_content(content_by_name: dict[str, list[str]], name: str) -> str:
    values = content_by_name.get(name) or []
    return values[0] if values else ""


def _clean_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html.unescape(value or ""))
    return re.sub(r"\s+", " ", without_tags).strip()


def _stable_identifier(*, url: str, title: str, rank: str | None) -> str:
    parsed = urlparse(url)
    if parsed.path:
        slug = parsed.path.rsplit("/", 1)[-1].removesuffix(".html")
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if not slug:
        slug = f"result-{rank or '0'}"
    return f"medlineplus:{slug}"


def expand_consumer_query(query: str) -> str:
    """Add common generic medication names for MedlinePlus free-text search."""

    lowered = query.lower()
    synonyms: list[str] = []
    for pattern, synonym in _CONSUMER_QUERY_SYNONYMS:
        if pattern.search(query) and synonym not in lowered:
            synonyms.append(synonym)
    if synonyms:
        expanded_terms = synonyms
        expanded_lowered = " ".join(expanded_terms).lower()
        if "acetaminophen" in expanded_lowered and "ibuprofen" in expanded_lowered:
            expanded_terms.append("pain relievers")
        if "side effect" in lowered:
            expanded_terms.append("side effects")
        return " ".join(expanded_terms).strip()
    return query.strip()


__all__ = ["MEDLINEPLUS_WEB_SERVICE_URL", "MedlinePlusClient", "MedlinePlusResult", "expand_consumer_query"]
