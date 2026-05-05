from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlparse


_PUBMED_HOSTS = {"pubmed.ncbi.nlm.nih.gov"}
_PMC_ID_RE = re.compile(r"^PMC\d+$", re.IGNORECASE)


@dataclass
class NormalizedSource:
    source_url: str
    pmid: str | None = None
    pmcid: str | None = None
    is_supported: bool = False

    @property
    def dedupe_key(self) -> str | None:
        if self.pmid:
            return f"pmid:{self.pmid}"
        if self.pmcid:
            return f"pmcid:{self.pmcid}"
        return None


def normalize_source_url(url: str) -> NormalizedSource:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    raw_path_parts = parsed.path.split("/")
    path_parts = [part for part in raw_path_parts if part]

    if host in _PUBMED_HOSTS and path_parts:
        candidate = path_parts[0]
        if candidate.isdigit():
            return NormalizedSource(source_url=url, pmid=candidate, is_supported=True)

    if host == "pmc.ncbi.nlm.nih.gov":
        pmcid = _extract_pmcid(raw_path_parts)
        if pmcid:
            return NormalizedSource(source_url=url, pmcid=pmcid, is_supported=True)

    if host == "www.ncbi.nlm.nih.gov":
        pmcid = _extract_legacy_www_pmcid(path_parts)
        if pmcid:
            return NormalizedSource(source_url=url, pmcid=pmcid, is_supported=True)

    return NormalizedSource(source_url=url)


def _extract_pmcid(path_parts: list[str]) -> str | None:
    if (
        len(path_parts) == 3
        and path_parts[0] == ""
        and path_parts[1].lower() == "articles"
    ):
        candidate = path_parts[2]
        if _PMC_ID_RE.match(candidate):
            return candidate.upper()
    if (
        len(path_parts) == 4
        and path_parts[0] == ""
        and path_parts[1].lower() == "articles"
        and path_parts[3] == ""
    ):
        candidate = path_parts[2]
        if _PMC_ID_RE.match(candidate):
            return candidate.upper()
    return None


def _extract_legacy_www_pmcid(path_parts: list[str]) -> str | None:
    if (
        len(path_parts) >= 3
        and path_parts[0].lower() == "pmc"
        and path_parts[1].lower() == "articles"
    ):
        candidate = path_parts[2]
        if _PMC_ID_RE.match(candidate):
            return candidate.upper()
    return None
