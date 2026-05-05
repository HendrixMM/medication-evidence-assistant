from __future__ import annotations

import re
from typing import Any

import requests


DEFAULT_PMC_ID_CONVERTER_BASE_URL = (
    "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
)
_PMC_ID_RE = re.compile(r"^PMC\d+$", re.IGNORECASE)


class PMCIDMapper:
    def __init__(
        self,
        base_url: str = DEFAULT_PMC_ID_CONVERTER_BASE_URL,
        session: requests.Session | None = None,
        email: str | None = None,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url
        self.session = session or requests.Session()
        self.email = email
        self.timeout = timeout

    def pmcid_to_pmid(self, pmcid: str) -> str | None:
        normalized_id = self._normalize_pmcid(pmcid)
        return self.pmcids_to_pmids([normalized_id]).get(normalized_id)

    def pmcids_to_pmids(self, pmcids: list[str]) -> dict[str, str]:
        normalized_ids = [self._normalize_pmcid(pmcid) for pmcid in pmcids]
        if not normalized_ids:
            return {}

        params: dict[str, Any] = {
            "ids": ",".join(normalized_ids),
            "idtype": "pmcid",
            "format": "json",
        }
        if self.email:
            params["email"] = self.email

        try:
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            return {}
        except ValueError:
            return {}

        return self._extract_mappings(data, set(normalized_ids))

    def _normalize_pmcid(self, pmcid: str) -> str:
        candidate = pmcid.strip().upper()
        if not _PMC_ID_RE.match(candidate):
            raise ValueError("PMCID must be in PMC1234567 form")
        return candidate

    def _extract_mappings(
        self, data: dict[str, Any], requested_ids: set[str]
    ) -> dict[str, str]:
        records = data.get("records") or []
        mappings: dict[str, str] = {}

        for record in records:
            pmcid = record.get("pmcid")
            pmid = record.get("pmid")
            if not pmcid or not pmid:
                continue

            normalized_pmcid = self._normalize_pmcid(str(pmcid))
            if normalized_pmcid in requested_ids:
                mappings[normalized_pmcid] = str(pmid)

        return mappings
