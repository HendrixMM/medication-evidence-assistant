from __future__ import annotations

import time
from copy import deepcopy

import requests

from .normalization_selection import NormalizationCandidate, build_aliases, select_best_candidate
from .schemas import NormalizedDrug

try:
    from cachetools import TTLCache
except Exception:  # pragma: no cover
    TTLCache = None  # type: ignore[assignment]


class RxNormResolver:
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"
    RELATED_INGREDIENT_TTYS = ("IN", "PIN", "MIN")
    BRANDED_TTYS = {"BN", "SBD"}

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        base_url: str | None = None,
        ttl_seconds: int = 86400,
        timeout_seconds: int = 10,
    ) -> None:
        self.session = session or requests.Session()
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.ttl_seconds = ttl_seconds
        self._cache = TTLCache(maxsize=512, ttl=ttl_seconds) if TTLCache is not None else {}
        self._timestamps: dict[str, float] = {}

    def resolve_names(self, names: list[str]) -> list[NormalizedDrug]:
        return [self._to_public_drug(item) for item in self.resolve_names_with_context(names)]

    def resolve_name(self, raw_name: str) -> NormalizedDrug:
        return self._to_public_drug(self.resolve_name_with_context(raw_name))

    def resolve_names_with_context(self, names: list[str]) -> list[NormalizationCandidate]:
        return [self.resolve_name_with_context(name) for name in names]

    def resolve_name_with_context(self, raw_name: str) -> NormalizationCandidate:
        key = raw_name.strip().lower()
        cached = self._get_cache(key)
        if cached is not None:
            return cached

        exact_candidates = self._collect_exact_candidates(raw_name)
        best_exact = select_best_candidate(raw_name, exact_candidates)
        if best_exact is not None:
            resolved = self._candidate_to_context(raw_name, best_exact)
            self._set_cache(key, resolved)
            return resolved

        best_approximate = select_best_candidate(raw_name, self._collect_approximate_candidates(raw_name))
        if best_approximate is not None:
            resolved = self._candidate_to_context(raw_name, best_approximate)
            self._set_cache(key, resolved)
            return resolved

        fallback = NormalizationCandidate(
            raw_name=raw_name,
            canonical_name=raw_name,
            match_type="none",
            rxnorm_cui=None,
            tty=None,
            source="none",
            aliases=build_aliases(raw_name),
            confidence=0.0,
        )
        self._set_cache(key, fallback)
        return fallback

    def _collect_exact_candidates(self, raw_name: str) -> list[dict[str, str | tuple[str, ...] | None]]:
        url = f"{self.base_url}/drugs.json"
        response = self.session.get(url, params={"name": raw_name}, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        concept_groups = payload.get("drugGroup", {}).get("conceptGroup", []) or []
        candidates: list[dict[str, str | tuple[str, ...] | None]] = []
        for group in concept_groups:
            tty = str(group.get("tty") or "")
            for concept in group.get("conceptProperties", []) or []:
                name = concept.get("name")
                rxcui = concept.get("rxcui")
                if name and rxcui:
                    concept_name = str(name)
                    concept_rxcui = str(rxcui)
                    candidates.append(
                        {
                            "name": concept_name,
                            "rxcui": concept_rxcui,
                            "tty": tty,
                            "source": "exact",
                            "aliases": build_aliases(concept_name),
                        }
                    )
                    if self._should_resolve_related_ingredient(tty):
                        related = self._resolve_related_ingredient(concept_rxcui)
                        if related is not None:
                            candidates.append(
                                {
                                    "name": related["name"],
                                    "rxcui": related["rxcui"],
                                    "tty": related.get("tty") or "IN",
                                    "source": "related",
                                    "aliases": build_aliases(concept_name, related["name"]),
                                }
                            )
        return candidates

    def _should_resolve_related_ingredient(self, tty: str) -> bool:
        return tty in self.BRANDED_TTYS

    def _resolve_related_ingredient(self, rxcui: str) -> dict[str, str] | None:
        url = f"{self.base_url}/rxcui/{rxcui}/related.json"
        response = self.session.get(
            url,
            params={"tty": list(self.RELATED_INGREDIENT_TTYS)},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        groups = response.json().get("relatedGroup", {}).get("conceptGroup", []) or []
        for preferred_tty in self.RELATED_INGREDIENT_TTYS:
            for group in groups:
                if str(group.get("tty") or "") != preferred_tty:
                    continue
                for concept in group.get("conceptProperties", []) or []:
                    if concept.get("name") and concept.get("rxcui"):
                        return {
                            "name": str(concept["name"]),
                            "rxcui": str(concept["rxcui"]),
                            "tty": preferred_tty,
                        }
        return None

    def _collect_approximate_candidates(self, raw_name: str) -> list[dict[str, str | tuple[str, ...] | None]]:
        url = f"{self.base_url}/approximateTerm.json"
        response = self.session.get(url, params={"term": raw_name, "maxEntries": 5}, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("approximateGroup", {}).get("candidate", []) or []
        results: list[dict[str, str | tuple[str, ...] | None]] = []
        for candidate in candidates:
            name = str(candidate.get("rxstring") or raw_name)
            rxcui = candidate.get("rxcui")
            results.append(
                {
                    "name": name,
                    "rxcui": str(rxcui) if rxcui else None,
                    "tty": candidate.get("tty"),
                    "source": "approximate",
                    "aliases": build_aliases(name),
                }
            )
        return results

    def _candidate_to_context(self, raw_name: str, candidate: dict[str, object]) -> NormalizationCandidate:
        source = str(candidate.get("source") or "")
        return NormalizationCandidate(
            raw_name=raw_name,
            canonical_name=str(candidate.get("name") or raw_name),
            match_type="approximate" if source == "approximate" else "exact",
            rxnorm_cui=str(candidate["rxcui"]) if candidate.get("rxcui") else None,
            tty=str(candidate["tty"]) if candidate.get("tty") else None,
            source=source or "exact",
            aliases=tuple(str(alias) for alias in candidate.get("aliases", ()) if alias),
            confidence=float(candidate.get("confidence") or 0.0),
        )

    def _to_public_drug(self, candidate: NormalizationCandidate) -> NormalizedDrug:
        return NormalizedDrug(
            raw_name=candidate.raw_name,
            generic_name=candidate.canonical_name,
            match_type=candidate.match_type,
            rxnorm_cui=candidate.rxnorm_cui,
        )

    def _get_cache(self, key: str) -> NormalizationCandidate | None:
        if key not in self._cache:
            return None
        if TTLCache is None:  # pragma: no cover
            created = self._timestamps.get(key, 0.0)
            if time.time() - created > self.ttl_seconds:
                self._cache.pop(key, None)
                self._timestamps.pop(key, None)
                return None
        return deepcopy(self._cache[key])

    def _set_cache(self, key: str, value: NormalizationCandidate) -> None:
        self._cache[key] = deepcopy(value)
        self._timestamps[key] = time.time()
