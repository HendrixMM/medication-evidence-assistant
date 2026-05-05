from __future__ import annotations

from src.evidence.rxnorm_resolver import RxNormResolver


class StubResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class StubSession:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = [StubResponse(item) for item in responses]
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict, timeout: int):
        self.calls.append((url, params))
        return self._responses.pop(0)


def test_rxnorm_resolver_uses_approximate_match_when_exact_missing() -> None:
    session = StubSession(
        [
            {"drugGroup": {"conceptGroup": []}},
            {"approximateGroup": {"candidate": [{"rxcui": "123", "rxstring": "acetaminophen extended release"}]}},
        ]
    )
    resolver = RxNormResolver(session=session)

    result = resolver.resolve_names(["acetaminophen"])

    assert result[0].match_type == "approximate"
    assert result[0].generic_name == "acetaminophen extended release"
    assert len(session.calls) == 2


def test_rxnorm_resolver_rejects_unrelated_exact_match_for_ibuprofen() -> None:
    session = StubSession(
        [
            {
                "drugGroup": {
                    "conceptGroup": [
                        {"tty": "IN", "conceptProperties": [{"rxcui": "4278", "name": "famotidine"}]}
                    ]
                }
            },
            {"approximateGroup": {"candidate": [{"rxcui": "5640", "rxstring": "ibuprofen"}]}},
        ]
    )
    resolver = RxNormResolver(session=session)

    result = resolver.resolve_names(["ibuprofen"])

    assert result[0].generic_name == "ibuprofen"
    assert result[0].match_type == "approximate"


def test_rxnorm_resolver_rejects_dose_form_exact_match_for_creatine() -> None:
    session = StubSession(
        [
            {
                "drugGroup": {
                    "conceptGroup": [
                        {"tty": "SCD", "conceptProperties": [{"rxcui": "2722959", "name": "creatine 1000 MG Chewable Tablet"}]}
                    ]
                }
            },
            {"approximateGroup": {"candidate": []}},
        ]
    )
    resolver = RxNormResolver(session=session)

    result = resolver.resolve_names(["creatine"])

    assert result[0].generic_name == "creatine"
    assert result[0].match_type == "none"


def test_rxnorm_resolver_rejects_unrelated_approximate_match_without_lexical_overlap() -> None:
    session = StubSession(
        [
            {"drugGroup": {"conceptGroup": []}},
            {"approximateGroup": {"candidate": [{"rxcui": "4278", "rxstring": "famotidine", "tty": "IN"}]}},
        ]
    )
    resolver = RxNormResolver(session=session)

    result = resolver.resolve_names(["Tylenol"])

    assert result[0].generic_name == "Tylenol"
    assert result[0].match_type == "none"


def test_rxnorm_resolver_returns_none_match_when_unresolved() -> None:
    session = StubSession([{"drugGroup": {"conceptGroup": []}}, {"approximateGroup": {"candidate": []}}])
    resolver = RxNormResolver(session=session)

    result = resolver.resolve_names(["creatine"])

    assert result[0].match_type == "none"
    assert result[0].generic_name == "creatine"


def test_rxnorm_resolver_skips_related_back_resolution_for_non_brand_exact_match() -> None:
    session = StubSession(
        [
            {
                "drugGroup": {
                    "conceptGroup": [
                        {
                            "tty": "IN",
                            "conceptProperties": [{"rxcui": "161", "name": "acetaminophen"}],
                        }
                    ]
                }
            }
        ]
    )
    resolver = RxNormResolver(session=session)

    result = resolver.resolve_names(["acetaminophen"])

    assert result[0].generic_name == "acetaminophen"
    assert result[0].match_type == "exact"
    assert result[0].rxnorm_cui == "161"
    assert len(session.calls) == 1


def test_rxnorm_resolver_reduces_brand_name_to_ingredient() -> None:
    session = StubSession(
        [
            {
                "drugGroup": {
                    "conceptGroup": [
                        {
                            "tty": "SBD",
                            "conceptProperties": [
                                {
                                    "rxcui": "1243440",
                                    "name": "8 HR acetaminophen 650 MG Extended Release Oral Tablet [Tylenol]",
                                }
                            ],
                        }
                    ]
                }
            },
            {
                "relatedGroup": {
                    "conceptGroup": [
                        {
                            "tty": "IN",
                            "conceptProperties": [
                                {"rxcui": "161", "name": "acetaminophen"}
                            ],
                        }
                    ]
                }
            },
        ]
    )
    resolver = RxNormResolver(session=session)

    result = resolver.resolve_names(["Tylenol"])

    assert result[0].generic_name == "acetaminophen"
    assert result[0].match_type == "exact"
    assert result[0].rxnorm_cui == "161"
    assert session.calls[1][1] == {"tty": ["IN", "PIN", "MIN"]}


def test_rxnorm_resolver_prefers_in_related_concept_over_other_types() -> None:
    session = StubSession(
        [
            {
                "drugGroup": {
                    "conceptGroup": [
                        {
                            "tty": "SBD",
                            "conceptProperties": [
                                {
                                    "rxcui": "1243440",
                                    "name": "8 HR acetaminophen 650 MG Extended Release Oral Tablet [Tylenol]",
                                }
                            ],
                        }
                    ]
                }
            },
            {
                "relatedGroup": {
                    "conceptGroup": [
                        {
                            "tty": "PIN",
                            "conceptProperties": [
                                {"rxcui": "999", "name": "acetaminophen pin"}
                            ],
                        },
                        {
                            "tty": "IN",
                            "conceptProperties": [
                                {"rxcui": "161", "name": "acetaminophen"}
                            ],
                        },
                    ]
                }
            },
        ]
    )
    resolver = RxNormResolver(session=session)

    result = resolver.resolve_names(["Tylenol"])

    assert result[0].generic_name == "acetaminophen"
    assert result[0].match_type == "exact"
    assert result[0].rxnorm_cui == "161"
