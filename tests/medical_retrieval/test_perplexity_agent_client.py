from __future__ import annotations

import pytest
import requests

from src.medical_retrieval.perplexity_agent_client import (
    DEFAULT_AGENT_SEARCH_DOMAINS,
    PerplexityAgentClient,
    PerplexityAgentError,
)


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse | None = None) -> None:
        self.response = response or FakeResponse(success_payload())
        self.posts: list[dict] = []

    def post(self, url: str, **kwargs):
        self.posts.append({"url": url, **kwargs})
        return self.response


class SequenceSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.posts: list[dict] = []

    def post(self, url: str, **kwargs):
        self.posts.append({"url": url, **kwargs})
        return self.responses.pop(0)


def success_payload() -> dict:
    return {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": """```json
{
  "strategies_executed": [
    {
      "strategy_label": "ingredient-interaction",
      "query": "aspirin warfarin interaction",
      "domains": ["pubmed.ncbi.nlm.nih.gov"],
      "result_count": 2
    }
  ],
  "candidates": [
    {
      "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
      "snippet": "trial snippet",
      "query": "aspirin warfarin interaction",
      "strategy_family": "interaction"
    }
  ],
  "recall_rationale": "One interaction-focused PubMed strategy found candidate articles.",
  "sufficient_recall": true,
  "agent_passes": 2,
  "tool_counts": {"web_search": 1, "pubmed_normalize_and_fetch": 1}
}
```""",
                    }
                ],
            }
        ],
    }


def function_call_payload(
    name: str = "pubmed_normalize_and_fetch",
    arguments: str = """{"candidates": [{"url": "https://pubmed.ncbi.nlm.nih.gov/123/", "snippet": "snippet", "query": "aspirin", "strategy_family": "interaction"}]}""",
) -> dict:
    return {
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "name": name,
                "call_id": "call_1",
                "arguments": arguments,
            }
        ],
    }


def agent_payload_with_output_text(text: str) -> dict:
    return {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                    }
                ],
            }
        ],
    }


def test_discover_posts_to_agent_endpoint_with_auth_without_key_leaking_in_errors() -> None:
    secret = "test-secret-key"
    session = FakeSession(FakeResponse({"status": "failed", "error": {"message": secret}}))
    client = PerplexityAgentClient(api_key=secret, session=session)

    with pytest.raises(PerplexityAgentError) as exc_info:
        client.discover("Does aspirin interact with warfarin?", {"question_type": "interaction"})

    assert session.posts[0]["url"] == "https://api.perplexity.ai/v1/agent"
    assert session.posts[0]["headers"]["Authorization"] == f"Bearer {secret}"
    assert secret not in str(exc_info.value)


def test_discover_payload_includes_pubmed_web_search_constraints() -> None:
    session = FakeSession()
    client = PerplexityAgentClient(api_key="test-key", session=session)

    client.discover("Does aspirin interact with warfarin?", {"question_type": "interaction"})

    payload = session.posts[0]["json"]
    web_search_tool = next(tool for tool in payload["tools"] if tool["type"] == "web_search")
    assert web_search_tool["filters"]["search_domain_filter"] == list(DEFAULT_AGENT_SEARCH_DOMAINS)
    assert web_search_tool["filters"]["max_tokens_per_page"] <= 512
    assert "PubMed" in payload["instructions"]
    assert "PMC" in payload["instructions"]
    assert payload["stream"] is False


def test_discover_payload_uses_discovery_result_json_schema_response_format() -> None:
    session = FakeSession()
    client = PerplexityAgentClient(api_key="test-key", session=session)

    client.discover("Does aspirin interact with warfarin?", {"question_type": "interaction"})

    response_format = session.posts[0]["json"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "pubmed_discovery_result"
    schema = response_format["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) >= {
        "strategies_executed",
        "candidates",
        "recall_rationale",
        "sufficient_recall",
    }
    assert set(schema["required"]) == set(schema["properties"])
    assert "agent_passes" not in schema["properties"]
    assert "tool_counts" not in schema["properties"]
    candidate_schema = schema["$defs"]["DiscoveryCandidate"]
    assert set(candidate_schema["required"]) == set(candidate_schema["properties"])
    assert "snippet" in candidate_schema["required"]


def test_discover_payload_includes_function_tool_schemas() -> None:
    session = FakeSession()
    client = PerplexityAgentClient(api_key="test-key", session=session)

    client.discover("Does aspirin interact with warfarin?", {"question_type": "interaction"})

    tools = {tool["name"]: tool for tool in session.posts[0]["json"]["tools"] if tool["type"] == "function"}
    assert set(tools) == {"pubmed_normalize_and_fetch", "rxnorm_lookup"}

    pubmed_schema = tools["pubmed_normalize_and_fetch"]["parameters"]
    assert pubmed_schema["type"] == "object"
    assert pubmed_schema["properties"]["candidates"]["type"] == "array"
    candidate_properties = pubmed_schema["properties"]["candidates"]["items"]["properties"]
    assert set(candidate_properties) >= {"url", "snippet", "query", "strategy_family"}

    rxnorm_schema = tools["rxnorm_lookup"]["parameters"]
    assert rxnorm_schema["properties"]["names"]["type"] == "array"


def test_discover_rejects_max_steps_above_agent_api_limit_before_network_call() -> None:
    session = FakeSession()
    client = PerplexityAgentClient(api_key="test-key", session=session)

    with pytest.raises(ValueError, match="10"):
        client.discover("Does aspirin interact with warfarin?", {}, max_steps=11)

    assert session.posts == []


def test_discover_parses_fenced_json_output_into_structured_result() -> None:
    client = PerplexityAgentClient(api_key="test-key", session=FakeSession())

    result = client.discover("Does aspirin interact with warfarin?", {"question_type": "interaction"})

    assert result.sufficient_recall is True
    assert result.recall_rationale.startswith("One interaction-focused")
    assert result.agent_passes == 2
    assert result.tool_counts == {"web_search": 1, "pubmed_normalize_and_fetch": 1}
    assert result.strategies_executed[0].strategy_label == "ingredient-interaction"
    assert result.candidates[0].url == "https://pubmed.ncbi.nlm.nih.gov/123/"
    assert result.candidates[0].strategy_family == "interaction"


def test_discover_executes_function_call_and_posts_function_call_output_followup() -> None:
    session = SequenceSession([FakeResponse(function_call_payload()), FakeResponse(success_payload())])
    handler_calls: list[dict] = []

    def normalize_handler(arguments: dict):
        handler_calls.append(arguments)
        return {"normalized": [{"pmid": "123", "url": arguments["candidates"][0]["url"]}]}

    client = PerplexityAgentClient(
        api_key="test-key",
        session=session,
        tool_handlers={"pubmed_normalize_and_fetch": normalize_handler},
    )

    result = client.discover("Does aspirin interact with warfarin?", {"question_type": "interaction"})

    assert result.candidates[0].url == "https://pubmed.ncbi.nlm.nih.gov/123/"
    assert handler_calls == [
        {
            "candidates": [
                {
                    "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
                    "snippet": "snippet",
                    "query": "aspirin",
                    "strategy_family": "interaction",
                }
            ]
        }
    ]
    assert len(session.posts) == 2
    second_input = session.posts[1]["json"]["input"]
    assert second_input[0] == function_call_payload()["output"][0]
    assert second_input[1]["type"] == "function_call_output"
    assert second_input[1]["call_id"] == "call_1"
    assert second_input[1]["output"] == (
        '{"normalized": [{"pmid": "123", "url": "https://pubmed.ncbi.nlm.nih.gov/123/"}]}'
    )


def test_discover_raises_expected_error_for_unknown_function_call() -> None:
    session = SequenceSession([FakeResponse(function_call_payload(name="unknown_tool"))])
    client = PerplexityAgentClient(api_key="test-key", session=session, tool_handlers={})

    with pytest.raises(PerplexityAgentError, match="unknown function"):
        client.discover("Does aspirin interact with warfarin?", {"question_type": "interaction"})


def test_discover_raises_expected_error_for_known_function_without_handler() -> None:
    session = SequenceSession([FakeResponse(function_call_payload())])
    client = PerplexityAgentClient(api_key="test-key", session=session, tool_handlers={})

    with pytest.raises(PerplexityAgentError, match="no handler"):
        client.discover("Does aspirin interact with warfarin?", {"question_type": "interaction"})


def test_discover_wraps_structured_output_validation_errors() -> None:
    session = FakeSession(
        FakeResponse(
            agent_payload_with_output_text(
                """{
  "strategies_executed": [],
  "candidates": [],
  "recall_rationale": "missing sufficient recall"
}"""
            )
        )
    )
    client = PerplexityAgentClient(api_key="test-key", session=session)

    with pytest.raises(PerplexityAgentError, match="structured output"):
        client.discover("Does aspirin interact with warfarin?", {"question_type": "interaction"})


def test_discover_wraps_top_level_extra_fields_in_structured_output() -> None:
    session = FakeSession(
        FakeResponse(
            agent_payload_with_output_text(
                """{
  "strategies_executed": [],
  "candidates": [],
  "recall_rationale": "contains final answer",
  "sufficient_recall": true,
  "final_medical_answer": "Take aspirin only if your doctor says so."
}"""
            )
        )
    )
    client = PerplexityAgentClient(api_key="test-key", session=session)

    with pytest.raises(PerplexityAgentError, match="structured output"):
        client.discover("Does aspirin interact with warfarin?", {"question_type": "interaction"})


def test_discover_wraps_candidate_extra_fields_in_structured_output() -> None:
    session = FakeSession(
        FakeResponse(
            agent_payload_with_output_text(
                """{
  "strategies_executed": [],
  "candidates": [
    {
      "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
      "snippet": "snippet",
      "query": "aspirin",
      "strategy_family": "interaction",
      "final_medical_answer": "unexpected"
    }
  ],
  "recall_rationale": "candidate contains extra data",
  "sufficient_recall": true
}"""
            )
        )
    )
    client = PerplexityAgentClient(api_key="test-key", session=session)

    with pytest.raises(PerplexityAgentError, match="structured output"):
        client.discover("Does aspirin interact with warfarin?", {"question_type": "interaction"})


def test_discover_wraps_failed_agent_status_as_expected_error() -> None:
    session = FakeSession(FakeResponse({"status": "failed", "error": {"message": "agent failed"}}))
    client = PerplexityAgentClient(api_key="test-key", session=session)

    with pytest.raises(PerplexityAgentError, match="Agent run failed"):
        client.discover("Does aspirin interact with warfarin?", {})


def test_discover_wraps_http_timeout_as_expected_error() -> None:
    class TimeoutSession:
        def post(self, url: str, **kwargs):
            raise requests.Timeout("timed out")

    client = PerplexityAgentClient(api_key="test-key", session=TimeoutSession())

    with pytest.raises(PerplexityAgentError, match="request failed"):
        client.discover("Does aspirin interact with warfarin?", {})


def test_unexpected_programming_errors_from_session_propagate() -> None:
    class BrokenSession:
        def post(self, url: str, **kwargs):
            raise AttributeError("fake session bug")

    client = PerplexityAgentClient(api_key="test-key", session=BrokenSession())

    with pytest.raises(AttributeError, match="fake session bug"):
        client.discover("Does aspirin interact with warfarin?", {})
