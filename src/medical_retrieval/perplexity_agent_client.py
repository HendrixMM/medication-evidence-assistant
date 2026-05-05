from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
import requests


DEFAULT_AGENT_BASE_URL = "https://api.perplexity.ai"
DEFAULT_AGENT_MODEL = "openai/gpt-5.4-mini"
DEFAULT_AGENT_MAX_STEPS = 6
DEFAULT_AGENT_SEARCH_DOMAINS = ("pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov")
DEFAULT_AGENT_MAX_TOKENS_PER_PAGE = 512
MAX_AGENT_STEPS = 10
KNOWN_FUNCTION_TOOLS = frozenset({"pubmed_normalize_and_fetch", "rxnorm_lookup"})
ToolHandler = Callable[[dict[str, Any]], Any]


PUBMED_DISCOVERY_AGENT_PROMPT = """You are the PubMed Evidence Discovery Agent for a medication-centric retrieval pipeline.

Scope:
- Focus on medications, supplements, ingredients, interactions, adverse effects, contraindications, dosing context, populations, and clinical outcomes named in the input.
- Search PubMed and PMC only. Do not use general web pages, commercial medical pages, forums, news, or guideline pages.
- Use web_search only with the PubMed/PMC domain filters provided by the tool configuration.
- Call pubmed_normalize_and_fetch for PubMed/PMC candidates before returning results.
- Call rxnorm_lookup when medication, supplement, or ingredient names need normalization.
- Do not synthesize a final medical answer, give patient advice, or make clinical recommendations.
- Return structured discovery results only.

Return a JSON object with:
- strategies_executed: array of executed search strategies with strategy_label, query, domains, and result_count.
- candidates: array of candidate PubMed/PMC records with url, snippet, query, and strategy_family.
- recall_rationale: short explanation of recall coverage and remaining gaps.
- sufficient_recall: boolean indicating whether discovery coverage is enough for downstream evidence fetching.
- agent_passes and tool_counts when available.
"""


class PerplexityAgentError(RuntimeError):
    """Expected operational failure from Perplexity Agent discovery."""


class DiscoveryStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_label: str
    query: str
    domains: list[str] = Field(default_factory=list)
    result_count: int = 0


class DiscoveryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    snippet: str | None = None
    query: str
    strategy_family: str


class PerplexityDiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategies_executed: list[DiscoveryStrategy]
    candidates: list[DiscoveryCandidate]
    recall_rationale: str
    sufficient_recall: bool
    agent_passes: int | None = None
    tool_counts: dict[str, int] = Field(default_factory=dict)


def _require_all_object_properties(schema: dict[str, Any]) -> None:
    """Adapt Pydantic JSON Schema to Perplexity Agent strict response_format rules."""
    if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
        schema["required"] = list(schema["properties"].keys())

    for value in schema.values():
        if isinstance(value, dict):
            _require_all_object_properties(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _require_all_object_properties(item)


class PerplexityAgentClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_AGENT_BASE_URL,
        session: requests.Session | None = None,
        timeout: int = 60,
        model: str = DEFAULT_AGENT_MODEL,
        max_steps: int = DEFAULT_AGENT_MAX_STEPS,
        tool_handlers: Mapping[str, ToolHandler] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.model = model
        self.max_steps = self._validate_max_steps(max_steps)
        self.tool_handlers = dict(tool_handlers or {})

    def discover(
        self,
        question: str,
        question_analysis: dict[str, Any] | BaseModel,
        *,
        max_steps: int | None = None,
        tool_handlers: Mapping[str, ToolHandler] | None = None,
    ) -> PerplexityDiscoveryResult:
        steps = self._validate_max_steps(max_steps if max_steps is not None else self.max_steps)
        payload = self._build_payload(question, question_analysis, steps)
        active_handlers = {**self.tool_handlers, **dict(tool_handlers or {})}
        conversation_items: list[dict[str, Any]] = []

        for attempt in range(steps):
            data = self._post_agent(payload)
            self._raise_for_agent_failure(data)

            output_items = self._output_items(data)
            function_calls = self._function_calls(output_items)
            if function_calls:
                if attempt == steps - 1:
                    raise PerplexityAgentError("Perplexity Agent function call loop exceeded max_steps")
                function_outputs = [
                    self._execute_function_call(function_call, active_handlers)
                    for function_call in function_calls
                ]
                conversation_items.extend(function_calls)
                conversation_items.extend(function_outputs)
                payload = {**payload, "input": conversation_items}
                continue

            output_text = self._extract_output_text(data)
            discovery_json = self._parse_output_json(output_text)
            try:
                return PerplexityDiscoveryResult.model_validate(discovery_json)
            except ValidationError as exc:
                raise PerplexityAgentError("Perplexity Agent structured output failed validation") from exc

        raise PerplexityAgentError("Perplexity Agent did not return final discovery output")

    def _post_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}/v1/agent",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise PerplexityAgentError("Perplexity Agent request failed") from exc
        except ValueError as exc:
            raise PerplexityAgentError("Perplexity Agent response was not valid JSON") from exc
        if not isinstance(data, dict):
            raise PerplexityAgentError("Perplexity Agent response JSON must be an object")
        return data

    def _build_payload(
        self,
        question: str,
        question_analysis: dict[str, Any] | BaseModel,
        max_steps: int,
    ) -> dict[str, Any]:
        analysis_payload: Any
        if isinstance(question_analysis, BaseModel):
            analysis_payload = question_analysis.model_dump()
        else:
            analysis_payload = question_analysis

        return {
            "input": json.dumps(
                {
                    "question": question,
                    "question_analysis": analysis_payload,
                },
                sort_keys=True,
            ),
            "instructions": PUBMED_DISCOVERY_AGENT_PROMPT,
            "model": self.model,
            "max_steps": max_steps,
            "stream": False,
            "tools": [self._web_search_tool(), self._pubmed_function_tool(), self._rxnorm_function_tool()],
            "response_format": self._response_format(),
        }

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _web_search_tool(self) -> dict[str, Any]:
        return {
            "type": "web_search",
            "filters": {
                "search_domain_filter": list(DEFAULT_AGENT_SEARCH_DOMAINS),
                "max_tokens_per_page": DEFAULT_AGENT_MAX_TOKENS_PER_PAGE,
            },
        }

    def _response_format(self) -> dict[str, Any]:
        schema = PerplexityDiscoveryResult.model_json_schema()
        properties = schema.get("properties")
        if isinstance(properties, dict):
            properties.pop("agent_passes", None)
            properties.pop("tool_counts", None)
        _require_all_object_properties(schema)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "pubmed_discovery_result",
                "schema": schema,
            },
        }

    def _pubmed_function_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "pubmed_normalize_and_fetch",
            "description": "Normalize PubMed/PMC candidates and fetch canonical article metadata.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "url": {"type": "string"},
                                "snippet": {"type": "string"},
                                "query": {"type": "string"},
                                "strategy_family": {"type": "string"},
                            },
                            "required": ["url", "snippet", "query", "strategy_family"],
                        },
                    }
                },
                "required": ["candidates"],
            },
        }

    def _rxnorm_function_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "rxnorm_lookup",
            "description": "Normalize medication, supplement, and ingredient names before search expansion.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["names"],
            },
        }

    def _validate_max_steps(self, max_steps: int) -> int:
        if not isinstance(max_steps, int):
            raise ValueError("Perplexity Agent max_steps must be an integer")
        if max_steps < 1 or max_steps > MAX_AGENT_STEPS:
            raise ValueError("Perplexity Agent max_steps must be between 1 and 10")
        return max_steps

    def _raise_for_agent_failure(self, data: dict[str, Any]) -> None:
        status = str(data.get("status") or "").lower()
        if status == "failed" or data.get("error"):
            raise PerplexityAgentError("Agent run failed")

    def _function_calls(self, output_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in output_items if item.get("type") == "function_call"]

    def _execute_function_call(
        self,
        function_call: dict[str, Any],
        tool_handlers: dict[str, ToolHandler],
    ) -> dict[str, Any]:
        name = function_call.get("name")
        call_id = function_call.get("call_id")
        if not isinstance(name, str) or name not in KNOWN_FUNCTION_TOOLS:
            raise PerplexityAgentError("Perplexity Agent requested unknown function")
        if name not in tool_handlers:
            raise PerplexityAgentError(f"Perplexity Agent requested {name} but no handler is configured")
        if not isinstance(call_id, str) or not call_id:
            raise PerplexityAgentError("Perplexity Agent function call is missing call_id")

        arguments = self._parse_function_arguments(function_call.get("arguments"))
        try:
            result = tool_handlers[name](arguments)
        except PerplexityAgentError:
            raise
        except requests.RequestException as exc:
            raise PerplexityAgentError(f"Perplexity Agent function handler {name} failed") from exc

        try:
            output = json.dumps(result, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise PerplexityAgentError(
                f"Perplexity Agent function handler {name} returned non-JSON-serializable output"
            ) from exc
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }

    def _parse_function_arguments(self, raw_arguments: Any) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if not isinstance(raw_arguments, str):
            raise PerplexityAgentError("Perplexity Agent function call arguments must be JSON")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise PerplexityAgentError("Perplexity Agent function call arguments were not valid JSON") from exc
        if not isinstance(arguments, dict):
            raise PerplexityAgentError("Perplexity Agent function call arguments must be a JSON object")
        return arguments

    def _extract_output_text(self, data: dict[str, Any]) -> str:
        text_parts: list[str] = []
        for item in self._output_items(data):
            content = item.get("content")
            if isinstance(content, str):
                text_parts.append(content)
                continue
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text") or block.get("output_text")
                if isinstance(text, str):
                    text_parts.append(text)

        output_text = "\n".join(part for part in text_parts if part.strip()).strip()
        if not output_text:
            raise PerplexityAgentError("Perplexity Agent response did not include output text")
        return output_text

    def _output_items(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        output = data.get("output")
        if isinstance(output, list):
            return [item for item in output if isinstance(item, dict)]
        if isinstance(output, dict):
            return [output]

        message = data.get("message")
        if isinstance(message, dict):
            return [message]

        choices = data.get("choices")
        if isinstance(choices, list):
            messages: list[dict[str, Any]] = []
            for choice in choices:
                if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                    messages.append(choice["message"])
            return messages
        return []

    def _parse_output_json(self, output_text: str) -> dict[str, Any]:
        raw_json = self._extract_fenced_json(output_text) or output_text
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise PerplexityAgentError("Perplexity Agent output was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise PerplexityAgentError("Perplexity Agent output JSON must be an object")
        return parsed

    def _extract_fenced_json(self, output_text: str) -> str | None:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", output_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
