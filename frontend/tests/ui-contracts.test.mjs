import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, it } from "node:test";

const root = new URL("..", import.meta.url).pathname;

function read(path) {
  return readFileSync(join(root, path), "utf8");
}

function readRepo(path) {
  return readFileSync(join(root, "..", path), "utf8");
}

describe("frontend agent chat contracts", () => {
  it("App is rewired to use the non-streaming agent chat hook", () => {
    const app = read("src/App.tsx");

    assert.match(app, /useAgentChat/);
    assert.doesNotMatch(app, /useChat/);
    assert.match(app, /<SessionCounter usage=\{latestAssistantUsage\} \/>/);
    assert.match(app, /<ChatInput disabled=\{isLoading\}/);
  });

  it("agentChat API posts message arrays to /api/agent/chat through apiUrl", () => {
    const agentChat = read("src/api/agentChat.ts");
    const baseUrl = read("src/api/baseUrl.ts");

    assert.match(baseUrl, /VITE_API_BASE_URL/);
    assert.match(agentChat, /apiUrl\("agent\/chat"\)/);
    assert.match(agentChat, /body: JSON\.stringify\(\{ messages: messages\.map/);
    assert.match(agentChat, /return \(await response\.json\(\)\)/);
  });

  it("useAgentChat stores assistant response meta instead of SSE session state", () => {
    const hook = read("src/hooks/useAgentChat.ts");

    assert.match(hook, /content: response\.message\.content/);
    assert.match(hook, /meta: response\.meta/);
    assert.doesNotMatch(hook, /streamChat|getSession|session_id|EventSource/);
  });

  it("MessageBubble renders assistant markdown and omits legacy trust-surface components", () => {
    const messageBubble = read("src/components/MessageBubble.tsx");

    assert.match(messageBubble, /ReactMarkdown/);
    assert.match(messageBubble, /remarkGfm/);
    assert.match(messageBubble, /rehypeExternalLinks/);
    assert.doesNotMatch(messageBubble, /SafetyBanner|SourceList|StatusTimeline|EvidenceLevelBadge/);
  });

  it("frontend types are reduced to chat messages plus response meta", () => {
    const types = read("src/types/index.ts");

    assert.match(types, /export type ChatMessage = UserMessage \| AssistantMessage/);
    assert.match(types, /export interface AssistantMeta/);
    assert.match(types, /usage: UsageStats \| null/);
    assert.doesNotMatch(types, /SessionState|StreamEvent|ConsumerAnswer|EvidenceSummary/);
  });

  it("docker compose forwards the new agent and evidence environment variables", () => {
    const compose = readRepo("docker-compose.yml");

    assert.match(compose, /AGENT_MODEL=/);
    assert.match(compose, /AGENT_MAX_TURNS=/);
    assert.match(compose, /EVIDENCE_QUERY_BUDGET_MAX=/);
    assert.match(compose, /EVIDENCE_MAX_LLM_SCORING_CANDIDATES=/);
    assert.match(compose, /EVIDENCE_CACHE_TTL_SECONDS=/);
  });
});
