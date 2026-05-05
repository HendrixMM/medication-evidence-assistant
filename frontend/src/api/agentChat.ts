import { apiUrl } from "./baseUrl";
import type { AgentChatResponse, ChatMessage } from "../types";

export async function agentChat(messages: ChatMessage[]): Promise<AgentChatResponse> {
  const response = await fetch(apiUrl("agent/chat"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: messages.map(({ role, content }) => ({ role, content })) })
  });

  if (!response.ok) {
    let message = "The service could not answer right now.";
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        message = payload.detail;
      }
    } catch {
      message = response.statusText || message;
    }
    throw new Error(message);
  }

  return (await response.json()) as AgentChatResponse;
}
