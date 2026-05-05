import { useCallback, useState } from "react";

import { agentChat } from "../api/agentChat";
import type { AssistantMessage, ChatMessage } from "../types";

function makeId(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`;
}

export function useAgentChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || isLoading) {
        return;
      }

      const nextMessages: ChatMessage[] = [
        ...messages,
        { id: makeId("user"), role: "user", content: trimmed, createdAt: new Date() }
      ];
      setMessages(nextMessages);
      setIsLoading(true);
      setLastError(null);

      try {
        const response = await agentChat(nextMessages);
        const assistantMessage: AssistantMessage = {
          id: makeId("assistant"),
          role: "assistant",
          content: response.message.content,
          createdAt: new Date(),
          meta: response.meta,
          error: null
        };
        setMessages((current) => [...current, assistantMessage]);
      } catch (error) {
        const message = error instanceof Error ? error.message : "The service could not answer right now.";
        setLastError(message);
        setMessages((current) => [
          ...current,
          {
            id: makeId("assistant"),
            role: "assistant",
            content: "",
            createdAt: new Date(),
            meta: null,
            error: message
          }
        ]);
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading, messages]
  );

  const exportConversation = useCallback(() => {
    const lines = ["Medication Questions - Chat History", `Generated: ${new Date().toLocaleString()}`, ""];
    for (const message of messages) {
      lines.push(`${message.role === "user" ? "User" : "Assistant"}: ${message.content}`);
      lines.push("---");
    }

    const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "medication-chat-history.txt";
    anchor.click();
    URL.revokeObjectURL(url);
  }, [messages]);

  return {
    messages,
    isLoading,
    lastError,
    sendMessage,
    exportConversation,
    canExport: messages.length > 0
  };
}
