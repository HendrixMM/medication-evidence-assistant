import { useAgentChat } from "./hooks/useAgentChat";
import { useRateLimit } from "./hooks/useRateLimit";
import { ChatContainer } from "./components/ChatContainer";
import { ChatInput } from "./components/ChatInput";
import { EmergencyRedirect } from "./components/EmergencyRedirect";
import { SessionCounter } from "./components/SessionCounter";
import type { AssistantMessage } from "./types";

const EMERGENCY_KEYWORDS = /\b(overdose|poisoning|chest pain|can't breathe|can’t breathe|cannot breathe|allergic reaction|anaphylaxis|911|emergency)\b/i;

export default function App() {
  const { rateLimit, rateLimitError, refreshRateLimit, triggerCooldown, isRateLimited, retryAfterSeconds } =
    useRateLimit();
  const { messages, isLoading, lastError, sendMessage, exportConversation, canExport } = useAgentChat();

  const chatInputRateLimit = rateLimit
    ? {
        ...rateLimit,
        remaining_minute: isRateLimited ? 0 : rateLimit.remaining_minute,
        retry_after_seconds: retryAfterSeconds > 0 ? retryAfterSeconds : rateLimit.retry_after_seconds
      }
    : isRateLimited
      ? {
          requests_last_minute: 0,
          max_requests_per_minute: 0,
          remaining_minute: 0,
          retry_after_seconds: retryAfterSeconds,
          daily_count: 0,
          daily_limit: null,
          remaining_daily: null
        }
      : null;
  const lastUserMessage = [...messages].reverse().find((message) => message.role === "user");
  const assistantMessages = messages.filter((message): message is AssistantMessage => message.role === "assistant");
  const latestCompletedAssistantMessage = [...assistantMessages]
    .reverse()
    .find((message) => message.role === "assistant" && message.meta?.usage);
  const latestAssistantUsage = latestCompletedAssistantMessage?.meta?.usage ?? null;
  const hasEmergencyQuestion = lastUserMessage ? EMERGENCY_KEYWORDS.test(lastUserMessage.content) : false;

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Medication Questions</p>
          <h1>Patient Medication Guide</h1>
        </div>
        <div className="app-header__actions">
          <button type="button" className="ghost-button" onClick={exportConversation} disabled={!canExport}>
            Export
          </button>
        </div>
      </header>

      {hasEmergencyQuestion ? <EmergencyRedirect /> : null}

      {lastError ? <p className="app-alert">{lastError}</p> : null}
      {rateLimitError ? <p className="app-alert app-alert--muted">{rateLimitError}</p> : null}

      <ChatContainer messages={messages} onSuggestedQuestion={sendMessage} />

      <div className="input-dock">
        <div className="input-dock__meta">
          <SessionCounter usage={latestAssistantUsage} />
        </div>
        <ChatInput disabled={isLoading} rateLimit={chatInputRateLimit} onSubmit={sendMessage} />
      </div>
    </main>
  );
}
