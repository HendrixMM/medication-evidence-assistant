export interface UsageStats {
  pubmed_calls: number;
  llm_calls: number;
  output_tokens: number;
  total_latency_ms: number;
}

export interface RateLimitStatus {
  requests_last_minute: number;
  max_requests_per_minute: number;
  remaining_minute: number;
  retry_after_seconds: number;
  daily_count: number;
  daily_limit: number | null;
  remaining_daily: number | null;
}

export interface HealthResponse {
  status: string;
  agent_mode_enabled: boolean;
  version: string;
  detail?: string | null;
}

export interface AssistantMeta {
  stop_reason: string;
  usage: UsageStats | null;
  tool_calls: number;
  total_latency_ms: number;
  errors: Array<{ layer: string; reason: string }>;
}

export interface UserMessage {
  id: string;
  role: "user";
  content: string;
  createdAt: Date;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  content: string;
  createdAt: Date;
  meta: AssistantMeta | null;
  error: string | null;
}

export type ChatMessage = UserMessage | AssistantMessage;

export interface AgentChatResponse {
  message: { role: "assistant"; content: string };
  meta: AssistantMeta;
}
