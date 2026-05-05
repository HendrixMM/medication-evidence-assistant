import type { UsageStats } from "../types";

interface SessionCounterProps {
  usage: UsageStats | null;
}

function formatLatency(totalLatencyMs: number): string {
  if (totalLatencyMs < 1000) {
    return `${Math.round(totalLatencyMs)} ms`;
  }
  return `${(totalLatencyMs / 1000).toFixed(1)} s`;
}

export function SessionCounter({ usage }: SessionCounterProps) {
  if (!usage) {
    return <span className="session-counter">Run telemetry appears after the answer</span>;
  }

  const telemetry = [
    `pubmed_calls ${usage.pubmed_calls}`,
    `llm_calls ${usage.llm_calls}`,
    `output_tokens ${usage.output_tokens}`,
    `total_latency_ms ${formatLatency(usage.total_latency_ms)}`
  ];

  return (
    <span className="session-counter" aria-label="Latest answer execution telemetry">
      {telemetry.join(" · ")}
    </span>
  );
}
