export function formatProcessingTime(totalLatencyMs?: number | null): string {
  if (!totalLatencyMs || totalLatencyMs <= 0) {
    return "Answered moments ago";
  }
  if (totalLatencyMs < 1000) {
    return `Answered in ${Math.round(totalLatencyMs)} ms`;
  }
  return `Answered in ${(totalLatencyMs / 1000).toFixed(1)} seconds`;
}

export function formatResetTime(resetAt: string | null): string {
  if (!resetAt) {
    return "";
  }
  const reset = new Date(resetAt);
  if (Number.isNaN(reset.getTime())) {
    return "";
  }
  return reset.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}
