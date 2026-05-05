import type { HealthResponse, RateLimitStatus } from "../types";
import { apiUrl } from "./baseUrl";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path));
  if (!response.ok) {
    throw new Error(response.statusText || "Request failed");
  }
  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>("health");
}

export function getRateLimit(): Promise<RateLimitStatus> {
  return getJson<RateLimitStatus>("rate-limit/status");
}
