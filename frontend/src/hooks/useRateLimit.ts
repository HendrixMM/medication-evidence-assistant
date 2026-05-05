import { useCallback, useEffect, useRef, useState } from "react";
import { getRateLimit } from "../api/client";
import type { RateLimitStatus } from "../types";

export function useRateLimit() {
  const [rateLimit, setRateLimit] = useState<RateLimitStatus | null>(null);
  const [rateLimitError, setRateLimitError] = useState<string | null>(null);
  const [cooldownUntil, setCooldownUntil] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const timeoutRef = useRef<number | null>(null);
  const intervalRef = useRef<number | null>(null);

  const refreshRateLimit = useCallback(async () => {
    try {
      const status = await getRateLimit();
      setRateLimit(status);
      setRateLimitError(null);
    } catch {
      setRateLimitError("Question limit is temporarily unavailable.");
    }
  }, []);

  const clearCooldownTimers = useCallback(() => {
    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const triggerCooldown = useCallback(
    (seconds: number) => {
      const durationMs = Math.max(0, seconds * 1000);
      const nextCooldownUntil = Date.now() + durationMs;

      clearCooldownTimers();
      setNow(Date.now());
      setCooldownUntil(nextCooldownUntil);

      intervalRef.current = window.setInterval(() => {
        setNow(Date.now());
      }, 1000);

      timeoutRef.current = window.setTimeout(() => {
        clearCooldownTimers();
        setCooldownUntil(null);
        setNow(Date.now());
        void refreshRateLimit();
      }, durationMs);
    },
    [clearCooldownTimers, refreshRateLimit]
  );

  useEffect(() => {
    void refreshRateLimit();
  }, [refreshRateLimit]);

  useEffect(() => () => clearCooldownTimers(), [clearCooldownTimers]);

  const cooldownActive = cooldownUntil !== null && cooldownUntil > now;
  const isRateLimited = cooldownActive || Boolean(rateLimit && rateLimit.remaining_minute <= 0);
  const retryAfterSeconds = cooldownActive ? Math.max(0, Math.ceil((cooldownUntil - now) / 1000)) : 0;

  return {
    rateLimit,
    rateLimitError,
    refreshRateLimit,
    triggerCooldown,
    isRateLimited,
    retryAfterSeconds
  };
}
