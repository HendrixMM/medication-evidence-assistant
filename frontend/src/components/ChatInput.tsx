import { FormEvent, useState } from "react";
import type { RateLimitStatus } from "../types";

interface ChatInputProps {
  disabled: boolean;
  rateLimit: RateLimitStatus | null;
  onSubmit: (question: string) => void;
}

export function ChatInput({ disabled, rateLimit, onSubmit }: ChatInputProps) {
  const [value, setValue] = useState("");
  const isRateLimited = Boolean(
    rateLimit &&
      (rateLimit.remaining_minute <= 0 ||
        (rateLimit.remaining_daily !== null && rateLimit.remaining_daily <= 0))
  );
  const retryAfter = rateLimit?.retry_after_seconds
    ? `${Math.ceil(rateLimit.retry_after_seconds)} seconds`
    : "";

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!value.trim() || disabled || isRateLimited) {
      return;
    }
    onSubmit(value);
    setValue("");
  }

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <label className="visually-hidden" htmlFor="question-input">
        Medication question
      </label>
      <textarea
        id="question-input"
        value={value}
        rows={2}
        onChange={(event) => setValue(event.target.value)}
        placeholder="Ask about side effects, interactions, timing, or what to discuss with your clinician."
        disabled={disabled || isRateLimited}
      />
      <div className="chat-input__footer">
        <span>
          {isRateLimited
            ? `Question limit reached${retryAfter ? ` for ${retryAfter}` : ""}`
            : `${rateLimit?.remaining_minute ?? "-"} questions left this minute`}
        </span>
        <button type="submit" disabled={disabled || isRateLimited || !value.trim()}>
          Ask
        </button>
      </div>
    </form>
  );
}
