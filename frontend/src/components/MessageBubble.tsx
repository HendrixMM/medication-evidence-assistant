import type { ChatMessage } from "../types";
import ReactMarkdown from "react-markdown";
import rehypeExternalLinks from "rehype-external-links";
import remarkGfm from "remark-gfm";

import { Disclaimer } from "./Disclaimer";
import { LoadingSkeleton } from "./LoadingSkeleton";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  if (message.role === "user") {
    return (
      <article className="message message--user">
        <p>{message.content}</p>
      </article>
    );
  }

  const isWaiting = !message.content && !message.error;
  const disclaimerText =
    message.content
      .split("\n")
      .map((line) => line.trim())
      .find((line) =>
        line.includes("This information is from published research and is not medical advice.")
      ) ?? null;

  return (
    <article className="message message--assistant">
      {isWaiting ? <LoadingSkeleton /> : null}
      {message.error ? <p className="message__error">{message.error}</p> : null}
      {message.content || disclaimerText ? (
        <div className="answer-reveal">
          {message.content ? (
            <div className="message__answer message__answer--markdown">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[[rehypeExternalLinks, { target: "_blank", rel: ["noopener"] }]]}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          ) : null}
          {disclaimerText !== null ? <Disclaimer text={disclaimerText} /> : null}
        </div>
      ) : null}
    </article>
  );
}
