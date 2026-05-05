import type { ChatMessage } from "../types";
import { MessageBubble } from "./MessageBubble";
import { SuggestedQuestions } from "./SuggestedQuestions";

interface ChatContainerProps {
  messages: ChatMessage[];
  onSuggestedQuestion: (question: string) => void;
}

export function ChatContainer({ messages, onSuggestedQuestion }: ChatContainerProps) {
  const botanicalUrl = new URL("../assets/botanical.svg", import.meta.url).href;

  if (!messages.length) {
    return (
      <section className="empty-state" aria-label="Start a medication question">
        <div>
          <p className="eyebrow">Research-backed medication answers</p>
          <h1>Ask About Your Medications</h1>
          <p>
            Plain-language answers grounded in MedlinePlus information, with safety notes and sources kept close at hand.
          </p>
        </div>
        <img src={botanicalUrl} alt="" />
        <SuggestedQuestions onSelect={onSuggestedQuestion} />
      </section>
    );
  }

  return (
    <section className="chat-thread" aria-live="polite" aria-label="Medication answer conversation">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
    </section>
  );
}
