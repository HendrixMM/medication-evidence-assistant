const QUESTIONS = [
  "What side effects should I watch for with metformin?",
  "Can I take ibuprofen with warfarin?",
  "When is the best time to take levothyroxine?",
  "What should I ask my doctor about statin muscle pain?",
  "Can fluconazole interact with my blood thinner?",
  "What does INR monitoring mean?"
];

interface SuggestedQuestionsProps {
  onSelect: (question: string) => void;
}

export function SuggestedQuestions({ onSelect }: SuggestedQuestionsProps) {
  return (
    <section className="suggested" aria-label="Suggested medication questions">
      {QUESTIONS.map((question) => (
        <button key={question} type="button" onClick={() => onSelect(question)}>
          {question}
        </button>
      ))}
    </section>
  );
}
