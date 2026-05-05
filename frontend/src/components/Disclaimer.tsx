interface DisclaimerProps {
  text: string;
}

const FALLBACK =
  "This information is for educational purposes only. Always consult your healthcare provider before making medication decisions.";

export function Disclaimer({ text }: DisclaimerProps) {
  return <footer className="disclaimer">{text || FALLBACK}</footer>;
}
