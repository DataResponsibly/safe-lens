import { useEffect, useRef } from "react";

function uncertaintyBg(token) {
  const { probs, selected_idx } = token;
  if (!Array.isArray(probs) || selected_idx === undefined) return "transparent";
  const p = probs[selected_idx];
  if (typeof p !== "number") return "transparent";
  const u = Math.max(0, Math.min(1, 1 - Number(p.toFixed(2))));
  return `rgba(231, 76, 60, ${u})`;
}

export default function ChatOutput({
  tokens,
  selectedIdx,
  onSelectToken,
  showUncertainty,
  safenudgeActive,
  errorMessage,
}) {
  const scrollRef = useRef(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    // Keep the view pinned to the bottom as tokens stream in, unless the
    // user has scrolled up manually.
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom < 80) {
      el.scrollTop = el.scrollHeight;
    }
  }, [tokens]);

  const handleClick = (token) => {
    if (safenudgeActive) {
      alert("Probability viewing is not allowed while SafeNudge(TM) is activated.");
      return;
    }
    onSelectToken(token.idx_counter);
  };

  return (
    <div
      ref={scrollRef}
      className="flex-1 min-h-0 overflow-y-auto p-4 sm:p-6 leading-6"
    >
      <div className="inline text-fg/60 mr-1">&gt;</div>
      {tokens.map((token, i) => {
        const selected = token.idx_counter === selectedIdx;
        const bg = showUncertainty ? uncertaintyBg(token) : "transparent";
        return (
          <span
            key={`${token.idx_counter}-${i}`}
            className={"token" + (selected ? " token-selected" : "")}
            style={{ backgroundColor: bg }}
            onClick={() => handleClick(token)}
          >
            {token.selected_text}
          </span>
        );
      })}
      {errorMessage && (
        <div className="mt-4 text-danger text-sm">{errorMessage}</div>
      )}
      {tokens.length === 0 && !errorMessage && (
        <div className="text-fg/40 text-sm mt-2">
          Prompt the model to begin. Tokens appear here and you can tap any
          token to inspect or edit its probability distribution.
        </div>
      )}
    </div>
  );
}
