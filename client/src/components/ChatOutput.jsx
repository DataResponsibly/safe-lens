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
  prompt,
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
  }, [tokens, prompt]);

  const handleClick = (token) => {
    if (safenudgeActive) {
      alert("Probability viewing is not allowed while SafeNudge is activated.");
      return;
    }
    onSelectToken(token.idx_counter);
  };

  const hasPrompt = !!prompt;
  const hasTokens = tokens.length > 0;

  return (
    <div
      ref={scrollRef}
      className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden p-4 sm:p-6 leading-6 break-words"
    >
      {hasPrompt && (
        <div className="mb-3 select-text">
          <div className="text-[10px] uppercase tracking-wider text-fg/50 mb-1">
            Prompt
          </div>
          <div className="border-l-2 border-fg/30 pl-3 whitespace-pre-wrap break-words text-sm text-fg/80">
            {prompt}
          </div>
        </div>
      )}
      {(hasPrompt || hasTokens) && (
        <div className="text-sm text-fg">
          <div className="text-[10px] uppercase tracking-wider text-fg/50 mb-1">
            Generation
          </div>
          <span className="text-fg/60 mr-1">&gt;</span>
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
        </div>
      )}
      {errorMessage && (
        <div className="mt-4 text-danger text-sm">{errorMessage}</div>
      )}
      {!hasPrompt && !hasTokens && !errorMessage && (
        <div className="text-fg/40 text-sm mt-2">
          Prompt the model to begin. Tokens appear here and you can tap any
          token to inspect or edit its probability distribution.
        </div>
      )}
    </div>
  );
}
