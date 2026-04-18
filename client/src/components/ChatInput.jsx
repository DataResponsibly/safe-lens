import { useCallback } from "react";

export default function ChatInput({
  value,
  onChange,
  onSubmit,
  onStop,
  isStreaming,
}) {
  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!isStreaming) onSubmit(value);
      }
    },
    [value, onSubmit, isStreaming]
  );

  const handleClick = useCallback(() => {
    if (isStreaming) {
      onStop();
    } else {
      onSubmit(value);
    }
  }, [isStreaming, onStop, onSubmit, value]);

  return (
    <div
      className="border-t border-border bg-panel p-2 sm:p-3 shrink-0"
      style={{
        paddingBottom: "max(0.5rem, env(safe-area-inset-bottom))",
      }}
    >
      <div className="flex gap-2 items-stretch">
        <textarea
          className="flex-1 min-w-0 resize-none bg-input text-fg border border-border p-2 min-h-[52px] max-h-40 outline-none focus:border-fg/60 text-base sm:text-sm"
          placeholder="What can I help you with?"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={2}
          autoCorrect="off"
          autoCapitalize="sentences"
          spellCheck={false}
        />
        <button
          type="button"
          onClick={handleClick}
          aria-label={isStreaming ? "Stop generation" : "Send prompt"}
          className={
            "shrink-0 w-14 sm:w-16 min-h-[52px] border border-border uppercase tracking-wider text-lg " +
            "flex items-center justify-center touch-manipulation " +
            (isStreaming
              ? "bg-danger/80 hover:bg-danger text-fg"
              : "bg-input hover:bg-panel text-fg")
          }
        >
          {isStreaming ? (
            <span className="block w-4 h-4 bg-fg" aria-hidden="true" />
          ) : (
            <span aria-hidden="true">&#10095;</span>
          )}
        </button>
      </div>
      <div className="mt-1 text-[11px] text-fg/50 hidden sm:block">
        Enter to send · Shift+Enter for newline
      </div>
    </div>
  );
}
