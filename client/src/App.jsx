import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ChatInput from "./components/ChatInput.jsx";
import ChatOutput from "./components/ChatOutput.jsx";
import SettingsPanel from "./components/SettingsPanel.jsx";
import Barplot from "./components/Barplot.jsx";
import { streamGenerate, streamRegenerate } from "./api/stream.js";

const DEFAULT_SETTINGS = {
  safenudge: false,
  uncertainty: false,
  randomSeed: "",
  sleepTime: 0,
  maxNewTokens: 300,
  k: 10,
  T: 1,
  target: "",
};

function resolveRandomSeed(input) {
  const n = Number(input);
  if (Number.isFinite(n) && input !== "" && input !== null && input !== undefined) {
    return n;
  }
  return Math.floor(Math.random() * 10000) + 1;
}

function resolveTemperature(input) {
  if (input === "" || input === null || input === undefined) return null;
  const n = Number(input);
  if (!Number.isFinite(n)) return null;
  return n > 0 ? n : null;
}

function resolveTopK(input) {
  if (input === "" || input === null || input === undefined) return null;
  const n = Number(input);
  if (!Number.isFinite(n)) return null;
  const k = Math.floor(n);
  return k >= 1 ? k : null;
}

function resolveMaxNewTokens(input) {
  if (input === "" || input === null || input === undefined) return null;
  const n = Number(input);
  if (!Number.isFinite(n)) return null;
  const maxNewTokens = Math.floor(n);
  return maxNewTokens >= 1 ? maxNewTokens : null;
}

export default function App() {
  const [isDark, setIsDark] = useState(() => {
    const saved = localStorage.getItem("theme");
    return saved ? saved === "dark" : false;
  });

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    localStorage.setItem("theme", isDark ? "dark" : "light");
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", isDark ? "#0c0a09" : "#fafaf9");
  }, [isDark]);

  const [prompt, setPrompt] = useState("");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [tokens, setTokens] = useState([]);
  const [submittedPrompt, setSubmittedPrompt] = useState("");
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [chartDrawerOpen, setChartDrawerOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const abortRef = useRef(null);

  const selectedToken = useMemo(() => {
    if (selectedIdx === null) return null;
    return tokens.find((t) => t.idx_counter === selectedIdx) ?? null;
  }, [tokens, selectedIdx]);

  const handleSettingsChange = useCallback((patch) => {
    setSettings((prev) => ({ ...prev, ...patch }));
  }, []);

  const runStream = useCallback(async (streamFn, params) => {
    setErrorMessage("");
    setIsStreaming(true);
    setSelectedIdx(null);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      for await (const obj of streamFn(params, controller.signal)) {
        setTokens((prev) => [...prev, obj]);
      }
    } catch (err) {
      if (err?.name === "AbortError") {
        // Expected when Stop is pressed.
      } else {
        console.error(err);
        setErrorMessage(err?.message || "Generation failed");
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setIsStreaming(false);
    }
  }, []);

  const handleSubmit = useCallback(
    (text) => {
      const trimmed = text.trim();
      if (!trimmed || isStreaming) return;
      const topK = resolveTopK(settings.k);
      if (topK === null) {
        alert("Please enter a top-k value of at least 1 before generating.");
        return;
      }
      const temperature = resolveTemperature(settings.T);
      if (temperature === null) {
        const msg = Number(settings.T) === 0
          ? "Temperature cannot be 0 — use a low value like 0.01 instead."
          : "Please enter a temperature value before generating.";
        alert(msg);
        return;
      }
      const maxNewTokens = resolveMaxNewTokens(settings.maxNewTokens);
      if (maxNewTokens === null) {
        alert("Please enter max new tokens of at least 1 before generating.");
        return;
      }

      setSubmittedPrompt(trimmed);
      setPrompt("");
      setTokens([]);
      setSelectedIdx(null);

      const seed = resolveRandomSeed(settings.randomSeed);
      const params = {
        init_prompt: trimmed,
        safenudge: settings.safenudge,
        k: topK,
        T: temperature,
        max_new_tokens: maxNewTokens,
        verbose: false,
        random_state: seed,
        sleep_time: settings.sleepTime,
        target: settings.target || undefined,
      };
      runStream(streamGenerate, params);
    },
    [isStreaming, runStream, settings]
  );

  const handleStop = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
    }
  }, []);

  const handleClear = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    setTokens([]);
    setSubmittedPrompt("");
    setSelectedIdx(null);
    setErrorMessage("");
  }, []);

  const handleSelectToken = useCallback((idx) => {
    setSelectedIdx(idx);
    // On mobile (where the right-side aside is hidden) the chart drawer is
    // the only place the bar chart can appear, so open it automatically.
    // The drawer itself is `md:hidden`, so this is a no-op on desktop.
    setChartDrawerOpen(true);
  }, []);

  const handleTokenEdit = useCallback(
    (newToken) => {
      if (isStreaming) return;
      if (settings.safenudge) {
        alert("Token editing is not allowed while SafeNudge is activated.");
        return;
      }
      const topK = resolveTopK(settings.k);
      if (topK === null) {
        alert("Please enter a top-k value of at least 1 before regenerating.");
        return;
      }
      const temperature = resolveTemperature(settings.T);
      if (temperature === null) {
        const msg = Number(settings.T) === 0
          ? "Temperature cannot be 0 — use a low value like 0.01 instead."
          : "Please enter a temperature value before regenerating.";
        alert(msg);
        return;
      }
      const maxNewTokens = resolveMaxNewTokens(settings.maxNewTokens);
      if (maxNewTokens === null) {
        alert("Please enter max new tokens of at least 1 before regenerating.");
        return;
      }
      if (selectedIdx === null) return;
      const initPrompt = submittedPrompt;
      if (!initPrompt) return;

      const content = JSON.stringify(tokens);
      const tokenPos = selectedIdx;
      setTokens([]);
      setSelectedIdx(null);

      const seed = resolveRandomSeed(settings.randomSeed);
      const params = {
        init_prompt: initPrompt,
        content,
        token_pos: tokenPos,
        new_token: newToken,
        k: topK,
        T: temperature,
        max_new_tokens: maxNewTokens,
        sleep_time: settings.sleepTime,
        verbose: false,
        random_state: seed,
      };
      runStream(streamRegenerate, params);
    },
    [isStreaming, runStream, selectedIdx, settings, submittedPrompt, tokens]
  );

  const barplotData = useMemo(() => {
    if (!selectedToken) return null;
    const { texts = [], probs = [] } = selectedToken;
    return texts.map((text, i) => ({
      text,
      prob: Number(probs[i]?.toFixed ? probs[i].toFixed(2) : probs[i] ?? 0),
    }));
  }, [selectedToken]);

  const hasTokens = tokens.length > 0;
  const canClear = hasTokens || !!submittedPrompt || !!errorMessage || isStreaming;

  return (
    <div className="h-screen h-[100dvh] w-full flex flex-col overflow-hidden bg-bg text-fg">
      <header
        className="flex items-center justify-between gap-2 border-b border-border px-3 sm:px-4 py-2 shrink-0"
        style={{ paddingTop: "max(0.5rem, env(safe-area-inset-top))" }}
      >
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <span className="text-base sm:text-lg tracking-wider uppercase truncate">
            Safe-Lens
          </span>
          {isStreaming && (
            <span
              aria-label="streaming"
              className="inline-block w-2 h-2 rounded-full bg-accent animate-pulse shrink-0"
            />
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            className="border border-border px-2 min-h-[36px] hover:bg-panel active:bg-panel touch-manipulation"
            onClick={() => setIsDark((d) => !d)}
            title={isDark ? "Switch to light mode" : "Switch to dark mode"}
            aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {isDark ? <SunIcon /> : <MoonIcon />}
          </button>
          <button
            type="button"
            className="border border-border px-3 min-h-[36px] uppercase text-xs tracking-wider hover:bg-panel active:bg-panel disabled:opacity-40 disabled:cursor-not-allowed touch-manipulation"
            onClick={handleClear}
            disabled={!canClear}
            title="Clear generation"
          >
            Clear
          </button>
          <button
            type="button"
            className="hidden md:inline-flex items-center border border-border px-3 min-h-[36px] uppercase text-xs tracking-wider hover:bg-panel active:bg-panel touch-manipulation"
            onClick={() => setSettingsOpen(true)}
            title="Open settings"
          >
            Settings
          </button>
          <button
            type="button"
            className="md:hidden border border-border px-3 min-h-[36px] uppercase text-xs tracking-wider hover:bg-panel active:bg-panel touch-manipulation"
            onClick={() => setChartDrawerOpen(true)}
            title="Show probabilities"
          >
            Chart
          </button>
          <button
            type="button"
            className="md:hidden border border-border px-3 min-h-[36px] uppercase text-xs tracking-wider hover:bg-panel active:bg-panel touch-manipulation"
            onClick={() => setSettingsOpen(true)}
          >
            Settings
          </button>
        </div>
      </header>

      <main className="flex-1 min-h-0 flex flex-col md:flex-row">
        <section className="flex flex-col flex-1 min-h-0 min-w-0 border-border md:border-r">
          <ChatOutput
            prompt={submittedPrompt}
            tokens={tokens}
            selectedIdx={selectedIdx}
            onSelectToken={handleSelectToken}
            showUncertainty={settings.uncertainty}
            safenudgeActive={settings.safenudge}
            errorMessage={errorMessage}
          />
          <ChatInput
            value={prompt}
            onChange={setPrompt}
            onSubmit={handleSubmit}
            onStop={handleStop}
            isStreaming={isStreaming}
          />
        </section>

        <aside className="hidden md:flex md:w-[360px] lg:w-[400px] xl:w-[440px] flex-col min-h-0 shrink-0">
          <DesktopBarplotPanel
            barplotData={barplotData}
            safenudgeActive={settings.safenudge}
            onTokenEdit={handleTokenEdit}
          />
        </aside>
      </main>

      {chartDrawerOpen && (
        <MobileDrawer
          title="Probabilities"
          onClose={() => setChartDrawerOpen(false)}
        >
          <div className="flex-1 min-h-0 p-3">
            <Barplot
              data={barplotData}
              onTickClick={handleTokenEdit}
              disabled={settings.safenudge}
            />
          </div>
        </MobileDrawer>
      )}

      {settingsOpen && (
        <MobileDrawer
          title="Settings"
          onClose={() => setSettingsOpen(false)}
        >
          <div className="flex-1 min-h-0 p-3 overflow-y-auto overscroll-contain text-xs">
            <SettingsPanel
              settings={settings}
              onChange={handleSettingsChange}
            />
          </div>
        </MobileDrawer>
      )}

      {settingsOpen && (
        <DesktopSettingsDrawer onClose={() => setSettingsOpen(false)}>
          <SettingsPanel settings={settings} onChange={handleSettingsChange} />
        </DesktopSettingsDrawer>
      )}
    </div>
  );
}

function DesktopSettingsDrawer({ onClose, children }) {
  return (
    <div className="hidden md:flex fixed inset-0 z-40">
      <div className="flex-1 bg-black/60" onClick={onClose} aria-hidden="true" />
      <div
        className="w-full max-w-sm bg-bg border-l border-border flex flex-col min-h-0 shadow-2xl"
        style={{
          paddingTop: "env(safe-area-inset-top)",
          paddingBottom: "env(safe-area-inset-bottom)",
        }}
      >
        <div className="flex items-center justify-between px-3 py-2 border-b border-border shrink-0">
          <span className="uppercase tracking-wider text-xs text-fg">
            Settings
          </span>
          <button
            type="button"
            className="border border-border px-3 min-h-[36px] text-xs uppercase tracking-wider touch-manipulation"
            onClick={onClose}
          >
            Close
          </button>
        </div>
        <div className="flex-1 min-h-0 p-3 overflow-y-auto overscroll-contain text-xs">
          {children}
        </div>
      </div>
    </div>
  );
}

function MobileDrawer({ title, onClose, children }) {
  return (
    <div className="md:hidden fixed inset-0 z-40 flex">
      <div className="flex-1 bg-black/60" onClick={onClose} />
      <div
        className="w-[90%] max-w-sm bg-bg border-l border-border flex flex-col min-h-0"
        style={{
          paddingTop: "env(safe-area-inset-top)",
          paddingBottom: "env(safe-area-inset-bottom)",
        }}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
          <span className="uppercase tracking-wider text-sm">{title}</span>
          <button
            type="button"
            className="border border-border px-3 min-h-[36px] text-xs uppercase touch-manipulation"
            onClick={onClose}
          >
            Close
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

function SunIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1" x2="12" y2="3" />
      <line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" />
      <line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}

function DesktopBarplotPanel({ barplotData, safenudgeActive, onTokenEdit }) {
  return (
    <div className="flex flex-col h-full min-h-0 p-3 overflow-hidden">
      <Barplot
        data={barplotData}
        onTickClick={onTokenEdit}
        disabled={safenudgeActive}
      />
    </div>
  );
}
