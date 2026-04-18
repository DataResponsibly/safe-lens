import { useCallback, useMemo, useRef, useState } from "react";
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
  k: 20,
  T: 1.3,
};

function resolveRandomSeed(input) {
  const n = Number(input);
  if (Number.isFinite(n) && input !== "" && input !== null && input !== undefined) {
    return n;
  }
  return Math.floor(Math.random() * 10000) + 1;
}

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [tokens, setTokens] = useState([]);
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [drawerOpen, setDrawerOpen] = useState(false);

  const abortRef = useRef(null);
  const lastPromptRef = useRef("");

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

      lastPromptRef.current = trimmed;
      setPrompt("");
      setTokens([]);
      setSelectedIdx(null);

      const seed = resolveRandomSeed(settings.randomSeed);
      const params = {
        init_prompt: trimmed,
        safenudge: settings.safenudge,
        k: settings.k,
        T: settings.T,
        max_new_tokens: settings.maxNewTokens,
        verbose: false,
        random_state: seed,
        sleep_time: settings.sleepTime,
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
    setSelectedIdx(null);
    setErrorMessage("");
  }, []);

  const handleTokenEdit = useCallback(
    (newToken) => {
      if (isStreaming) return;
      if (settings.safenudge) {
        alert("Token editing is not allowed while SafeNudge(TM) is activated.");
        return;
      }
      if (selectedIdx === null) return;
      const initPrompt = lastPromptRef.current;
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
        k: settings.k,
        T: settings.T,
        max_new_tokens: settings.maxNewTokens,
        sleep_time: settings.sleepTime,
        verbose: true,
        random_state: seed,
      };
      runStream(streamRegenerate, params);
    },
    [isStreaming, runStream, selectedIdx, settings, tokens]
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
            className="border border-border px-3 min-h-[36px] uppercase text-xs tracking-wider hover:bg-panel active:bg-panel disabled:opacity-40 disabled:cursor-not-allowed touch-manipulation"
            onClick={handleClear}
            disabled={!hasTokens && !errorMessage && !isStreaming}
            title="Clear generation"
          >
            Clear
          </button>
          <button
            type="button"
            className="md:hidden border border-border px-3 min-h-[36px] uppercase text-xs tracking-wider hover:bg-panel active:bg-panel touch-manipulation"
            onClick={() => setDrawerOpen(true)}
          >
            Settings
          </button>
        </div>
      </header>

      <main className="flex-1 min-h-0 flex flex-col md:flex-row">
        <section className="flex flex-col flex-1 min-h-0 min-w-0 border-border md:border-r">
          <ChatOutput
            tokens={tokens}
            selectedIdx={selectedIdx}
            onSelectToken={setSelectedIdx}
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

        <aside className="hidden md:flex md:w-[340px] lg:w-[380px] flex-col min-h-0">
          <SettingsPanelContent
            settings={settings}
            onChange={handleSettingsChange}
            barplotData={barplotData}
            safenudgeActive={settings.safenudge}
            onTokenEdit={handleTokenEdit}
          />
        </aside>
      </main>

      {drawerOpen && (
        <div className="md:hidden fixed inset-0 z-40 flex">
          <div
            className="flex-1 bg-black/60"
            onClick={() => setDrawerOpen(false)}
          />
          <div
            className="w-[90%] max-w-sm bg-bg border-l border-border flex flex-col min-h-0"
            style={{
              paddingTop: "env(safe-area-inset-top)",
              paddingBottom: "env(safe-area-inset-bottom)",
            }}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
              <span className="uppercase tracking-wider text-sm">Settings</span>
              <button
                type="button"
                className="border border-border px-3 min-h-[36px] text-xs uppercase touch-manipulation"
                onClick={() => setDrawerOpen(false)}
              >
                Close
              </button>
            </div>
            <SettingsPanelContent
              settings={settings}
              onChange={handleSettingsChange}
              barplotData={barplotData}
              safenudgeActive={settings.safenudge}
              onTokenEdit={handleTokenEdit}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function SettingsPanelContent({
  settings,
  onChange,
  barplotData,
  safenudgeActive,
  onTokenEdit,
}) {
  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="h-[42%] min-h-[240px] max-h-[460px] shrink-0 border-b border-border p-3 overflow-hidden">
        <Barplot
          data={barplotData}
          onTickClick={onTokenEdit}
          disabled={safenudgeActive}
        />
      </div>
      <div className="flex-1 min-h-0 p-3 overflow-y-auto overscroll-contain">
        <SettingsPanel settings={settings} onChange={onChange} />
      </div>
    </div>
  );
}
