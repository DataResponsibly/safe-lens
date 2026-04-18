export default function SettingsPanel({ settings, onChange }) {
  const setField = (field) => (e) => {
    const target = e.target;
    const value = target.type === "checkbox" ? target.checked : target.value;
    onChange({ [field]: value });
  };

  const setNumber = (field, { min, max, allowEmpty = false } = {}) => (e) => {
    const raw = e.target.value;
    if (allowEmpty && raw === "") {
      onChange({ [field]: "" });
      return;
    }
    let n = Number(raw);
    if (!Number.isFinite(n)) return;
    if (min !== undefined) n = Math.max(min, n);
    if (max !== undefined) n = Math.min(max, n);
    onChange({ [field]: n });
  };

  return (
    <div className="flex flex-col gap-4 text-sm">
      <Row label="Display uncertainty">
        <input
          type="checkbox"
          className="toggle"
          checked={settings.uncertainty}
          onChange={setField("uncertainty")}
        />
      </Row>

      <Row label="SafeNudge&trade;">
        <input
          type="checkbox"
          className="toggle"
          checked={settings.safenudge}
          onChange={setField("safenudge")}
        />
      </Row>

      <Row label="Random state">
        <input
          type="number"
          min="0"
          step="1"
          placeholder="random"
          value={settings.randomSeed}
          onChange={setField("randomSeed")}
          className="w-24 h-[26px] bg-transparent border border-border px-2 text-fg uppercase tracking-wider"
        />
      </Row>

      <Row label="Max new tokens">
        <input
          type="number"
          min="1"
          max="4096"
          step="1"
          value={settings.maxNewTokens}
          onChange={setNumber("maxNewTokens", { min: 1, max: 4096 })}
          className="w-24 h-[26px] bg-transparent border border-border px-2 text-fg tracking-wider"
        />
      </Row>

      <div>
        <div className="flex items-center justify-between mb-1">
          <span>Delay</span>
          <span className="text-fg/60">{Number(settings.sleepTime).toFixed(1)}s</span>
        </div>
        <div className="relative">
          <div className="absolute top-2 left-2 text-xs text-fg/50 pointer-events-none">
            0s
          </div>
          <div className="absolute top-2 right-2 text-xs text-fg/50 pointer-events-none">
            2s
          </div>
          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={settings.sleepTime}
            onChange={setNumber("sleepTime", { min: 0, max: 2 })}
            className="slider"
          />
        </div>
      </div>

      <details className="border border-border">
        <summary className="px-2 py-1 cursor-pointer uppercase text-xs tracking-wider">
          Advanced
        </summary>
        <div className="p-2 flex flex-col gap-3">
          <Row label="Top-k (k)">
            <input
              type="number"
              min="1"
              max="200"
              step="1"
              value={settings.k}
              onChange={setNumber("k", { min: 1, max: 200 })}
              className="w-24 h-[26px] bg-transparent border border-border px-2 text-fg"
            />
          </Row>
          <Row label="Temperature (T)">
            <input
              type="number"
              min="0.1"
              max="5"
              step="0.1"
              value={settings.T}
              onChange={setNumber("T", { min: 0.1, max: 5 })}
              className="w-24 h-[26px] bg-transparent border border-border px-2 text-fg"
            />
          </Row>
        </div>
      </details>
    </div>
  );
}

function Row({ label, children }) {
  return (
    <label className="flex items-center justify-between gap-3">
      <span
        className="flex-1"
        dangerouslySetInnerHTML={{ __html: label }}
      />
      {children}
    </label>
  );
}
