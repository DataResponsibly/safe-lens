/**
 * Streaming client for the Safe-Lens FastAPI backend.
 *
 * The backend emits JSON objects delimited by `}\n`, so we buffer bytes
 * from a ReadableStream, split on that boundary, and yield parsed objects.
 * The underlying fetch is cancellable via an AbortController signal, which
 * is how the UI's Stop button interrupts generation.
 */

function toFormBody(params) {
  const form = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    form.append(key, String(value));
  }
  return form;
}

function parseChunk(raw) {
  let s = raw.trim();
  if (!s) return null;
  if (!s.startsWith("{")) s = "{" + s;
  if (!s.endsWith("}")) s = s + "}";
  try {
    return JSON.parse(s);
  } catch (err) {
    console.error("JSON parse error", err, raw);
    return null;
  }
}

export async function* streamEndpoint(url, params, signal) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: toFormBody(params),
    signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Request to ${url} failed: ${res.status} ${text}`);
  }
  if (!res.body) {
    throw new Error("Streaming response body unavailable");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Split on the backend's `}\n` delimiter. The last piece is a partial
      // chunk that we keep in the buffer for the next iteration.
      const parts = buffer.split("}\n");
      buffer = parts.pop() ?? "";

      for (const part of parts) {
        const obj = parseChunk(part);
        if (obj) yield obj;
      }
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      const obj = parseChunk(buffer);
      if (obj) yield obj;
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // ignore
    }
  }
}

export function streamGenerate(params, signal) {
  return streamEndpoint("/generate", params, signal);
}

export function streamRegenerate(params, signal) {
  return streamEndpoint("/regenerate", params, signal);
}
