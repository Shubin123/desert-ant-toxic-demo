export async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not load ${url.pathname.split("/").pop()} (HTTP ${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    if (onProgress && total) onProgress(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const part of parts) {
    bytes.set(part, offset);
    offset += part.length;
  }
  return bytes;
}

export async function fetchJSON(url) {
  return JSON.parse(new TextDecoder().decode(await fetchBytes(url)));
}

export async function loadRuntime(baseUrl, { numThreads = 1 } = {}) {
  const ort = await import(new URL("ort.wasm.min.mjs", baseUrl));
  ort.env.wasm.wasmPaths = baseUrl.toString();
  ort.env.wasm.numThreads = numThreads;
  return ort;
}

export async function createSession(manifest, { onProgress } = {}) {
  const { runtime, model, numThreads = 1, sessionOptions = {} } = manifest;
  const ort = runtime && runtime.InferenceSession ? runtime : await loadRuntime(runtime, { numThreads });
  const modelData = model instanceof Uint8Array ? model : await fetchBytes(model, onProgress);
  const session = await ort.InferenceSession.create(modelData, {
    executionProviders: ["wasm"],
    ...sessionOptions,
  });
  return { ort, session, modelData };
}

