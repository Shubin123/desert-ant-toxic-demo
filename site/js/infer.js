// Model loading + inference glue. The 100.2 MB toxic.onnx is fetched from
// the public HuggingFace repo at runtime (with download progress), cached
// via the Cache API, and run with the vendored ONNX Runtime Web (wasm EP)
// at an exact pinned version, which reproduces the natively measured scores
// bit-for-float. Inference is 100% on-device: text never leaves the page;
// the only network requests are for the model files.

import { fetchBytes, fetchJSON, createSession } from "./session.js";
import { normalize } from "./normalize.js";
import { Tokenizer } from "./tokenizer.js";

export const HF_BASE = "https://huggingface.co/desert-ant-labs/toxic/resolve/v0.1.0/";
const CACHE_NAME = "toxic-demo-model-v4";
const MODEL_URL = new URL("toxic.onnx", HF_BASE);

async function cachedFetchBytes(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {
    // Cache API unavailable (insecure context / private mode): plain fetch.
  }
  const bytes = await fetchBytes(url, onProgress);
  try {
    await cache?.put(url, new Response(bytes.slice(), {
      headers: { "Content-Type": "application/octet-stream" },
    }));
  } catch {
    // Quota or storage refusal: keep going without the cache.
  }
  return bytes;
}

/**
 * Loads the ONNX runtime, the model (with progress callbacks), and the
 * tokenizer/metadata sidecars. Returns a ready classifier.
 */
export async function loadToxic({ onProgress, onStage } = {}) {
  const stage = (s) => onStage?.(s);

  stage("runtime");
  const runtimeDir = new URL("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/");
  // Threads need cross-origin isolation (serve.py and the Space's headers
  // provide it); numerics are identical either way, threading is latency only.
  const threaded = self.crossOriginIsolated === true;
  const numThreads = threaded
    ? Math.min(4, Math.max(1, (navigator.hardwareConcurrency ?? 2) - 1))
    : 1;

  stage("sidecars");
  const [tokenizerJSON, meta, labels] = await Promise.all([
    fetchJSON(new URL("tokenizer.json", HF_BASE)),
    fetchJSON(new URL("toxic_meta.json", HF_BASE)),
    fetchJSON(new URL("labels.json", HF_BASE)),
  ]);
  const tokenizer = new Tokenizer(tokenizerJSON);

  stage("model");
  const bytes = await cachedFetchBytes(MODEL_URL, onProgress);

  stage("compile");
  const { ort, session } = await createSession(
    { runtime: runtimeDir, model: bytes, numThreads });

  const maxLength = meta.recommended?.max_length ?? 128;
  const contentLabels = Object.values(labels.content.id2label);
  const targetLabels = Object.values(labels.target.id2label);
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  return {
    meta,
    contentLabels,
    targetLabels,
    threaded,
    /** Classify one text. Returns per-label sigmoid probabilities. */
    async classify(text) {
      const normalized = normalize(text);
      const { inputIds, attentionMask } = tokenizer.encode(normalized, maxLength);
      const shape = [1, maxLength];
      const feeds = {
        input_ids: new ort.Tensor("int64", BigInt64Array.from(inputIds, BigInt), shape),
        attention_mask: new ort.Tensor("int64", BigInt64Array.from(attentionMask, BigInt), shape),
      };
      const t0 = performance.now();
      const outputs = await session.run(feeds);
      const ms = performance.now() - t0;
      const content = Array.from(outputs.content_logits.data);
      const target = Array.from(outputs.target_logits.data);
      return {
        content: Object.fromEntries(contentLabels.map((l, i) => [l, sigmoid(content[i])])),
        target: Object.fromEntries(targetLabels.map((l, i) => [l, sigmoid(target[i])])),
        tokens: attentionMask.reduce((n, m) => n + m, 0),
        ms,
      };
    },
  };
}
